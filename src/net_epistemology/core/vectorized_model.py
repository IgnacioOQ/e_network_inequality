from net_epistemology.utils.imports import *
from .vectorized_agents import VectorizedBandit
from scipy.stats import beta


class VectorizedModel:
    """
    Vectorized version of the Model class.
    """

    def __init__(
        self,
        network,
        n_experiments: int,
        agent_type: str = "beta",
        uncertainty: float = 0.001,
        tolerance=5 * 1e-03,
        histories=False,
        sampling_update=False,
        variance_stopping=False,
        tstep_stopping=True,
        directed_network=True,
        seed=None,
        seeded=False,
        compute_convergence=False,
        compute_root_analysis=False,
        *args,
        **kwargs
    ):
        self.network = network
        self.n_agents = len(network.nodes)
        self.nodes = list(self.network.nodes)
        self.id_to_index_map = {u: index for index, u in enumerate(self.nodes)}

        self.n_experiments = n_experiments
        self.agent_type = agent_type
        self.histories = histories
        self.sampling_update = sampling_update
        self.epsilon = 0  # As per original code default
        self.tolerance = tolerance
        self.tstep_stopping = tstep_stopping
        self.compute_convergence = compute_convergence
        self.compute_root_analysis = compute_root_analysis
        
        # Convergence tracking attributes (populated after simulation if compute_convergence=True)
        self.credences_prior_final = None  # Beliefs at step N-1
        self.belief_change_abs = None  # Per-agent, per-theory absolute difference (final step)
        self.belief_change_kl = None  # Per-agent, per-theory KL divergence (final step)
        self.belief_change_history = None  # Per-step mean belief change: list of (mean_T0, mean_T1)
        
        # Root analysis attributes (populated after simulation if compute_root_analysis=True)
        self.root_analysis = None
        if seeded:
            if seed is None:
                seed = np.random.randint(0, 2**32 - 1)
            rd.seed(seed)

        self.bandit = VectorizedBandit(uncertainty, self.n_agents)

        # --- 1. Vectorized State Initialization ---
        # Original: BetaAgent init calls:
        # prior_T1 = rd.uniform(0, 4, size=2)
        # prior_T2 = rd.uniform(0, 4, size=2)
        # self.alphas_betas = np.array([prior_T1, prior_T2])

        # To match the sequence of random numbers:
        # Loop over agents to initialize state (only for init, which is once).
        # This ensures the seed produces the exact same initial state.

        self.alphas_betas = np.zeros((self.n_agents, 2, 2))
        self.credences = (
            np.zeros((self.n_agents, 2))
            if agent_type == "beta"
            else np.zeros(self.n_agents)
        )

        if self.agent_type == "beta":
            for i in range(self.n_agents):
                prior_T1 = rd.uniform(0, 4, size=2)
                prior_T2 = rd.uniform(0, 4, size=2)
                self.alphas_betas[i] = np.array([prior_T1, prior_T2])

                mean_T1 = beta.stats(prior_T1[0], prior_T1[1], moments="m")
                mean_T2 = beta.stats(prior_T2[0], prior_T2[1], moments="m")

                if self.sampling_update:
                    self.credences[i] = np.array(
                        [
                            rd.beta(prior_T1[0], prior_T1[1], size=1)[0],
                            rd.beta(prior_T2[0], prior_T2[1], size=1)[0],
                        ]
                    )
                else:
                    self.credences[i] = np.array([mean_T1, mean_T2])

        elif self.agent_type == "bayes":
            self.credences = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                self.credences[i] = rd.uniform(0, 1)

        # History
        if self.histories:
            self.credences_history = [[] for _ in range(self.n_agents)]
            for i in range(self.n_agents):
                self.credences_history[i].append(self.credences[i])

        # --- 2. Vectorized Graph ---
        # We need the adjacency matrix.
        # Logic: If A -> B, A is predecessor of B. B observes A.
        # We need to sum over predecessors.
        # Adjacency matrix A_adj: A_adj[u, v] = 1 if u -> v.
        # Predecessors of v are u such that A_adj[u, v] = 1.
        # We want to aggregate info from u to v.
        # Result[v] = Sum(Outcomes[u]) for u in Pred(v).
        # Result[v] = Sum_u (A_adj[u, v] * Outcomes[u]).
        # Vector form: Result = A_adj.T @ Outcomes.

        self.adj_matrix = nx.to_numpy_array(self.network, nodelist=self.nodes)
        if not directed_network:
            # If undirected, A_adj is symmetric, so A.T == A.
            pass

        # Convert to sparse if large? For now dense is fine for 100 agents.

        self.n_steps = 0
        self.conclusion = 0.0
        self.conclusion_core = 0.0
        self.proportion_reached_by_truth = 0.0

    def step(self):
        self.n_steps += 1

        # --- Experiment Step ---
        # 1. Choices (Epsilon-Greedy)
        # Vectorized choice
        # Random numbers for epsilon check

        theory_indices = np.zeros(self.n_agents, dtype=int)

        if self.agent_type == "beta":
            rand_epsilon = rd.rand(self.n_agents)
            explore_mask = rand_epsilon < self.epsilon
            exploit_mask = ~explore_mask

            # Explore: Random choice
            if np.any(explore_mask):
                theory_indices[explore_mask] = rd.randint(
                    0, 2, size=np.sum(explore_mask)
                )

            # Exploit: Best credence
            # Break ties randomly? Original: rd.choice(max_indices).
            # np.argmax always takes the first one.
            # To break ties randomly in vector: add small random noise?
            # Or simpler:
            if np.any(exploit_mask):
                # credences shape (N, 2).
                # To handle ties properly vectorized is hard.
                # But floating point equality is rare unless initialized same.
                # Standard argmax:
                theory_indices[exploit_mask] = np.argmax(
                    self.credences[exploit_mask], axis=1
                )

        elif self.agent_type == "bayes":
            # Bayes Agent Choice: if cred > 0.5 -> 1 (Good), else 0 (Bad)
            # self.credences is shape (N,)
            theory_indices = (self.credences > 0.5).astype(int)

        # 2. Run Experiments
        # BayesAgent logic in agents.py: if choice is 0, return 0,0,0.
        # VectorizedBandit.experiment returns n_success, n_total.
        # We need to mask out agents who chose 0.
        # But wait, agents.py BayesAgent.experiment:
        # if choice() == 0: return 0, 0, 0 (index, success, failure)
        # VectorizedBandit returns results based on index.
        # If index is 0 (Bad), p_bad = 0.5. It runs experiment.
        # But BayesAgent DOES NOT run experiment if choice is 0.
        # So for Bayes, we must set success/failure to 0 if choice is 0.

        if self.agent_type == "bayes":
            # Only run experiments for those who chose 1
            # But VectorizedBandit runs for all.
            # We can run for all and then mask.
            n_success, n_total = self.bandit.experiment(
                theory_indices, self.n_experiments
            )
            n_failures = n_total - n_success

            # Mask
            mask_0 = theory_indices == 0
            n_success[mask_0] = 0
            n_failures[mask_0] = 0
            # n_total[mask_0] = 0 # Not strictly needed but cleaner

        else:  # Beta
            n_success, n_total = self.bandit.experiment(
                theory_indices, self.n_experiments
            )
            n_failures = n_total - n_success

        # Store results for update
        # experiments_results: matrix of shape (N, 2, 2) -> (success, failure) for theory 0 and 1?
        # Actually, each agent only tested ONE theory.
        # So agent i has result for theory T_i.
        # We need to construct the update vector.

        # --- Update Step ---
        # We need to aggregate results from neighbors.
        # For each theory T (0 or 1):
        #   Identify agents who tested T.
        #   Their Successes S_T and Failures F_T.
        #   Broadcast these to their successors.

        # Let's create Outcome Matrices:
        # Outcome_Success[i, t] = successes of agent i on theory t (0 otherwise)
        # Outcome_Failure[i, t] = failures of agent i on theory t

        outcome_success = np.zeros((self.n_agents, 2))
        outcome_failure = np.zeros((self.n_agents, 2))

        # Fill them
        # theory_indices is (N,). n_success is (N,).
        rows = np.arange(self.n_agents)
        outcome_success[rows, theory_indices] = n_success
        outcome_failure[rows, theory_indices] = n_failures

        # Aggregate from neighbors
        # Aggregated_Success = A.T @ Outcome_Success
        # Shape: (N, N) @ (N, 2) -> (N, 2)

        # Note on self.directedness in original model:
        # if directed: predecessors.
        # if undirected: neighbors.
        # nx.to_numpy_array handles this (if undirected, symmetric).
        # And we established A.T sums over predecessors (incoming edges).
        # For undirected, A.T = A, sums over neighbors. Correct.

        agg_success = self.adj_matrix.T @ outcome_success
        agg_failure = self.adj_matrix.T @ outcome_failure

        # Add OWN results?
        # Original code:
        # theories_exp_results[theory_index] += results (own)
        # for id in neighbors: ... += results (neighbor)
        # So YES, add own results.

        total_success = agg_success + outcome_success
        total_failure = agg_failure + outcome_failure

        if self.agent_type == "beta":
            # Update Alphas/Betas
            # self.alphas_betas shape (N, 2, 2) -> (Agent, Theory, Param)
            # Param 0 is alpha, Param 1 is beta.

            # total_success is (N, 2).
            self.alphas_betas[:, :, 0] += total_success
            self.alphas_betas[:, :, 1] += total_failure

            # Update Credences
            # New means.
            # alpha = self.alphas_betas[:, :, 0]
            # beta_param = self.alphas_betas[:, :, 1]

            # Estimate = alpha / (alpha + beta) (Mean)
            # Or sampling.

            if self.sampling_update:
                # Vectorized sampling from beta?
                # np.random.beta(a, b) works with arrays.
                self.credences = rd.beta(
                    self.alphas_betas[:, :, 0], self.alphas_betas[:, :, 1]
                )
            else:
                # Mean
                # beta.stats(moment='m') is basically a / (a+b) for beta dist.
                a = self.alphas_betas[:, :, 0]
                b = self.alphas_betas[:, :, 1]
                self.credences = a / (a + b)

        elif self.agent_type == "bayes":
            # Bayes Update Logic
            # Only update if Evidence for Theory 1 is present.
            # Wait, agents.py says: if theory_index != good_theory_id: pass.
            # This refers to the theory index OF THE AGENT providing the evidence?
            # No, 'update(theory_index, n_s, n_f)' means "Update belief given evidence for theory_index".
            # In agents_update (model.py), it calls update(0, ...) then update(1, ...).
            # So BayesAgent ignores evidence for Theory 0.
            # It only updates for Theory 1.

            # Evidence for Theory 1:
            S1 = total_success[:, 1]
            F1 = total_failure[:, 1]

            # Bayes formula:
            # new_C = 1 / (1 + (1-C)/C * L^k)
            # k = S - F
            # L = (0.5 - u) / (0.5 + u)

            uncertainty = (
                self.bandit.uncertainty
            )  # Assuming all agents share uncertainty (Model param)
            L = (0.5 - uncertainty) / (0.5 + uncertainty)
            k = S1 - F1

            # We only update if k != 0? Or always?
            # If S1=0, F1=0, k=0. L^0 = 1.
            # new_C = 1 / (1 + (1-C)/C * 1) = 1 / (1 + 1/C - 1) = 1/(1/C) = C.
            # So no change if no evidence. Correct.

            # Avoid division by zero if C is 0 or 1.
            # C is in (0, 1) initially (uniform).
            # It might reach boundaries.
            # Clip C to epsilon?
            epsilon = 1e-9
            C = np.clip(self.credences, epsilon, 1 - epsilon)

            term = ((1 - C) / C) * (L**k)
            self.credences = 1 / (1 + term)

        if self.histories:
            for i in range(self.n_agents):
                self.credences_history[i].append(self.credences[i])

    def run_simulation(
        self, number_of_steps: int = 10**6, show_bar: bool = False, *args, **kwargs
    ):
        # Copy logic from Model.run_simulation but using vectorized state

        def stop_condition(credences_prior):
            # Original: np.allclose(prior, post) with tolerance.
            # But here just return False or implement check.
            if self.agent_type == "bayes":
                # Bayes stop: all credences <= 0.5 or > 0.99 (Consensus)
                # Model.py uses: all(credences <= 0.5) or all(credences > 0.99)
                return np.all(self.credences <= 0.5) or np.all(self.credences > 0.99)
            if self.agent_type == "beta":
                # Beta convergence check: np.allclose(prior, post)
                return np.allclose(
                    credences_prior,
                    self.credences,
                    rtol=self.tolerance,
                    atol=self.tolerance,
                )
            else:
                return print("Unknown agent type for stopping condition.")

        def determine_conclusion():
            if self.agent_type == "beta":
                # self.credences is (N, 2).
                # counts = pair[1] > pair[0]
                counts = np.sum(self.credences[:, 1] > self.credences[:, 0])
                return counts / self.n_agents
            elif self.agent_type == "bayes":
                # Bayes: count > 0.99
                counts = np.sum(self.credences > 0.99)
                return counts / self.n_agents

        def determine_conclusion_core():
            # in_degree > 1.
            # vector of degrees.
            degrees = np.sum(
                self.adj_matrix, axis=0
            )  # Sum of rows (outgoing) or cols (incoming)?
            # A_ij = 1 if i -> j.
            # In-degree of j is sum over i of A_ij.
            # This is sum of column j. axis=0.

            core_mask = degrees > 1
            if np.sum(core_mask) == 0:
                return 0.0

            if self.agent_type == "beta":
                core_credences = self.credences[core_mask]
                counts = np.sum(core_credences[:, 1] > core_credences[:, 0])
                return counts / len(core_credences)
            elif self.agent_type == "bayes":
                core_credences = self.credences[core_mask]
                counts = np.sum(core_credences > 0.99)
                return counts / len(core_credences)

        iterable = range(number_of_steps)
        if show_bar:
            iterable = tqdm.tqdm(iterable)

        credences_prior = self.credences.copy()
        alphas_betas_prior = self.alphas_betas.copy() if self.agent_type == "beta" and self.compute_convergence else None
        
        # Track per-step belief change if computing convergence
        if self.compute_convergence and self.agent_type == "beta":
            self.belief_change_history = []

        for _ in iterable:
            credences_prior = self.credences.copy()
            if self.agent_type == "beta" and self.compute_convergence:
                alphas_betas_prior = self.alphas_betas.copy()
            
            self.step()
            
            # Compute per-step belief change for convergence tracking
            if self.compute_convergence and self.agent_type == "beta":
                # Compute mean of prior and posterior
                a_prior = alphas_betas_prior[:, :, 0]
                b_prior = alphas_betas_prior[:, :, 1]
                mean_prior = a_prior / (a_prior + b_prior)
                
                a_post = self.alphas_betas[:, :, 0]
                b_post = self.alphas_betas[:, :, 1]
                mean_post = a_post / (a_post + b_post)
                
                # Mean absolute change across agents for each theory
                abs_change = np.abs(mean_post - mean_prior)
                self.belief_change_history.append((
                    np.mean(abs_change[:, 0]),  # Theory 0
                    np.mean(abs_change[:, 1])   # Theory 1
                ))

            if self.tstep_stopping:
                continue

            if stop_condition(credences_prior):
                break
        
        # Store final prior for convergence analysis
        if self.compute_convergence:
            self.credences_prior_final = credences_prior

        self.conclusion = determine_conclusion()
        self.conclusion_core = determine_conclusion_core()

        # Root nodes metric
        # in_degree == 0.
        degrees = np.sum(self.adj_matrix, axis=0)
        root_mask = degrees == 0

        if not np.any(root_mask):
            self.proportion_reached_by_truth = 0.0
        else:
            # Truthful roots
            if self.agent_type == "beta":
                is_truthful = self.credences[:, 1] > self.credences[:, 0]
            elif self.agent_type == "bayes":
                is_truthful = self.credences > 0.5

            truthful_mask = root_mask & is_truthful
            truthful_roots = np.where(truthful_mask)[0]

            if len(truthful_roots) > 0:
                # Descendants using matrix powers or BFS?
                # NetworkX is available in self.network.
                # Let's use NX for this part as it's run once at end.

                truthful_root_nodes = [self.nodes[i] for i in truthful_roots]
                collective_reach_set = set()
                for node in truthful_root_nodes:
                    collective_reach_set.add(node)
                    collective_reach_set.update(nx.descendants(self.network, node))

                self.proportion_reached_by_truth = (
                    len(collective_reach_set) / self.n_agents
                )
            else:
                self.proportion_reached_by_truth = 0.0
        
        # Compute convergence metrics if requested (Beta agent only)
        if self.compute_convergence and self.agent_type == "beta" and alphas_betas_prior is not None:
            self._compute_convergence_metrics(alphas_betas_prior)
        
        # Compute root node analysis if requested
        if self.compute_root_analysis:
            self._compute_root_analysis()
    
    def _compute_convergence_metrics(self, alphas_betas_prior):
        """
        Compute belief change metrics between last two steps.
        
        For Beta distributions:
        - Absolute difference: |mean_final - mean_prior|
        - KL divergence: KL(prior || posterior) using scipy.stats.entropy
        
        Args:
            alphas_betas_prior: (N, 2, 2) array of prior alpha/beta values
        """
        # Compute means for prior and posterior
        # Mean of Beta(a, b) = a / (a + b)
        a_prior = alphas_betas_prior[:, :, 0]
        b_prior = alphas_betas_prior[:, :, 1]
        mean_prior = a_prior / (a_prior + b_prior)
        
        a_post = self.alphas_betas[:, :, 0]
        b_post = self.alphas_betas[:, :, 1]
        mean_post = a_post / (a_post + b_post)
        
        # Absolute difference in means: shape (N, 2)
        self.belief_change_abs = np.abs(mean_post - mean_prior)
        
        # KL divergence for Beta distributions
        # KL(Beta(a1,b1) || Beta(a2,b2)) = log(B(a2,b2)/B(a1,b1)) 
        #   + (a1-a2)*psi(a1) + (b1-b2)*psi(b1) + (a2-a1+b2-b1)*psi(a1+b1)
        # where B is beta function and psi is digamma
        from scipy.special import betaln, digamma
        
        # Prior is P, Posterior is Q, we compute KL(P || Q)
        a1, b1 = a_prior, b_prior  # Prior
        a2, b2 = a_post, b_post    # Posterior
        
        kl = (betaln(a2, b2) - betaln(a1, b1) +
              (a1 - a2) * digamma(a1) +
              (b1 - b2) * digamma(b1) +
              (a2 - a1 + b2 - b1) * digamma(a1 + b1))
        
        self.belief_change_kl = kl  # Shape (N, 2)
    
    def _compute_root_analysis(self):
        """
        Analyze root nodes (nodes with no incoming edges) and their influence.
        
        Computes:
        - Which nodes are roots
        - Their final credences
        - Whether they believe the truth
        - Their descendant sets and counts
        - Weighted truth share (root beliefs weighted by descendant count)
        - Node-level predictions and accuracy (AUC-ROC)
        """
        # Identify root nodes (in_degree == 0)
        degrees = np.sum(self.adj_matrix, axis=0)  # In-degree
        root_mask = degrees == 0
        root_indices = np.where(root_mask)[0]
        
        if len(root_indices) == 0:
            self.root_analysis = {
                'n_roots': 0,
                'root_indices': np.array([]),
                'root_node_ids': [],
                'root_credences': np.array([]),
                'root_believes_truth': np.array([]),
                'descendant_counts': np.array([]),
                'descendants': [],
                'weighted_truth_share': None,
                'unweighted_truth_share': None,
                'node_predictions': np.array([]),
                'node_actuals': np.array([]),
                'node_accuracy': None,
                'node_auc_roc': None,
            }
            return
        
        # Get root credences
        if self.agent_type == "beta":
            root_credences = self.credences[root_indices]  # Shape (n_roots, 2)
            # Root believes truth if credence for Theory 1 > Theory 0
            root_believes_truth = root_credences[:, 1] > root_credences[:, 0]
        elif self.agent_type == "bayes":
            root_credences = self.credences[root_indices]  # Shape (n_roots,)
            root_believes_truth = root_credences > 0.5
        
        # Compute descendants for each root
        root_node_ids = [self.nodes[i] for i in root_indices]
        descendants = []
        descendant_counts = []
        
        for node_id in root_node_ids:
            desc_set = nx.descendants(self.network, node_id)
            desc_set.add(node_id)  # Include the root itself
            # Convert to indices
            desc_indices = [self.id_to_index_map[n] for n in desc_set if n in self.id_to_index_map]
            descendants.append(set(desc_indices))
            descendant_counts.append(len(desc_indices))
        
        descendant_counts = np.array(descendant_counts)
        
        # Compute weighted truth share
        # Weight = proportion of network each root influences
        weights = descendant_counts / descendant_counts.sum()
        weighted_truth_share = np.sum(weights * root_believes_truth.astype(float))
        unweighted_truth_share = np.mean(root_believes_truth.astype(float))
        
        # ─── NODE-LEVEL PREDICTIONS ───
        # Prediction: A node will believe truth if it's reachable from ANY truthful root
        node_predictions = np.zeros(self.n_agents)
        
        for i, (root_idx, believes_truth) in enumerate(zip(root_indices, root_believes_truth)):
            if believes_truth:
                # Mark all descendants of this truthful root as predicted to believe truth
                for desc_idx in descendants[i]:
                    node_predictions[desc_idx] = 1.0
        
        # Get actual outcomes (which nodes actually believe truth)
        if self.agent_type == "beta":
            node_actuals = (self.credences[:, 1] > self.credences[:, 0]).astype(float)
        else:  # bayes
            node_actuals = (self.credences > 0.5).astype(float)
        
        # Compute accuracy metrics
        node_accuracy = np.mean(node_predictions == node_actuals)
        
        # Compute AUC-ROC (handle edge case where all predictions are same class)
        node_auc_roc = None
        try:
            from sklearn.metrics import roc_auc_score
            # AUC-ROC requires both classes in y_true
            if len(np.unique(node_actuals)) > 1 and len(np.unique(node_predictions)) > 1:
                node_auc_roc = roc_auc_score(node_actuals, node_predictions)
            elif len(np.unique(node_actuals)) > 1:
                # Predictions all same, but actuals vary - still informative
                node_auc_roc = roc_auc_score(node_actuals, node_predictions)
        except Exception:
            # sklearn not available or other error
            node_auc_roc = None
        
        self.root_analysis = {
            'n_roots': len(root_indices),
            'root_indices': root_indices,
            'root_node_ids': root_node_ids,
            'root_credences': root_credences,
            'root_believes_truth': root_believes_truth,
            'descendant_counts': descendant_counts,
            'descendants': descendants,
            'weighted_truth_share': weighted_truth_share,
            'unweighted_truth_share': unweighted_truth_share,
            # Node-level prediction metrics
            'node_predictions': node_predictions,
            'node_actuals': node_actuals,
            'node_accuracy': node_accuracy,
            'node_auc_roc': node_auc_roc,
        }

