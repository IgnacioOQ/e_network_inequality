from utils.imports import beta, np, nx, rd, tqdm
from .bandit import VectorizedBandit


class VectorizedModel:
    """
    Vectorized network, bandit model for efficient multi-agent simulation.

    This class implements a vectorized version of the network, bandit model,
    where agents are arranged in a network and learn about theories through
    experimentation and social learning from neighbors. The vectorized implementation
    uses NumPy arrays for efficient computation with many agents.

    The model supports two theories with different success rates:
    - Theory 0 (bad): success rate = 0.5
    - Theory 1 (good): success rate = 0.5 + uncertainty

    Agents can be of two types:
    - Beta agents: Maintain Beta distributions over theory credences, updated via
      Bayesian inference. Can use epsilon-greedy exploration.
    - Bayes agents: Maintain single credence values, updated via Bayesian updating.
      Only consider evidence for the "good" theory (Theory 1).

    Args:
        network: NetworkX graph representing agent connections. Directed edges
            indicate information flow (if A->B, B observes A's results).
        n_experiments: Number of experiments each agent performs per step.
        agent_type: Type of agents - "beta" or "bayes". Default: "beta".
        uncertainty: Difference in success probability between theories.
            Default: 0.001.
        tolerance: Convergence tolerance for stopping condition.
            Default: 0.005.
        histories: Whether to track credence history for each agent.
            Default: False.
        sampling_update: If True, Beta agents sample from posterior; if False,
            use mean. Default: False.
        variance_stopping: (Unused) Placeholder for variance-based stopping.
            Default: False.
        tolerance_stopping: If True, checks for tolerance-based convergence
            every step and stops early if converged. Default: True.
        tstep_stopping: If True, only stops at max steps (no convergence
            check). Default: False.
        choice_stability_stopping: If True, stop when *every* agent's chosen
            theory (greedy argmax of its credence) has been unchanged for the
            last ``choice_stability_window`` steps. This is a decision-stability
            criterion: since ``epsilon == 0`` an agent's choice is the
            deterministic pick ``credences[:, 1] > credences[:, 0]`` (Beta) or
            ``credences > 0.5`` (Bayes), so the rule is equivalent to "no agent
            has crossed the 0.5 decision boundary in the last W steps". It is
            mutually exclusive with tolerance_stopping / tstep_stopping (see the
            dispatch in run_simulation) and defaults OFF, so existing callers
            are unaffected. NOTE: it guarantees *decision* stability, not
            *parameter* stability — α/β may keep growing while a credence stays
            on one side of 0.5. Default: False.
        choice_stability_window: Number of consecutive flip-free steps required
            before choice_stability_stopping fires (the W above). Default: 500.
        record_choice_flips: If True, record ``(step, truth_share)`` into
            self.choice_flip_history on every step where at least one agent
            flips its choice (plus a baseline entry at run start). Orthogonal to
            the stopping mode — enable it alongside tstep_stopping to run to
            max_steps once and derive the choice-stability stop step for *any*
            window W offline (truth_share is constant between flips). Default:
            False.
        directed_network: Whether network is directed. Default: True.
        seed: Random seed for reproducibility. If None and seeded=True,
            generates random seed.
        seeded: Whether to use seeded randomness. Default: False.
        compute_convergence: If True, compute belief change metrics during
            simulation. Default: False.
        compute_root_analysis: If True, analyze root nodes and their influence
            after simulation. Default: False.

    Attributes:
        network: The agent network graph.
        n_agents (int): Number of agents.
        nodes (list): List of node IDs.
        id_to_index_map (dict): Mapping from node IDs to array indices.
        n_experiments (int): Experiments per agent per step.
        agent_type (str): "beta" or "bayes".
        epsilon (float): Exploration rate for epsilon-greedy (Beta agents).
        tolerance (float): Convergence tolerance.
        bandit (VectorizedBandit): Bandit model.
        alphas_betas (ndarray): Shape (n_agents, 2, 2). Beta distribution
            parameters for Beta agents. [agent, theory, param] where
            param 0=alpha, 1=beta.
        credences (ndarray): Current agent beliefs. Shape (n_agents, 2) for
            Beta agents (credences for both theories), (n_agents,) for Bayes
            agents (credence for Theory 1).
        adj_matrix (sparse matrix): Network adjacency matrix for vectorized
            operations.
        n_steps (int): Current simulation step count.
        conclusion (float): Proportion of agents believing truth at simulation
            end.
        conclusion_core (float): Proportion of core agents (in-degree > 1)
            believing truth.
        proportion_reached_by_truth (float): Proportion of network reachable
            from truthful root nodes.
        credences_history (list): History of credences if histories=True.
        credences_prior_final (ndarray): Credences from penultimate step
            (for convergence).
        belief_change_abs (ndarray): Absolute belief change in final step
            (Beta only).
        belief_change_kl (ndarray): KL divergence of beliefs in final step
            (Beta only).
        belief_change_history (list): Per-step mean belief changes
            (if compute_convergence=True).
        root_analysis (dict): Analysis of root nodes and their influence
            (if compute_root_analysis=True).

    Methods:
        step(): Execute one simulation step (experiment + update).
        run_simulation(number_of_steps, show_bar, auc_stopping,
            auc_threshold, auc_check_interval): Run full simulation with
            optional stopping conditions.

    Example:
        >>> import networkx as nx
        >>> G = nx.erdos_renyi_graph(100, 0.05, directed=True)
        >>> model = VectorizedModel(G, n_experiments=10, agent_type="beta",
        ...                         uncertainty=0.05)
        >>> model.run_simulation(number_of_steps=1000)
        >>> print(f"Conclusion: {model.conclusion:.2%}")
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
        # variance_stopping=False,
        tolerance_stopping=True,
        tstep_stopping=False,
        choice_stability_stopping: bool = False,
        choice_stability_window: int = 500,
        record_choice_flips: bool = False,
        # directed_network=True,
        seed=None,
        seeded=False,
        compute_convergence=False,
        compute_root_analysis=False,
        snapshot_interval: int = 0,
        # *args,
        # **kwargs
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
        self.tolerance_stopping = tolerance_stopping
        self.tstep_stopping = tstep_stopping
        self.choice_stability_stopping = choice_stability_stopping
        self.choice_stability_window = choice_stability_window
        self.record_choice_flips = record_choice_flips
        # Choice-stability tracking state (initialized in run_simulation when
        # choice_stability_stopping or record_choice_flips is enabled):
        #   _prev_choices    : bool vector of each agent's last chosen theory
        #   _last_flip_step  : most recent step (any agent) at which a choice flipped
        #   choice_flip_history : list of (step, truth_share) at each flip, for the
        #                         offline window sweep (record_choice_flips only)
        self._prev_choices = None
        self._last_flip_step = 0
        self.choice_flip_history = []
        self.compute_convergence = compute_convergence
        self.compute_root_analysis = compute_root_analysis
        self.snapshot_interval = snapshot_interval

        self.snapshots = {"step": [], "truth_share": [], "max_belief_change": []}

        # Convergence tracking attributes (populated after simulation if
        # compute_convergence=True)
        self.credences_prior_final = None  # Beliefs at step N-1
        self.belief_change_abs = (
            None  # Per-agent, per-theory absolute difference (final step)
        )
        self.belief_change_kl = None  # Per-agent, per-theory KL divergence (final step)
        self.belief_change_history = (
            None  # Per-step mean belief change: list of (mean_T0, mean_T1)
        )

        # Root analysis attributes (populated after simulation
        # if compute_root_analysis=True)
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
            priors = rd.uniform(0, 4, size=(self.n_agents, 2, 2))
            self.alphas_betas = priors.copy()
            if self.sampling_update:
                self.credences = rd.beta(priors[:, :, 0], priors[:, :, 1])
            else:
                # Mean of Beta(a, b) = a / (a + b)
                a = priors[:, :, 0]
                b = priors[:, :, 1]
                self.credences = a / (a + b)
            # for i in range(self.n_agents):
            #     prior_T1 = rd.uniform(0, 4, size=2)
            #     prior_T2 = rd.uniform(0, 4, size=2)
            #     self.alphas_betas[i] = np.array([prior_T1, prior_T2])

            #     if self.sampling_update:
            #         self.credences[i] = np.array(
            #             [
            #                 rd.beta(prior_T1[0], prior_T1[1], size=1)[0],
            #                 rd.beta(prior_T2[0], prior_T2[1], size=1)[0],
            #             ]
            #         )
            #     else:
            #         mean_T1 = beta.stats(prior_T1[0], prior_T1[1], moments="m")
            #         mean_T2 = beta.stats(prior_T2[0], prior_T2[1], moments="m")
            #         self.credences[i] = np.array([mean_T1, mean_T2])

        elif self.agent_type == "bayes":
            self.credences = rd.uniform(0, 1, size=self.n_agents)
            # for i in range(self.n_agents):
            #     self.credences[i] = rd.uniform(0, 1)

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

        self.adj_matrix = nx.to_scipy_sparse_array(self.network, nodelist=self.nodes)
        # if not directed_network:  # Hein: remove redundancy?
        #     # If undirected, A_adj is symmetric, so A.T == A.
        #     pass

        # Convert to sparse if large? For now dense is fine for 100 agents.
        # Hein: Yes, I would opt for sparse: nx.to_scipy_sparse_array(self.network)

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
            explore_mask = rd.rand(self.n_agents) < self.epsilon
            exploit_mask = ~explore_mask

            # Explore: Random choice
            if np.any(explore_mask):
                theory_indices[explore_mask] = rd.randint(
                    0, 2, size=int(np.sum(explore_mask))
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
                    self.credences[exploit_mask], axis=1  # type: ignore[index]
                )

        elif self.agent_type == "bayes":
            # Bayes Agent Choice: if cred > 0.5 -> 1 (Good), else 0 (Bad)
            # self.credences is shape (N,)
            theory_indices = (self.credences > 0.5).astype(int)  # type: ignore

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

        if self.agent_type == "beta":
            n_success, n_total = self.bandit.experiment(
                theory_indices, self.n_experiments
            )
            n_failures = n_total - n_success

        elif self.agent_type == "bayes":
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

        # Store results for update
        # experiments_results: matrix of shape (N, 2, 2) ->
        # (success, failure) for theory 0 and 1?
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

        individual_success = np.zeros((self.n_agents, 2), dtype=int)
        individual_failure = np.zeros((self.n_agents, 2), dtype=int)

        # Fill them
        # theory_indices is (N,). n_success is (N,).
        rows = np.arange(self.n_agents)
        individual_success[rows, theory_indices] = n_success  # type: ignore
        individual_failure[rows, theory_indices] = n_failures  # type: ignore

        # Aggregate from neighbors
        # Aggregated_Success = A.T @ Outcome_Success
        # Shape: (N, N) @ (N, 2) -> (N, 2)

        # Note on self.directedness in original model:
        # if directed: predecessors.
        # if undirected: neighbors.
        # nx.to_numpy_array handles this (if undirected, symmetric).
        # And we established A.T sums over predecessors (incoming edges).
        # For undirected, A.T = A, sums over neighbors. Correct.

        communicate_success = self.adj_matrix.T @ individual_success
        communicate_failure = self.adj_matrix.T @ individual_failure

        # Add OWN results?
        # Original code:
        # theories_exp_results[theory_index] += results (own)
        # for id in neighbors: ... += results (neighbor)
        # So YES, add own results.

        total_success = communicate_success + individual_success
        total_failure = communicate_failure + individual_failure

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
                # Mean of Beta(a, b) = a / (a + b)
                a = self.alphas_betas[:, :, 0]
                b = self.alphas_betas[:, :, 1]
                self.credences = a / (a + b)

        elif self.agent_type == "bayes":
            # Bayes Update Logic
            # Only update if Evidence for Theory 1 is present.
            # Wait, agents.py says: if theory_index != good_theory_id: pass.
            # This refers to the theory index OF THE AGENT providing the evidence?
            # No, 'update(theory_index, n_s, n_f)' means "Update belief given evidence
            # for theory_index".
            # In agents_update (model.py), it calls update(0, ...) then update(1, ...).
            # So BayesAgent ignores evidence for Theory 0.
            # It only updates for Theory 1.

            # Evidence for Theory 1:
            successes_1 = total_success[:, 1]
            failures_1 = total_failure[:, 1]

            # Bayes formula:
            # new_C = 1 / (1 + (1-C)/C * L^k)
            # k = S - F
            # L = (0.5 - u) / (0.5 + u)

            uncertainty = (
                self.bandit.uncertainty
            )  # Assuming all agents share uncertainty (Model param)
            factor_per_observation = (0.5 - uncertainty) / (0.5 + uncertainty)
            net_score = successes_1 - failures_1

            # We only update if k != 0? Or always?
            # If S1=0, F1=0, k=0. L^0 = 1.
            # new_C = 1 / (1 + (1-C)/C * 1) = 1 / (1 + 1/C - 1) = 1/(1/C) = C.
            # So no change if no evidence. Correct.

            # Avoid division by zero if C is 0 or 1.
            # C is in (0, 1) initially (uniform).
            # It might reach boundaries.
            # Clip C to epsilon?
            credence_clipped = np.clip(self.credences, 1e-9, 1 - 1e-9)

            self.credences = 1 / (
                1
                + ((1 - credence_clipped) / credence_clipped)
                * (factor_per_observation**net_score)
            )

        if self.histories:
            for i in range(self.n_agents):
                self.credences_history[i].append(self.credences[i])

    def run_simulation(
        self,
        number_of_steps: int = 10**6,
        show_bar: bool = False,
        auc_stopping: bool = False,
        auc_threshold: float = 0.95,
        auc_check_interval: int = 500,
        *args,
        **kwargs
    ):
        """
        Run the simulation.

        Args:
            number_of_steps: Maximum number of steps to run
            show_bar: Whether to show progress bar
            auc_stopping: If True, stop when node-level AUC-ROC >= auc_threshold
            auc_threshold: AUC-ROC threshold for stopping (default 0.95)
            auc_check_interval: How often to check AUC (default every 500 steps)
        """
        # Copy logic from Model.run_simulation but using vectorized state

        def stop_condition(credences_prior):
            # Original: np.allclose(prior, post) with tolerance.
            # But here just return False or implement check.
            if self.agent_type == "bayes":
                return np.all(self.credences <= 0.5) or np.all(self.credences > 0.99)  # type: ignore
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

        def compute_current_auc():
            """
            Compute current node-level AUC-ROC for stopping check.
            Falls back to node_accuracy if AUC is unavailable.
            Returns tuple: (auc_or_accuracy, is_auc)
            """
            # Get root nodes and their descendants (cached info)
            degrees = np.sum(self.adj_matrix, axis=0)
            root_mask = degrees == 0
            root_indices = np.where(root_mask)[0]

            if len(root_indices) == 0:
                return None, False

            # Get which roots believe truth
            if self.agent_type == "beta":
                root_believes_truth = (
                    self.credences[root_indices, 1] > self.credences[root_indices, 0]
                )
            else:
                root_believes_truth = self.credences[root_indices] > 0.5

            # Compute node predictions based on reachability from truthful roots
            node_predictions = np.zeros(self.n_agents)
            root_node_ids = [self.nodes[i] for i in root_indices]

            for i, (node_id, believes_truth) in enumerate(
                zip(root_node_ids, root_believes_truth)
            ):
                if believes_truth:
                    desc_set = nx.descendants(self.network, node_id)
                    desc_set.add(node_id)
                    for n in desc_set:
                        if n in self.id_to_index_map:
                            node_predictions[self.id_to_index_map[n]] = 1.0

            # Get actual outcomes
            if self.agent_type == "beta":
                node_actuals = (self.credences[:, 1] > self.credences[:, 0]).astype(
                    float
                )
            else:
                node_actuals = (self.credences > 0.5).astype(float)

            # Try to compute AUC-ROC first
            try:
                from sklearn.metrics import roc_auc_score

                if (
                    len(np.unique(node_actuals)) > 1
                    and len(np.unique(node_predictions)) > 1
                ):
                    return roc_auc_score(node_actuals, node_predictions), True
            except Exception:
                pass

            # Fallback to node_accuracy if AUC not computable
            node_accuracy = np.mean(node_predictions == node_actuals)
            return node_accuracy, False

        iterable = range(number_of_steps)
        if show_bar:
            iterable = tqdm(iterable)

        credences_prior = self.credences.copy()
        alphas_betas_prior = (
            self.alphas_betas.copy()
            if self.agent_type == "beta" and self.compute_convergence
            else None
        )

        # Track per-step belief change if computing convergence
        if self.compute_convergence and self.agent_type == "beta":
            self.belief_change_history = []

        # Track AUC stopping
        auc_stopped = False

        # Choice-stability tracking init (choice_stability_stopping and/or
        # record_choice_flips). Seeded from the current state so it also works
        # when run_simulation is called on an injected/resumed state. n_steps is
        # cumulative across calls, so _last_flip_step is anchored to it.
        def _current_choices():
            # Each agent's chosen theory = greedy argmax of its credence (epsilon
            # is hardcoded to 0). For beta this is the indicator
            # credences[:,1] > credences[:,0]; for bayes, credences > 0.5. A
            # "flip" is any change in this boolean vector between steps.
            if self.agent_type == "beta":
                return self.credences[:, 1] > self.credences[:, 0]
            return self.credences > 0.5

        if self.choice_stability_stopping or self.record_choice_flips:
            self._prev_choices = _current_choices()
            self._last_flip_step = self.n_steps
            if self.record_choice_flips:
                # Baseline entry: truth_share holding until the first flip.
                self.choice_flip_history = [(self.n_steps, determine_conclusion())]

        for step_num in iterable:
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
                self.belief_change_history.append(
                    (
                        np.mean(abs_change[:, 0]),  # Theory 0
                        np.mean(abs_change[:, 1]),  # Theory 1
                    )
                )

            # Snapshots
            if self.snapshot_interval > 0 and (step_num + 1) % self.snapshot_interval == 0:
                current_truth_share = determine_conclusion()
                self.snapshots["step"].append(step_num + 1)
                self.snapshots["truth_share"].append(current_truth_share)
                
                if self.agent_type == "beta":
                    max_change = np.max(np.abs(credences_prior - self.credences))
                    self.snapshots["max_belief_change"].append(max_change)
                else:
                    self.snapshots["max_belief_change"].append(None)

            # Choice-stability tracking (runs regardless of stopping mode so
            # record_choice_flips works alongside tstep_stopping). Detect a
            # flip in any agent's chosen theory since the previous step.
            if self.choice_stability_stopping or self.record_choice_flips:
                current_choices = _current_choices()
                if np.any(current_choices != self._prev_choices):
                    self._last_flip_step = self.n_steps
                    self._prev_choices = current_choices
                    if self.record_choice_flips:
                        self.choice_flip_history.append(
                            (self.n_steps, determine_conclusion())
                        )

            # Stopping Conditions
            if self.tolerance_stopping:
                if stop_condition(credences_prior):
                    break  # Old convergence-based stopping
            elif self.tstep_stopping:
                continue  # tstep mode: run for fixed number of steps
            elif self.choice_stability_stopping:
                # Stop once no agent has flipped for `choice_stability_window`
                # consecutive steps (i.e. all agents' choices have been stable).
                if self.n_steps - self._last_flip_step >= self.choice_stability_window:
                    break
            elif auc_stopping:
                # AUC-based stopping (check every auc_check_interval steps)
                if (step_num + 1) % auc_check_interval == 0:
                    result = compute_current_auc()
                    if result[0] is not None and result[0] >= auc_threshold:
                        auc_stopped = True
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
        if (
            self.compute_convergence
            and self.agent_type == "beta"
            and alphas_betas_prior is not None
        ):
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
        a2, b2 = a_post, b_post  # Posterior

        kl = (
            betaln(a2, b2)
            - betaln(a1, b1)
            + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1)
            + (a2 - a1 + b2 - b1) * digamma(a1 + b1)
        )

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
                "n_roots": 0,
                "root_indices": np.array([]),
                "root_node_ids": [],
                "root_credences": np.array([]),
                "root_believes_truth": np.array([]),
                "descendant_counts": np.array([]),
                "descendants": [],
                "weighted_truth_share": None,
                "unweighted_truth_share": None,
                "node_predictions": np.array([]),
                "node_actuals": np.array([]),
                "node_accuracy": None,
                "node_auc_roc": None,
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
            desc_indices = [
                self.id_to_index_map[n] for n in desc_set if n in self.id_to_index_map
            ]
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

        for i, (root_idx, believes_truth) in enumerate(
            zip(root_indices, root_believes_truth)
        ):
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
            if (
                len(np.unique(node_actuals)) > 1
                and len(np.unique(node_predictions)) > 1
            ):
                node_auc_roc = roc_auc_score(node_actuals, node_predictions)
            elif len(np.unique(node_actuals)) > 1:
                # Predictions all same, but actuals vary - still informative
                node_auc_roc = roc_auc_score(node_actuals, node_predictions)
        except Exception:
            # sklearn not available or other error
            node_auc_roc = None

        self.root_analysis = {
            "n_roots": len(root_indices),
            "root_indices": root_indices,
            "root_node_ids": root_node_ids,
            "root_credences": root_credences,
            "root_believes_truth": root_believes_truth,
            "descendant_counts": descendant_counts,
            "descendants": descendants,
            "weighted_truth_share": weighted_truth_share,
            "unweighted_truth_share": unweighted_truth_share,
            # Node-level prediction metrics
            "node_predictions": node_predictions,
            "node_actuals": node_actuals,
            "node_accuracy": node_accuracy,
            "node_auc_roc": node_auc_roc,
        }
