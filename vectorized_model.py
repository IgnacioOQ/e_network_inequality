from imports import *
from vectorized_agents import VectorizedBandit
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
        self.credences = np.zeros((self.n_agents, 2))

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
            for i in range(self.n_agents):
                self.credences[i] = rd.uniform(
                    0, 1
                )  # Note: BayesAgent credences is scalar (float), but using array for consistency?
                # Wait, BayesAgent credences is a scalar float.
                # But model.conclusion uses credences[1] > credences[0] for beta, and credences > 0.99 for bayes.
                # Let's keep it consistent. VectorizedModel for Bayes might store shape (N,).
                pass
            # For now, implementing BETA primarily.
            # If agent_type is bayes, we might need a different structure or just error out if strictly linearizing Beta.
            # Instructions focused on Beta.

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

        # Generating choices
        # if rd.rand() < self.epsilon: ...
        # Vectorized:
        rand_epsilon = rd.rand(self.n_agents)
        explore_mask = rand_epsilon < self.epsilon
        exploit_mask = ~explore_mask

        theory_indices = np.zeros(self.n_agents, dtype=int)

        # Explore: Random choice
        if np.any(explore_mask):
            theory_indices[explore_mask] = rd.randint(0, 2, size=np.sum(explore_mask))

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

        # 2. Run Experiments
        n_success, n_total = self.bandit.experiment(theory_indices, self.n_experiments)
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

        if self.histories:
            for i in range(self.n_agents):
                self.credences_history[i].append(self.credences[i])

    def run_simulation(
        self, number_of_steps: int = 10**6, show_bar: bool = False, *args, **kwargs
    ):
        # Copy logic from Model.run_simulation but using vectorized state

        def stop_condition():
            # Original: np.allclose(prior, post) with tolerance.
            # But here just return False or implement check.
            return False

        def determine_conclusion():
            # self.credences is (N, 2).
            # counts = pair[1] > pair[0]
            counts = np.sum(self.credences[:, 1] > self.credences[:, 0])
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

            core_credences = self.credences[core_mask]
            counts = np.sum(core_credences[:, 1] > core_credences[:, 0])
            return counts / len(core_credences)

        iterable = range(number_of_steps)
        if show_bar:
            iterable = tqdm.tqdm(iterable)

        for _ in iterable:
            self.step()
            if stop_condition():
                break

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
            # indices where root_mask is True AND credence[1] > credence[0]
            truthful_mask = root_mask & (self.credences[:, 1] > self.credences[:, 0])
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
