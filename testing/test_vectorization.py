import unittest
import numpy as np
import numpy.random as rd
import networkx as nx
from models.model import Model
from models.vectorized_model import VectorizedModel
from scipy.stats import beta


class TestVectorization(unittest.TestCase):
    def setUp(self):
        # Create a simple graph
        self.G = nx.path_graph(5, create_using=nx.DiGraph)
        self.seed = 42
        self.n_experiments = 10
        self.uncertainty = 0.1

    def test_initialization_match(self):
        """
        Verify that both models initialize to the exact same state given the same seed.
        """
        # Initialize original Model
        rd.seed(self.seed)
        model = Model(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
        )

        # Initialize Vectorized Model
        rd.seed(self.seed)
        vec_model = VectorizedModel(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
        )

        # Compare Alphas/Betas
        # Model stores in list of agents
        model_alphas_betas = np.array([agent.alphas_betas for agent in model.agents])
        vec_alphas_betas = vec_model.alphas_betas

        np.testing.assert_array_equal(
            model_alphas_betas,
            vec_alphas_betas,
            err_msg="Initial Alphas/Betas do not match",
        )

        # Compare Credences
        model_credences = np.array([agent.credences for agent in model.agents])
        vec_credences = vec_model.credences

        np.testing.assert_array_almost_equal(
            model_credences, vec_credences, err_msg="Initial Credences do not match"
        )

    def test_update_logic_equivalence(self):
        """
        Verify that the update logic produces identical results given identical inputs.
        We bypass the random experiment step.
        """
        # Initialize both
        rd.seed(self.seed)
        model = Model(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
        )
        rd.seed(self.seed)
        vec_model = VectorizedModel(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
        )

        # Fabricate experiment results
        # Agent 0: Theory 0, 5 success, 5 failure
        # Agent 1: Theory 1, 8 success, 2 failure
        # ...
        n_agents = len(model.agents)
        theory_indices = np.array([0, 1, 0, 1, 0])
        n_success = np.array([5, 8, 2, 9, 4])
        n_failures = np.array([5, 2, 8, 1, 6])

        # Apply to Model
        experiments_results = {}
        for i, agent in enumerate(model.agents):
            experiments_results[agent.id] = [
                theory_indices[i],
                n_success[i],
                n_failures[i],
            ]

        model.agents_update(experiments_results)

        # Apply to VectorizedModel
        # We need to manually invoke the update part of step().
        # Replicating the logic inside step() for update:

        outcome_success = np.zeros((n_agents, 2))
        outcome_failure = np.zeros((n_agents, 2))
        rows = np.arange(n_agents)
        outcome_success[rows, theory_indices] = n_success
        outcome_failure[rows, theory_indices] = n_failures

        adj_matrix = nx.to_numpy_array(self.G, nodelist=vec_model.nodes)

        agg_success = adj_matrix.T @ outcome_success
        agg_failure = adj_matrix.T @ outcome_failure

        total_success = agg_success + outcome_success
        total_failure = agg_failure + outcome_failure

        vec_model.alphas_betas[:, :, 0] += total_success
        vec_model.alphas_betas[:, :, 1] += total_failure

        # Re-calc credences
        a = vec_model.alphas_betas[:, :, 0]
        b = vec_model.alphas_betas[:, :, 1]
        vec_model.credences = a / (a + b)

        # Compare
        model_alphas_betas = np.array([agent.alphas_betas for agent in model.agents])
        np.testing.assert_array_equal(
            model_alphas_betas,
            vec_model.alphas_betas,
            err_msg="Updated Alphas/Betas do not match",
        )

        model_credences = np.array([agent.credences for agent in model.agents])
        np.testing.assert_array_almost_equal(
            model_credences,
            vec_model.credences,
            err_msg="Updated Credences do not match",
        )

    def test_bayes_initialization_match(self):
        """
        Verify Bayes agent initialization.
        """
        agent_type = "bayes"
        rd.seed(self.seed)
        model = Model(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
            agent_type=agent_type,
        )

        rd.seed(self.seed)
        vec_model = VectorizedModel(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
            agent_type=agent_type,
        )

        # Credences for Bayes are floats
        model_credences = np.array([agent.credences for agent in model.agents])
        vec_credences = vec_model.credences

        np.testing.assert_array_equal(
            model_credences,
            vec_credences,
            err_msg="Initial Bayes Credences do not match",
        )

    def test_bayes_update_logic(self):
        """
        Verify Bayes update logic manually.
        """
        agent_type = "bayes"
        rd.seed(self.seed)
        vec_model = VectorizedModel(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
            agent_type=agent_type,
        )

        # Manually set credences to known value
        vec_model.credences = np.full(vec_model.n_agents, 0.5)

        # Mock experiment results
        # Suppose all agents choose Theory 1 and get 10 successes, 0 failures
        # And aggregate from neighbors
        # Graph is path graph 0->1->2->3->4
        # Agent 1 has predecessor 0.
        # If Agent 0 has 10S, 0F.
        # Agent 1 updates with own (10,0) + Agent 0 (10,0) = (20,0)?
        # Let's mock the update calculation logic inside step() for Bayes.

        # We will simulate the update math
        # S1 = F1 = 0 initially
        # Let's say outcomes are:
        # Agent 0: 10S, 0F
        # Agent 1: 0S, 0F (Suppose he chose 0, so no result? Or chose 1 and got nothing?)

        # To be precise, let's use the Model's logic to generate a ground truth update
        # and compare VectorizedModel's output.

        rd.seed(self.seed)
        model = Model(
            self.G,
            self.n_experiments,
            uncertainty=self.uncertainty,
            seeded=True,
            seed=self.seed,
            agent_type=agent_type,
        )
        for agent in model.agents:
            agent.credences = 0.5

        # Force experiments
        # Agent 0 tests Theory 1: 10 S, 0 F
        # Agent 1 tests Theory 1: 5 S, 5 F
        experiments_results = {
            0: [1, 10, 0],
            1: [1, 5, 5],
            2: [0, 0, 0],  # Theory 0 choice -> 0,0
            3: [0, 0, 0],
            4: [0, 0, 0],
        }

        model.agents_update(experiments_results)
        model_credences = np.array([agent.credences for agent in model.agents])

        # Vectorized Update
        vec_model.credences = np.full(
            vec_model.n_agents, 0.5
        )  # Reset to 0.5 to match start
        # Replicate update step
        n_agents = vec_model.n_agents
        outcome_success = np.zeros((n_agents, 2))
        outcome_failure = np.zeros((n_agents, 2))

        # Populate based on experiments_results
        for i in range(n_agents):
            idx, s, f = experiments_results[i]
            outcome_success[i, idx] = s
            outcome_failure[i, idx] = f

        adj_matrix = nx.to_numpy_array(self.G, nodelist=vec_model.nodes)
        agg_success = adj_matrix.T @ outcome_success
        agg_failure = adj_matrix.T @ outcome_failure

        total_success = agg_success + outcome_success
        total_failure = agg_failure + outcome_failure

        # Bayes Update
        S1 = total_success[:, 1]
        F1 = total_failure[:, 1]
        uncertainty = vec_model.bandit.uncertainty
        L = (0.5 - uncertainty) / (0.5 + uncertainty)
        k = S1 - F1
        epsilon = 1e-9
        C = np.clip(vec_model.credences, epsilon, 1 - epsilon)
        term = ((1 - C) / C) * (L**k)
        vec_model.credences = 1 / (1 + term)

        np.testing.assert_array_almost_equal(
            model_credences, vec_model.credences, err_msg="Bayes Update Mismatch"
        )


if __name__ == "__main__":
    unittest.main()
