from.utils.imports import *
from.core.agents import Bandit, BetaAgent


class TestBandit(unittest.TestCase):

    def test_bandit_initialization(self):
        bandit = Bandit(uncertainty=0.2)
        self.assertEqual(bandit.uncertainty, 0.2)
        self.assertEqual(bandit.p_bad_theory, 0.5)
        self.assertEqual(bandit.p_good_theory, 0.7)

    def test_experiment(self):
        bandit = Bandit()
        n_experiments = 100
        n_success, total = bandit.experiment(0, n_experiments)
        self.assertEqual(total, n_experiments)
        self.assertTrue(0 <= n_success <= n_experiments)

        n_success, total = bandit.experiment(1, n_experiments)
        self.assertEqual(total, n_experiments)
        self.assertTrue(0 <= n_success <= n_experiments)

    def test_experiment_invalid_index(self):
        bandit = Bandit()
        with self.assertRaises(ValueError):
            bandit.experiment(2, 10)


class TestBetaAgent(unittest.TestCase):

    def setUp(self):
        self.bandit = Bandit()
        self.agent = BetaAgent(id=1, bandit=self.bandit, histories=True)

    def test_initialization(self):
        self.assertEqual(self.agent.id, 1)
        self.assertEqual(self.agent.bandit, self.bandit)
        # Initialization is random, so we check shape and non-negativity
        self.assertEqual(self.agent.alphas_betas.shape, (2, 2))
        self.assertTrue(np.all(self.agent.alphas_betas >= 0))
        # np.testing.assert_array_equal(self.agent.alphas_betas, np.array([[1, 1], [1, 1]]))
        # mean = beta.stats(1, 1, moments='m')
        # np.testing.assert_array_equal(self.agent.credences, np.array([mean, mean]))
        self.assertTrue(self.agent.histories)
        self.assertEqual(len(self.agent.credences_history), 1)

    def test_greedy_choice(self):
        choice = self.agent.egreedy_choice()
        self.assertIn(choice, [0, 1])

    def test_experiment(self):
        n_experiments = 10
        theory_index, n_success, n_failures = self.agent.experiment(n_experiments)
        self.assertIn(theory_index, [0, 1])
        self.assertEqual(n_success + n_failures, n_experiments)
        self.assertTrue(0 <= n_success <= n_experiments)

    def test_beta_update(self):
        # Capture initial state
        initial_alpha_0 = self.agent.alphas_betas[0][0]
        initial_beta_0 = self.agent.alphas_betas[0][1]

        # Update
        self.agent.update(0, 5, 5)

        # Assertions based on dynamic initial state
        self.assertEqual(self.agent.alphas_betas[0][0], initial_alpha_0 + 5)
        self.assertEqual(self.agent.alphas_betas[0][1], initial_beta_0 + 5)

        new_alpha = initial_alpha_0 + 5
        new_beta = initial_beta_0 + 5
        mean = beta.stats(new_alpha, new_beta, moments="m")
        self.assertEqual(self.agent.credences[0], mean)
        self.assertEqual(len(self.agent.credences_history), 2)

    def test_beta_update_invalid_index(self):
        with self.assertRaises(IndexError):
            self.agent.update(2, 5, 5)


if __name__ == "__main__":
    unittest.main()
