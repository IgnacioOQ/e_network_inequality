from imports import *
from agents import Bandit, BetaAgent

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
        np.testing.assert_array_equal(self.agent.alphas_betas, np.array([[1, 1], [1, 1]]))
        mean = beta.stats(1, 1, moments='m')
        np.testing.assert_array_equal(self.agent.credences, np.array([mean, mean]))
        self.assertTrue(self.agent.histories)
        self.assertEqual(len(self.agent.credences_history), 1)
    
    def test_greedy_choice(self):
        choice = self.agent.greedy_choice()
        self.assertIn(choice, [0, 1])
    
    def test_experiment(self):
        n_experiments = 10
        theory_index, n_success, n_failures = self.agent.experiment(n_experiments)
        self.assertIn(theory_index, [0, 1])
        self.assertEqual(n_success + n_failures, n_experiments)
        self.assertTrue(0 <= n_success <= n_experiments)
    
    def test_beta_update(self):
        self.agent.beta_update(0, 5, 5)
        self.assertEqual(self.agent.alphas_betas[0][0], 6)
        self.assertEqual(self.agent.alphas_betas[0][1], 6)
        mean = beta.stats(6, 6, moments='m')
        self.assertEqual(self.agent.credences[0], mean)
        self.assertEqual(len(self.agent.credences_history), 2)
        
    def test_beta_update_invalid_index(self):
        with self.assertRaises(IndexError):
            self.agent.beta_update(2, 5, 5)

if __name__ == "__main__":
    unittest.main()
