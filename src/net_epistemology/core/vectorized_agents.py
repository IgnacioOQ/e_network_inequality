from ..utils.imports import np, rd


class VectorizedBandit:
    """
    Vectorized version of the Bandit class.
    """

    def __init__(self, uncertainty=0.1, n_agents=1):
        self.uncertainty = uncertainty
        self.p_bad_theory = 0.5
        self.p_good_theory = 0.5 + uncertainty
        self.n_agents = n_agents

    def experiment(self, theory_indices, n_experiments):
        """
        Vectorized experiment.

        Args:
            theory_indices: np.array of shape (n_agents,), containing 0 or 1.
            n_experiments: int, number of experiments per agent.

        Returns:
            n_success: np.array of shape (n_agents,)
            n_total: int or np.array of shape (n_agents,)
        """
        # Create an array of probabilities based on choices
        probs = np.where(theory_indices == 0, self.p_bad_theory, self.p_good_theory)

        # Generate outcomes
        # n_success = np.random.binomial(n_experiments, probs)
        # However, to match the original code's randomness structure exactly
        # (if possible),
        # we might need to be careful. But for 'statistically equivalent',
        # using vectorized operations is the goal.

        n_success = rd.binomial(n_experiments, probs)

        return n_success, n_experiments
