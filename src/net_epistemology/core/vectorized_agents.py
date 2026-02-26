from typing import Tuple

from numpy.typing import ArrayLike, NDArray

from ..utils.imports import np, rd


class VectorizedBandit:
    """
    Vectorized version of the Bandit class.
    """

    def __init__(self, uncertainty: float = 0.1, n_agents: int = 1) -> None:
        self.uncertainty: float = uncertainty
        self.p_bad_theory: float = 0.5
        self.p_good_theory: float = 0.5 + uncertainty
        self.n_agents: int = n_agents

    def experiment(
        self, theory_indices: ArrayLike, n_experiments: int
    ) -> Tuple[NDArray[np.int_], int]:
        """
        Vectorized experiment.

        Args:
            theory_indices: Array-like of shape (n_agents,), containing 0 or 1.
                Can be a list, tuple, or numpy array of integers.
            n_experiments: Number of experiments per agent.s

        Returns:
            Tuple containing:
                - n_success: numpy array of shape (n_agents,) with number of successes per agent
                - n_total: int, total number of experiments (same for all agents)
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
