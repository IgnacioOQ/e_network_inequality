# import numpy as np
# import numpy.random as rd
# from scipy.stats import beta
from imports import *  

class Bandit:
    """
    A model representing the problem of theory choice, where a new theory is compared 
    to an existing one with an added margin of uncertainty.

    Attributes:
    - uncertainty (float): The margin of uncertainty in favor of the new theory.
    - p_bad_theory (float): The probability of success for the existing (bad) theory.
    - p_good_theory (float): The probability of success for the new (good) theory, 
      which is improved by the uncertainty margin.

    Methods:
    - experiment(self, index: int, n_experiments: int) -> tuple[int, int]: 
      Simulates a series of experiments and returns the number of successes and 
      total experiments.
    """

    def __init__(self, uncertainty: float = 0.1):
        """
        Initializes the Bandit model with a given uncertainty margin.

        Args:
        - uncertainty (float, optional): The uncertainty margin added to the probability
          of the new theory. Defaults to 0.1.
        """
        self.uncertainty = uncertainty
        self.p_bad_theory = 0.5
        self.p_good_theory = 0.5 + uncertainty

    def experiment(self, theory_index: int, n_experiments: int) -> tuple[int, int]:
        """
        Simulates a set of experiments based on the selected theory.

        Args:
        - index (int): Indicates which theory to test (0 for bad theory, 1 for good theory).
        - n_experiments (int): The number of experiments to run.

        Returns:
        - tuple[int, int]: A tuple containing the number of successful experiments 
          and the total number of experiments.

        Raises:
        - ValueError: If the index is not 0 or 1.
        """
        import numpy.random as rd

        if theory_index == 0:
            n_success = rd.binomial(n_experiments, self.p_bad_theory)
        elif theory_index == 1:
            n_success = rd.binomial(n_experiments, self.p_good_theory)
        else:
            raise ValueError("Index must be 0 (bad theory) or 1 (good theory).")

        return n_success, n_experiments



class BetaAgent:
    """
    An agent in a network epistemology playground using Bayesian updating.
    
    Attributes:
    - id (int): Unique identifier for the agent.
    - uncertainty_problem (UncertaintyProblem): An instance representing the uncertainty model.
    - alphas_betas (np.ndarray): A 2D array storing alpha (success) and beta (failure) parameters for each theory.
    - credences (np.ndarray): The agent's belief in each theory, initialized as the mean of beta distributions.
    - histories (bool): Whether to store the history of credences.
    - credences_history (list): A list storing past credences if histories are enabled.
    
    Methods:
    - __init__(self, id, uncertainty_problem=UncertaintyProblem, histories=False):
      Initializes the agent with given ID and uncertainty problem.
    - greedy_experiment(self, n_experiments: int):
      Selects the theory with the highest credence (breaking ties randomly) and runs an experiment.
    - beta_update(self, theory_index, n_success, n_failures):
      Updates the agent's belief using Bayesian updating based on observed successes and failures.
    """

    def __init__(self, id, bandit: Bandit, histories=False):
        self.id = id
        self.bandit = bandit
        
        # Initializing Beta Agent: Each theory starts with one success and one failure
        self.alphas_betas = np.array([[1, 1], [1, 1]])
        mean = beta.stats(1, 1, moments='m')        
        self.credences = np.array([mean, mean])
        
        self.histories = histories
        if self.histories:
            self.credences_history = []
            self.credences_history.append(self.credences)
        
    def greedy_experiment(self, n_experiments: int):
        """
        Selects the theory with the highest credence (breaking ties randomly) and runs an experiment.
        
        Parameters:
        - n_experiments (int): Number of experiments to perform.
        
        Returns:
        - best_theory_index (int): Index of the selected theory.
        - n_success (int): Number of successful experiments.
        - n_failures (int): Number of failed experiments.
        """
        max_value = np.max(self.credences)
        max_indices = np.where(self.credences == max_value)[0]
        best_theory_index = np.random.choice(max_indices)
        
        n_success, n_experiments = self.bandit.experiment(best_theory_index, n_experiments)
        n_failures = n_experiments - n_success
        return best_theory_index, n_success, n_failures
        
    def beta_update(self, theory_index, n_success, n_failures):
        """
        Updates the agent's belief using Bayesian updating based on observed successes and failures.
        
        Parameters:
        - theory_index (int): Index of the theory being updated.
        - n_success (int): Number of successful experiments.
        - n_failures (int): Number of failed experiments.
        """
        self.alphas_betas[theory_index][0] += n_success
        self.alphas_betas[theory_index][1] += n_failures
        
        alpha = self.alphas_betas[theory_index][0]
        beta_param = self.alphas_betas[theory_index][1]  # Avoid using 'beta' as it conflicts with scipy.stats.beta
        
        new_credences = self.credences.copy()
        mean = beta.stats(alpha, beta_param, moments='m')
        new_credences[theory_index] = mean
        self.credences = new_credences
        
        if self.histories:
            self.credences_history.append(new_credences)
