# import numpy as np
# import numpy.random as rd
# from scipy.stats import beta
from imports import *  

class Bandit:
    """
    The problem of theory choice involves two theories where the new_theory is better
    by the margin of uncertainty.

    Attributes:
    - uncertainty (float): The uncertainty in the theory choice.

    Methods
    - experiment(self, n_experiments): Performs an experiment using the new_theory.
    """

    def __init__(self, uncertainty: float = 0.1):
        self.uncertainty = uncertainty
        self.p_bad_theory = 0.5
        self.p_good_theory = 0.5 + uncertainty
        
    def experiment(self, index, n_experiments: int):
        """
        Performs an experiment using the new_theory.

        Args:
        - n_experiments (int): the number of experiments.
        """
        if index == 0:
            n_success = rd.binomial(n_experiments, self.p_bad_theory)
        if index == 1:
            n_success = rd.binomial(n_experiments, self.p_good_theory)
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

    def __init__(self, id, Bandit, histories=False):
        self.id = id
        self.Bandit = Bandit
        
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
        
        n_success, n_experiments = self.uncertainty_problem.experiment(best_theory_index, n_experiments)
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
