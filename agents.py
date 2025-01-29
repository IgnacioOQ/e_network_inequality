# import numpy as np
# import numpy.random as rd
# from scipy.stats import beta
from imports import *  

class UncertaintyProblem:
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
        self.p_old_theory = 0.5
        self.p_new_theory = 0.5 + uncertainty
        
    def experiment(self, index, n_experiments: int):
        """
        Performs an experiment using the new_theory.

        Args:
        - n_experiments (int): the number of experiments.
        """
        if index == 0:
            n_success = rd.binomial(n_experiments, self.p_old_theory)
        if index == 1:
            n_success = rd.binomial(n_experiments, self.p_new_theory)
        return n_success, n_experiments


class BetaAgent:
    """
    An agent in a network epistemology playground, either Bayesian or Jeffreyan.
    Adapted from https://github.com/jweisber/sep-sen/blob/master/bg/agent.py

    Attributes:
    - credence (float): The agent's initial credence that the new theory is better.
    - n_success (int): The number of successful experiments.
    - n_experiments (int): The total number of experiments.

    Methods:
    - __init__(self): Initializes the Agent object.
    - __str__(self): Returns a string representation of the Agent object.
    - experiment(self, n_experiments, uncertainty): Performs an experiment.
    - bayes_update(self, n_success, n_experiments, uncertainty): Updates the agent's
    credence using Bayes' rule.
    - jeffrey_update(self, neighbor, uncertainty, mistrust_rate): Updates the agent's
    credence using Jeffrey's rule.
    """

    def __init__(self, id, uncertainty_problem = UncertaintyProblem, histories=False):
        self.id = id
        self.uncertainty_problem = uncertainty_problem
        
        # Initializing Beta Agent
        # For the beta agent, for each of the two theories we store accumulated successess and failures
        # initialize with one success and one failure for each theory
        self.alphas_betas = np.array([np.array([1,1]),np.array([1,1])])
        mean = beta.stats(1, 1, moments='m')        
        self.credences = np.array([mean,mean])
        self.histories = histories
        if self.histories:
            self.credences_history = []
            self.credences_history.append(self.credences)
        # self.choice_history = []
        # self.epsilon=0.1
        
    # Fully greedy update (no epsilon)
    def greedy_experiment(self, n_experiments: int):
        # I want to break ties at random
        # Find all indices of the maximum value
        max_value = np.max(self.credences)
        max_indices = np.where(self.credences == max_value)[0]
        # Randomly choose one of the indices
        best_theory_index = np.random.choice(max_indices)

        #print(best_theory_index)
        n_success, n_experiments = self.uncertainty_problem.experiment(best_theory_index,n_experiments)
        n_failures = n_experiments - n_success
        return best_theory_index, n_success, n_failures
                    
    def beta_update(self,theory_index, n_success, n_failures):
        # update alphas and betas
        # print(theory_index)
        self.alphas_betas[theory_index][0]+= n_success
        self.alphas_betas[theory_index][1]+= n_failures
        # update credences
        alpha = self.alphas_betas[theory_index][0]
        b = self.alphas_betas[theory_index][1] # cant use 'beta' because thats the stats function
        new_credences = self.credences.copy()
        mean= beta.stats(alpha, b, moments='m')
        new_credences[theory_index] = mean
        self.credences = new_credences
        # self.credences[theory_index] = mean
        if self.histories:
            self.credences_history.append(new_credences) # this is usually a vector to factor multiple theories
