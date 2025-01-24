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

    def __init__(self, uncertainty: float = 0.1) -> None:
        self.uncertainty = uncertainty
        self.p_old_theory = 0.5
        self.p_new_theory = 0.5 + uncertainty
        
    def experiment(self, n_experiments: int):
        """
        Performs an experiment using the new_theory.

        Args:
        - n_experiments (int): the number of experiments.
        """
        n_success = rd.binomial(n_experiments, self.p_new_theory)
        return n_success, n_experiments


class Agent:
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

    def __init__(self, id, uncertainty_problem: UncertaintyProblem):
        self.id = id
        self.uncertainty_problem = uncertainty_problem
        # I initialize with 1 rather than zero so that we can sample from the beta
        self.n_success: int = 1
        self.n_experiments: int = 1
        # For the beta agent
        self.accumulated_successes = np.zeros(1)+1
        self.accumulated_failures = np.zeros(1)+1      
        self.choice_history = []
        self.credence: float = rd.uniform(0, 1)
        self.credence_history = []
        self.credence_history.append(self.credence)
        self.epsilon=0.1
        
    def init_bayes(self):
        self.credence: float = rd.uniform(0, 1)
        self.credence_history = []
        self.credence_history.append(self.credence)
    # Instead of initializing with just alpha=beta=1, I ALSO initialize be sampling from the binomial/uncertainty problem
    def init_beta(self):
        n_success, n_experiments = self.uncertainty_problem.experiment(2)
        self.accumulated_successes+=n_success
        self.accumulated_failures+=(n_experiments-n_success)
        mean, var= beta.stats(self.accumulated_successes, self.accumulated_failures, moments='mv')
        #print(self.accumulated_successes/(self.accumulated_failures+self.accumulated_successes))
        self.credence = mean[0]
        self.credence_history = []
        self.credence_history.append(self.credence)

    def __str__(self):
        return (
            f"credence = {round(self.credence, 2)}, n_success = {self.n_success}, "
            f"n_experiments = {self.n_experiments}"
        )

    def experiment(self, n_experiments: int):
        """
        Performs an experiment with the given parameters.

        Args:
        - n_experiments (int): The total number of experiments.
        - uncertainty (float): The uncertainty in the experiment.
        """
        if self.credence > 0.5:
            self.n_success, self.n_experiments = self.uncertainty_problem.experiment(
                n_experiments
            )
            self.choice_history.append(1)
        else:
            self.n_success = 0
            self.n_experiments = 0
            self.choice_history.append(0)

    # I am not sure adding epsilon greedy helps because in this case all models will
    # achieve correct true consensus
    def egreedy_experiment(self, n_experiments: int):
        if np.random.rand() < self.epsilon:
            self.n_success, self.n_experiments = self.uncertainty_problem.experiment(
                n_experiments
            )
            self.choice_history.append(1)
        else:
            self.experiment(n_experiments)
            
    def bayes_update(self, n_success, n_experiments):
        """
        Updates the agent's credence using Bayes' rule. The basic setting is that the
        agent knows the probability of an old theory but does not know the probability
        of a new theory. The probability of the new theory is assumed to be either
        0.5 + uncertainty or 0.5 - uncertainty.

        Args:
        - n_success (int): The number of successful experiments.
        - n_experiments: The total number of experiments.
        """
        p_new_better = 0.5 + self.uncertainty_problem.uncertainty
        p_new_worse = 0.5 - self.uncertainty_problem.uncertainty
        n_failures = n_experiments - n_success
        credence_new_worse = 1 - self.credence
        likelihood_ratio_credence = credence_new_worse / self.credence
        likelihood_ratio_evidence_given_probability = (p_new_worse / p_new_better) ** (
            n_success - n_failures
        )
        self.credence = 1 / (
            1 + likelihood_ratio_credence * likelihood_ratio_evidence_given_probability
        )
        self.credence_history.append(self.credence)

    def beta_update(self,n_success,n_experiments):
        self.accumulated_successes += n_success
        self.accumulated_failures += (n_experiments-n_success)
        mean, var= beta.stats(self.accumulated_successes, self.accumulated_failures, moments='mv')
        self.credence = mean[0]
        self.credence_history.append(self.credence) # this is usually a vector to factor multiple theories

           
class Bandit:
    def __init__(self, p_theories=None):
        if p_theories is None:
            self.n_theories = 2
            self.p_theories = np.random.random(2)
        if p_theories is not None:
            self.n_theories = len(p_theories)
            self.p_theories = p_theories

    def experiment(self, theory, n_experiments):
        p_theory = self.p_theories[theory]
        n_success = rd.binomial(n_experiments, p_theory)
        return n_success, n_experiments



# class Bandit:
#     def __init__(self, p_theories=None):
#         if p_theories is None:
#             self.n_theories = 2
#             self.p_theories = np.random.random(2)
#         if p_theories is not None:
#             self.n_theories = len(p_theories)
#             self.p_theories = p_theories

#     def experiment(self, theory, n_experiments):
#         p_theory = self.p_theories[theory]
#         n_success = rd.binomial(n_experiments, p_theory)
#         return n_success, n_experiments


# class BetaAgent:
#     """Inspired by Zollman, Kevin J. S. 2010. The Epistemic Benefit of Transient
#     Diversity. Erkenntnis 72 (1): 17--35. https://doi.org/10.1007/s10670-009-9194-6.
#     (Especially sections 2 and 3.)

#     Attributes:
#     - id: The id of the BetaAgent
#     - beliefs (np.array): The beliefs of the agent. Each index of the array represents a
#     theory and contains an array the form [alpha (float), beta (float)]
#     representing the beta-distribution that models the agent's beliefs about that
#     theory.
#     - experiment_result (np.array): The result of the agent's last experiment.
#     Each index of the array represents a theory and contains an array of the form
#     [n_success (int), n_experiments (int)] representing the result of the experiment on
#     that theory, if any. If there is no experiment on a given theory, then the result of
#     the experiment on that theory is [0, 0].

#     Methods:
#     - __init__(self): Initializes the BetaAgent object.
#     - __str__(self): Returns a string representation of the BetaAgent object.
#     - n_theories (int): The number of theories under consideration.
#     - experiment(self, n_experiments, p_theories): Performs an experiment and updates
#     the agent's experiment_result.
#     - beta_update(self, experiment_results): Updates the agent's beliefs on the basis of
#     experiments. Experiments are represented by an array, where each index of the array
#     represents a theory and contains an array of the form [n_success (int),
#     n_experiments (int)] representing the result of the experiments.
#     """

#     def __init__(self, id, bandit: Bandit):
#         self.id = id
#         self.bandit = bandit
#         self.n_theories = bandit.n_theories
#         self.beliefs: np.array = np.array(
#             [[rd.random(), rd.random()] for _ in range(self.n_theories)]
#         )
#         self.experiment_result: np.array = np.array(
#             [[0, 0] for _ in range(self.n_theories)]
#         )

#     def __str__(self):
#         return (
#             # f"credence = {round(self.credence, 2)}, n_success = {self.n_success},"
#             # f"n_experiments = {self.n_experiments}, alpha = {self.alpha},"
#             # f"beta = {self.beta}"
#         )

#     def experiment(self, n_experiments: int):
#         """Performs an experiment and updates the agent's experiment_result.

#         Args:
#         - n_experiments (int): The number of experiments.
#         - p_theories (np.array): The probabilities of success, one for each theory."""
#         # Reset experiment_result
#         self.experiment_result = np.array([[0, 0] for _ in range(self.n_theories)])

#         decision = self.decision()

#         # Perform experiment on that theory and update experiment_result
#         n_success, n_experiments = self.bandit.experiment(decision, n_experiments)
#         self.experiment_result[decision] = [n_success, n_experiments]

#     def decision(self):
#         credences = np.array(
#             [
#                 self.beliefs[theory_id][0]
#                 / (self.beliefs[theory_id][0] + self.beliefs[theory_id][1])
#                 for theory_id in range(self.n_theories)
#             ]
#         )
#         return rd.choice(np.flatnonzero(credences == np.max(credences)))

#     def beta_update(self, experiment_results):
#         """Updates the agent's beliefs based on experiment_results.

#         Args:
#         - experiment_results (np.array): An array representing the results from
#         experiments represented in an array.
#         """
#         for theory in range(self.n_theories):
#             n_success = experiment_results[theory][0]
#             n_experiments = experiment_results[theory][1]
#             n_failures = n_experiments - n_success
#             self.beliefs[theory][0] += n_success
#             self.beliefs[theory][1] += n_failures
