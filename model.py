# import numpy as np
# import tqdm
# import pandas as pd
from imports import *  
from agents import BetaAgent, Bandit

class Model:
    """
    Adapted from https://github.com/jweisber/sep-sen/blob/master/bg/agent.py
    Represents an agent in a network epistemology playground.

    Attributes:
    - network: The network.
    - n_experiments (int): The number of experiments per step.
    - agent_type (str): The type of agents, "bayes", "beta" or "jeffrey"
    - uncertainty (float): The uncertainty in the experiment.
    - p_theories (list): The success probabilities of the theories.

    Methods:
    - __init__(self): Initializes the Model object.
    - __str__(self): Returns a string representation of the Model object.
    - simulation(self, number_of_steps): Runs a simulation of the model.
    - step(self): Updates the model with one step, consisting of experiments and
    updates.
    - agents_experiment(self): Updates the model with one round of experiments.
    - agents_update(self): Updates the model with one round of updates.
    """

    def __init__(
        self,
        network,
        n_experiments: int,
        # agent_type: str,
        uncertainty: float = None,
        tolerance = 1e-04,
        histories = False,
        sampling_update = False,
        variance_stopping = False,
        *args,
        **kwargs
    ):
        self.network = network
        self.n_agents = len(network.nodes)
        #print(self.n_agents)
        self.n_experiments = n_experiments
        # else:
        self.bandit = Bandit(uncertainty)
        self.agents = [
            BetaAgent(i, self.bandit,histories=histories,sampling_update=sampling_update) for i in range(self.n_agents)
        ]
        # self.agent_type = agent_type
        self.n_steps = 0
        self.tolerance = tolerance
        self.histories = histories
        self.variance_stopping = variance_stopping
        
    def run_simulation(
        self, number_of_steps: int = 10**6, show_bar: bool = False, *args, **kwargs
    ):
        """Runs a simulation of the model and sets model.conclusion.

        Args:
            number_of_steps (int, optional): Number of steps in the simulation
            (it will end sooner if the stop condition is met). Defaults to 10**6."""

        def stop_condition(credences_prior, credences_post) -> bool:
            # the tolerance is too tight, originally: rtol=1e-05, atol=1e-08
            return np.allclose(credences_prior, credences_post,rtol=self.tolerance, atol=self.tolerance)
        
        def true_consensus_condition(credences: np.array) -> float:
            second_coordinates = np.array([credences[1] for pair in credences])
            # Count how many pairs have the second coordinate larger than the first
            counts = np.sum([pair[1] > pair[0] for pair in credences])
            return counts/len(credences) #(second_coordinates > 0.5).mean()

        iterable = range(number_of_steps)

        if show_bar:
            iterable = tqdm.tqdm(iterable)

        alternative_stop = False
        self.conclusion_alternative_stop = False
        for _ in iterable:
            # Lots of if elses but oh well
            if self.variance_stopping:
                betas_prior = np.array([agent.alphas_betas for agent in self.agents])
                mv_prior = np.array([beta.stats(prior[0], prior[1], moments='mv') for prior in betas_prior])
            else:
                credences_prior = np.array([agent.credences for agent in self.agents])
            
            self.step()
            
            if self.variance_stopping:
                betas_post = np.array([agent.alphas_betas for agent in self.agents])
                mv_post = np.array([beta.stats(post[0], post[1], moments='mv') for post in betas_post])
            else:
                credences_post = np.array([agent.credences for agent in self.agents])

            if self.variance_stopping:
                if stop_condition(mv_prior, mv_post):
                    self.conclusion = true_consensus_condition(credences_post)
                    if not alternative_stop:
                        self.conclusion_alternative_stop = self.conclusion
                    break
            else:
                if stop_condition(credences_prior, credences_post):
                    self.conclusion = true_consensus_condition(credences_post)
                    if not alternative_stop:
                        self.conclusion_alternative_stop = self.conclusion
                    break
                self.conclusion = true_consensus_condition(credences_post)  # We should set this even if we don't break, right??? - MN
        
        if self.histories:
            self.add_agents_history()
            

    def step(self):
        """Updates the model with one step, consisting of experiments and updates."""
        self.n_steps+=1
        experiments_results = self.agents_experiment()
        self.agents_update(experiments_results)

    def agents_experiment(self):
        experiments_results = {}
        for agent in self.agents:
            theory_index, n_success, n_failures = agent.experiment(self.n_experiments)
            experiments_results[agent.id]=[theory_index, n_success, n_failures]
        # print('experiments done')
        return experiments_results

    def agents_update(self,experiments_results):
        for agent in self.agents:
            # gather information from neighbors
            # if the graph is directed, the neighbors are the successors
            # https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.neighbors.html
            # I will keep it like this so its compatible with undirected graphs
            # BUT the directed citation networks have to go from citing to cited
            # namely inverse of the direction of information flow.
            # and in studying gini we need to consider in-degree mostly
            neighbor_nodes = list(self.network.neighbors(agent.id))
            theories_exp_results = np.array([np.array([0,0]),np.array([0,0])])
            results = experiments_results[agent.id]
            theory_index = results[0]
            theories_exp_results[theory_index][0]+=results[1]
            theories_exp_results[theory_index][1]+=results[2]
            for id in neighbor_nodes:
                results = experiments_results[id]
                theory_index = results[0]
                theories_exp_results[theory_index][0]+=results[1] #n_success
                theories_exp_results[theory_index][1]+=results[2] #n_failures

            # update
            agent.beta_update(0,theories_exp_results[0][0], theories_exp_results[0][1])
            agent.beta_update(1,theories_exp_results[1][0], theories_exp_results[1][1])

                
    def add_agents_history(self):
        self.agent_histories = [agent.credences_history for agent in self.agents]
        #agent_choices = [agent.choice_history for agent in self.agents]
        #self.agents_choices = pd.DataFrame(agent_choices)
