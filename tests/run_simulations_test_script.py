#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import sys
import os

# Add src to path to allow importing net_epistemology without installing the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# In[7]:


from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent
from net_epistemology.core.model import Model
from net_epistemology.utils.network_utils import *
# from network_randomization import *
from net_epistemology.utils.network_generation import *
from net_epistemology.simulation.simulation_functions import *


# # Testing Generated

# In[8]:


n_simulations = 4 # Reduced for speed
G_default = barabasi_albert_directed(50,2) # Reduced size for speed

num_cores = 2 # Fixed small number for test
print(num_cores)

# Define a partial function to pass G_perceptron to generate_parameters_empir
# This ensures that generate_parameters_empir is called with the correct argument within the pool
# The 'partial' function allows you to create a new function with some of the arguments pre-filled.
from functools import partial
generate_params_with_G = partial(generate_parameters, G=G_default)

if __name__ == '__main__':
    with Pool(num_cores) as pool:
        # Use tqdm to display a progress bar
        # Now, 'generate_params_with_G' is the function that will be executed by each worker.
        # Each worker will receive an index from 'range(n_simulations)' as its argument,
        # which is ignored in 'generate_params_with_G' but is required by the 'imap_unordered' function.
        param_dict = list(tqdm.tqdm(pool.imap_unordered(generate_params_with_G, range(n_simulations)), total=n_simulations))


    # In[9]:


    print(len(param_dict))
    # param_dict[0]


    # In[10]:


    # Run simulations in parallel
    with Pool(num_cores) as pool:
        simulation_results = list(tqdm.tqdm(pool.imap_unordered(run_simulation_with_params, param_dict),
                                            total=len(param_dict), desc="Running simulations"))

    # Convert results to a DataFrame
    basic_results_df = pd.DataFrame(simulation_results)
    # display(basic_results_df) # No display in script


    # In[11]:


    # basic_results_df.to_csv("basic_results_df.csv", index=False)  # Saves without index
    # scatter_plot(basic_results_df)
    # scatter_plot(basic_results_df, target_variable="convergence_step")


    # In[ ]:





    # # Testing Empirical

    # In[12]:


    # with open('./empirical_networks/perc_pruned_lcc.pkl', 'rb') as f:
    #   G_perceptron = pickle.load(f)

    # n_agents = G_perceptron.number_of_nodes()
    # print(n_agents)

    # # Create a mapping from node names to indexes
    # mapping = {node: index for index, node in enumerate(G_perceptron.nodes())}

    # # Relabel the nodes in the graph
    # G_perceptron_indexed = nx.relabel_nodes(G_perceptron, mapping)
    # G_default = G_perceptron_indexed


    # n_simulations = 10

    # num_cores = cpu_count()  # Get the number of available CPU cores
    # print(num_cores)

    # # Define a partial function to pass G_perceptron to generate_parameters_empir
    # # This ensures that generate_parameters_empir is called with the correct argument within the pool
    # # The 'partial' function allows you to create a new function with some of the arguments pre-filled.
    # from functools import partial
    # generate_params_with_G = partial(generate_parameters, G=G_default)

    # with Pool(num_cores) as pool:
    #     # Use tqdm to display a progress bar
    #     # Now, 'generate_params_with_G' is the function that will be executed by each worker.
    #     # Each worker will receive an index from 'range(n_simulations)' as its argument,
    #     # which is ignored in 'generate_params_with_G' but is required by the 'imap_unordered' function.
    #     param_dict = list(tqdm.tqdm(pool.imap_unordered(generate_params_with_G, range(n_simulations)), total=n_simulations))

    # param_dict[0]


    # # In[ ]:


    # # Run simulations in parallel
    # with Pool(num_cores) as pool:
    #     simulation_results = list(tqdm.tqdm(pool.imap_unordered(run_simulation_wrapper, param_dict),
    #                                         total=len(param_dict), desc="Running simulations"))

    # # Convert results to a DataFrame
    # basic_results_df = pd.DataFrame(simulation_results)
    # display(basic_results_df)
    # scatter_plot(basic_results_df)
    # scatter_plot(basic_results_df, target_variable="convergence_step")
