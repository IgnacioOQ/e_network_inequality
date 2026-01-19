#!/usr/bin/env python
# coding: utf-8

# Add src to path to allow importing net_epistemology without installing the package
import matplotlib
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent
from net_epistemology.core.model import Model
from net_epistemology.utils.network_utils import *
# from network_randomization import *
from net_epistemology.utils.network_generation import *
from net_epistemology.simulation.simulation_functions import *
from functools import partial

def main():
    # # Testing Generated

    n_simulations = 2 # Reduced for smoke test
    G_default = barabasi_albert_directed(200,5)

    num_cores = cpu_count()  # Get the number of available CPU cores
    print(num_cores)

    # Define a partial function to pass G_perceptron to generate_parameters_empir
    # This ensures that generate_parameters_empir is called with the correct argument within the pool
    # The 'partial' function allows you to create a new function with some of the arguments pre-filled.
    generate_params_with_G = partial(generate_parameters, G=G_default)

    with Pool(num_cores) as pool:
        # Use tqdm to display a progress bar
        # Now, 'generate_params_with_G' is the function that will be executed by each worker.
        # Each worker will receive an index from 'range(n_simulations)' as its argument,
        # which is ignored in 'generate_params_with_G' but is required by the 'imap_unordered' function.
        param_dict = list(tqdm.tqdm(pool.imap_unordered(generate_params_with_G, range(n_simulations)), total=n_simulations))


    print(len(param_dict))
    print(param_dict[0])


    # Run simulations in parallel
    with Pool(num_cores) as pool:
        simulation_results = list(tqdm.tqdm(pool.imap_unordered(run_simulation_with_params, param_dict),
                                            total=len(param_dict), desc="Running simulations"))

    # Convert results to a DataFrame
    basic_results_df = pd.DataFrame(simulation_results)
    # display(basic_results_df)


    # basic_results_df.to_csv("basic_results_df.csv", index=False)  # Saves without index
    scatter_plot(basic_results_df)
    scatter_plot(basic_results_df, target_variable="convergence_step")


    # # Testing Empirical

    # I found that perceptron_final.pkl exists, let's use it.
    try:
        with open(os.path.join(os.path.dirname(__file__), '../data/empirical_networks/perceptron_final.pkl'), 'rb') as f:
            G_perceptron = pickle.load(f)

        n_agents = G_perceptron.number_of_nodes()
        print(n_agents)

        # Create a mapping from node names to indexes
        mapping = {node: index for index, node in enumerate(G_perceptron.nodes())}

        # Relabel the nodes in the graph
        G_perceptron_indexed = nx.relabel_nodes(G_perceptron, mapping)
        G_default = G_perceptron_indexed


        n_simulations = 2 # Reduced

        num_cores = cpu_count()  # Get the number of available CPU cores
        print(num_cores)

        # Define a partial function to pass G_perceptron to generate_parameters_empir
        # This ensures that generate_parameters_empir is called with the correct argument within the pool
        # The 'partial' function allows you to create a new function with some of the arguments pre-filled.
        generate_params_with_G = partial(generate_parameters, G=G_default)

        with Pool(num_cores) as pool:
            # Use tqdm to display a progress bar
            # Now, 'generate_params_with_G' is the function that will be executed by each worker.
            # Each worker will receive an index from 'range(n_simulations)' as its argument,
            # which is ignored in 'generate_params_with_G' but is required by the 'imap_unordered' function.
            param_dict = list(tqdm.tqdm(pool.imap_unordered(generate_params_with_G, range(n_simulations)), total=n_simulations))

        print(param_dict[0])

        # Run simulations in parallel
        # Replaced run_simulation_wrapper with run_simulation_with_params
        with Pool(num_cores) as pool:
            simulation_results = list(tqdm.tqdm(pool.imap_unordered(run_simulation_with_params, param_dict),
                                                total=len(param_dict), desc="Running simulations"))

        # Convert results to a DataFrame
        basic_results_df = pd.DataFrame(simulation_results)
        # display(basic_results_df)
        scatter_plot(basic_results_df)
        scatter_plot(basic_results_df, target_variable="convergence_step")
    except Exception as e:
        print(f"Skipping Empirical Testing due to error: {e}")

if __name__ == "__main__":
    main()
