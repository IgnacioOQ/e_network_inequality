import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent
from net_epistemology.core.model import Model
from net_epistemology.utils.network_utils import *
from net_epistemology.utils.network_generation import *
from net_epistemology.simulation.simulation_functions import *

def main():
    print("Starting simulation test...")
    n_simulations = 2
    G_default = barabasi_albert_directed(200, 5)

    num_cores = cpu_count()
    print(f"Using {num_cores} cores")

    generate_params_with_G = partial(generate_parameters, G=G_default)

    with Pool(num_cores) as pool:
        param_dict = list(tqdm.tqdm(pool.imap_unordered(generate_params_with_G, range(n_simulations)), total=n_simulations))

    print(f"Generated parameters for {len(param_dict)} simulations")

    # Run simulations
    with Pool(num_cores) as pool:
        simulation_results = list(tqdm.tqdm(pool.imap_unordered(run_simulation_with_params, param_dict),
                                            total=len(param_dict), desc="Running simulations"))

    basic_results_df = pd.DataFrame(simulation_results)
    print("Simulation results dataframe shape:", basic_results_df.shape)

    if basic_results_df.shape[0] == n_simulations:
        print("Test Passed: correct number of simulations run.")
    else:
        print(f"Test Failed: Expected {n_simulations} results, got {basic_results_df.shape[0]}")
        sys.exit(1)

if __name__ == "__main__":
    main()
