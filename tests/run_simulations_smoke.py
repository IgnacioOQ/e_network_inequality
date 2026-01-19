import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm
import uuid
import random

from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent
from net_epistemology.core.model import Model
from net_epistemology.utils.network_utils import *
from net_epistemology.utils.network_generation import *
from net_epistemology.simulation.simulation_functions import run_simulation_with_params, network_statistics

# Wrapper to run simulation with reduced steps for smoke test
def run_sim_smoke(params):
    return run_simulation_with_params(params, number_of_steps=20, show_bar=False)

# Define a local generate_parameters to control n_experiments for smoke test
def generate_parameters_smoke(_, G):
    unique_id = uuid.uuid4().hex
    process_seed = int.from_bytes(os.urandom(4), byteorder="little")
    # rd.seed(process_seed) # Don't need to seed numpy here necessarily for smoke test

    uncertainty = 0.001
    n_experiments = 10 # Reduced for smoke test

    # Do randomization (simplified for smoke test)
    # n_edges = int(0.1 * len(G.edges()))
    # randomized_network = randomize_network(G, n_edges=n_edges)
    # Just use G for speed
    randomized_network = G

    params = {
        "randomized": True,
        "unique_id": unique_id,
        "n_agents": int(len(randomized_network.nodes)),
        "network": randomized_network,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        "p_rewiring": 0.1,
    }
    stats = network_statistics(randomized_network)
    for stat in stats.keys():
        params[stat] = stats[stat]

    return params

def test_simulations():
    print("Starting simulation smoke test...")
    n_simulations = 2 # Reduced
    G_default = barabasi_albert_directed(20, 2) # Reduced size

    num_cores = min(2, cpu_count()) # Limit cores
    print(f"Using {num_cores} cores")

    generate_params_with_G = partial(generate_parameters_smoke, G=G_default)

    # Generate parameters
    print("Generating parameters...")
    # We can just run this serially for 2 sims
    param_dict = [generate_params_with_G(i) for i in range(n_simulations)]

    print(f"Generated {len(param_dict)} parameter sets.")
    print(f"Sample params: {param_dict[0]['n_experiments']} experiments, {param_dict[0]['n_agents']} agents")

    # Run simulations
    print("Running simulations...")
    # Using run_simulation_with_params instead of wrapper
    # Also reducing number_of_steps in the call if possible, but run_simulation_with_params takes it as kwarg

    with Pool(num_cores) as pool:
        simulation_results = list(tqdm.tqdm(pool.imap_unordered(run_sim_smoke, param_dict),
                                            total=len(param_dict), desc="Running simulations"))

    # Convert results to a DataFrame
    basic_results_df = pd.DataFrame(simulation_results)
    print("Simulation results:")
    print(basic_results_df.head())

    if 'share_of_correct_agents_at_convergence' in basic_results_df.columns:
        print("Verification successful: Results contain 'share_of_correct_agents_at_convergence'")
    else:
        print("Verification failed: Missing expected columns")

    print("Simulation smoke test completed.")

if __name__ == "__main__":
    test_simulations()
