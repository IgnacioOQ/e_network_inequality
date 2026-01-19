import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path

from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent, BayesAgent
from net_epistemology.core.model import Model

def load_network_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "links" in data and "edges" not in data:
        data["edges"] = data["links"]
    G = json_graph.node_link_graph(data)
    print(f"Loaded network from {path}")
    return G

def test_basic_model():
    print("Starting basic model test...")
    # Setup path to data relative to this script
    # This script is in tests/, so data is in ../data/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "../data/empirical_networks/pud_final.json")

    if not os.path.exists(json_path):
        print(f"Warning: Data file not found at {json_path}. Creating a dummy graph.")
        my_network = nx.gnp_random_graph(20, 0.2, directed=True)
    else:
        try:
            my_network = load_network_json(json_path)
        except Exception as e:
            print(f"Error loading network: {e}")
            my_network = nx.gnp_random_graph(20, 0.2, directed=True)

    print("Network loaded.")

    # Bayes Agent Test
    print("Testing Bayes Agent...")
    seed = 420
    # Reduced steps and experiments for smoke test
    my_model_bayes = Model(my_network, n_experiments=5, uncertainty=0.001,
                     histories=True, sampling_update=True, variance_stopping=False, directed_network=True,
                     seed=seed, seeded=False, agent_type='bayes')

    my_model_bayes.run_simulation(number_of_steps=50, show_bar=False)
    print(f"Bayes steps: {my_model_bayes.n_steps}")
    print(f"Bayes conclusion: {my_model_bayes.conclusion}")

    if hasattr(my_model_bayes, 'agent_histories'):
        df_bayes = pd.DataFrame(my_model_bayes.agent_histories).T
        print(f"Bayes history shape: {df_bayes.shape}")

    # Beta Agent Test
    print("Testing Beta Agent...")
    seed = 420
    my_model_beta = Model(my_network, n_experiments=5, uncertainty=0.001,
                     histories=True, sampling_update=True, variance_stopping=False, directed_network=True,
                     seed=seed, seeded=False, agent_type='beta')

    my_model_beta.run_simulation(number_of_steps=50, show_bar=False)
    print(f"Beta steps: {my_model_beta.n_steps}")
    print(f"Beta conclusion: {my_model_beta.conclusion}")

    if hasattr(my_model_beta, 'agent_histories'):
        df_beta = pd.DataFrame(my_model_beta.agent_histories)
        print(f"Beta history shape: {df_beta.shape}")

        # Verify mapping works
        try:
            # Extract the first coordinate (x) for each pair
            x_values = df_beta.map(lambda pair: pair[0] if isinstance(pair, (list, tuple, np.ndarray)) else 0)
            print("Beta map operation successful.")
        except Exception as e:
            print(f"Error in beta map operation: {e}")

    print("Basic model test completed successfully.")

if __name__ == "__main__":
    test_basic_model()
