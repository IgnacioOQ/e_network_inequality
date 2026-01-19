#!/usr/bin/env python
# coding: utf-8

# Add src to path to allow importing net_epistemology without installing the package
import matplotlib
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# # Basic Testing (Vectorized)
#
# In this notebook we test that the vectorized files work well and produce similar results to the original model.

# ## Setup

from net_epistemology.utils.imports import *
from net_epistemology.core.vectorized_model import VectorizedModel
import matplotlib.pyplot as plt

def main():
    # Save and load pud_final as JSON
    import json
    from networkx.readwrite import json_graph
    from pathlib import Path

    # Fix relative path
    JSON_PATH = Path(os.path.join(os.path.dirname(__file__), "../data/empirical_networks/pud_final.json"))

    # Save to JSON
    def save_network_json(G, path):
        data = json_graph.node_link_data(G)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved network to {path}")

    # Load from JSON
    def load_network_json(path):
        with open(path, "r") as f:
            data = json.load(f)
        # Fix for networkx expecting 'edges' instead of 'links' or vice-versa
        if "links" in data and "edges" not in data:
            data["edges"] = data["links"]
        G = json_graph.node_link_graph(data)
        print(f"Loaded network from {path}")
        return G

    # Example usage:
    # save_network_json(pud_final, JSON_PATH)
    try:
        my_network = load_network_json(JSON_PATH)
    except Exception as e:
        print(f"Could not load network: {e}")
        print("Falling back to random network.")
        my_network = nx.gnp_random_graph(100, 0.2, directed=True)

    # n_agents = 1000
    # my_network = nx.gnp_random_graph(n_agents, p=0.1, directed=True) #nx.complete_graph(n_agents, create_using=nx.DiGraph())


    # ## Try with Bayes Agent (Vectorized)

    seed=420
    my_model = VectorizedModel(my_network, n_experiments=10, uncertainty=0.001,
                     histories=True,sampling_update=True,variance_stopping = False,directed_network = True,
                     seed=seed,seeded=False, agent_type='bayes')
    my_model.run_simulation(number_of_steps=1000,show_bar=True) # Reduced steps
    print('steps: ',my_model.n_steps)
    print('conclusion: ',my_model.conclusion)
    print('conclusion core', my_model.conclusion_core)

    df_bayes = pd.DataFrame(my_model.credences_history).T
    df_bayes.head(3)


    # Plot mean credence for Bayes
    # Credences are 1D arrays (scalar per agent)
    mean_credence = df_bayes.mean(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_credence, label='Mean Credence')
    plt.title('Bayes Agent: Average Credence Evolution')
    plt.xlabel('Steps')
    plt.ylabel('Credence')
    plt.legend()
    # plt.show()


    # ## Try with Beta Agent (Vectorized)

    seed=420
    my_model = VectorizedModel(my_network, n_experiments=10, uncertainty=0.001,
                     histories=True,sampling_update=True,variance_stopping = False,directed_network = True,
                     seed=seed,seeded=False, agent_type='beta')

    my_model.run_simulation(number_of_steps=1000,show_bar=True) # Reduced steps
    print('steps: ',my_model.n_steps)
    print('conclusion: ',my_model.conclusion)

    # agent_histories in VectorizedModel is a list of lists of numpy arrays
    df = pd.DataFrame(my_model.credences_history).T # Transpose because history[agent] is list of steps
    df.head(3)


    # Extract the first coordinate (x) for each pair and calculate column-wise mean
    # In vectorized model, credences are stored as arrays [c0, c1].
    x_means = df.map(lambda pair: pair[0]).mean(axis=1)
    y_means = df.map(lambda pair: pair[1]).mean(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_means, label='Theory 0')
    plt.plot(y_means, label='Theory 1')
    plt.title('Beta Agent: Average Credence Evolution')
    plt.legend()
    # plt.show()


    # Extract the first coordinate (x) for each pair
    x_values = df.map(lambda pair: pair[0])

    # Plot the first coordinate for each row (agent)
    plt.figure(figsize=(10, 6))
    # Plot a subset of agents to avoid clutter if N is large
    for agent_idx in range(min(10, x_values.shape[1])):
        plt.plot(x_values[agent_idx], label=f'Agent {agent_idx}')
    plt.title('Beta Agent: Individual Credence (Theory 0)')
    # plt.show()

if __name__ == "__main__":
    main()
