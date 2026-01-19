#!/usr/bin/env python
# coding: utf-8

# Add src to path to allow importing net_epistemology without installing the package
import matplotlib
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# # Basic Testing
#
# In this notebook we test that the main files work well.

# ## Setup

from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent, BayesAgent
from net_epistemology.core.model import Model

def main():
    # n_agents = 100
    # my_network = nx.gnp_random_graph(n_agents, p=0.2, directed=True) #nx.complete_graph(n_agents, create_using=nx.DiGraph())

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


    # ## Try with Bayes Agent

    seed=420
    my_model = Model(my_network, n_experiments=10, uncertainty=0.001,
                     histories=True,sampling_update=True,variance_stopping = False,directed_network = True,
                     seed=seed,seeded=False, agent_type='bayes')
    my_model.run_simulation(number_of_steps=1000,show_bar=True) # Reduced steps for smoke test
    print('steps: ',my_model.n_steps)
    print('conclusion: ',my_model.conclusion)
    print('conclusion core', my_model.conclusion_core)
    # df = pd.DataFrame(my_model.agent_histories)
    # df.head(3)

    # Plot mean credence for Bayes
    # Credences are 1D arrays (scalar per agent)
    df_bayes = pd.DataFrame(my_model.agent_histories).T
    df_bayes.head(3)
    mean_credence = df_bayes.mean(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_credence, label='Mean Credence')
    plt.title('Bayes Agent: Average Credence Evolution')
    plt.xlabel('Steps')
    plt.ylabel('Credence')
    plt.legend()
    # plt.show() # Headless

    # ## Try with Beta Agent

    seed=420
    my_model = Model(my_network, n_experiments=10, uncertainty=0.001,
                     histories=True,sampling_update=True,variance_stopping = False,directed_network = True,
                     seed=seed,seeded=False, agent_type='beta')

    my_model.run_simulation(number_of_steps=1000,show_bar=True) # Reduced steps for smoke test
    print('steps: ',my_model.n_steps)
    print('conclusion: ',my_model.conclusion)
    df = pd.DataFrame(my_model.agent_histories)
    df.head(3)

    #Extract the first coordinate (x) for each pair and calculate column-wise mean
    x_means = df.map(lambda pair: pair[0]).mean()
    y_means = df.map(lambda pair: pair[1]).mean()
    plt.figure()
    plt.plot(x_means)
    plt.plot(y_means)

    # Extract the first coordinate (x) for each pair
    x_values = df.map(lambda pair: pair[0])

    # Plot the first coordinate for each row
    plt.figure(figsize=(10, 6))
    for row_idx in range(x_values.shape[0]):
        plt.plot(x_values.columns, x_values.iloc[row_idx, :], label=f'Row {row_idx+1}' if row_idx < 5 else None)

if __name__ == "__main__":
    main()
