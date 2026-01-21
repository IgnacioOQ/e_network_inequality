#!/usr/bin/env python
# coding: utf-8

# # Convergence Studies for Vectorized Beta Agent
#
# Shows how the mean belief change (prior vs posterior) decreases over time steps,
# demonstrating convergence of the simulation.

import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import networkx as nx

from net_epistemology.core.vectorized_model import VectorizedModel

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# ## Load Empirical Network

print("Loading pud_final.json network...")
network_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'empirical_networks', 'pud_final.json')

with open(network_path, 'r') as f:
    network_data = json.load(f)

network = nx.node_link_graph(network_data)
print(f"Network: {len(network.nodes())} nodes, {len(network.edges())} edges")

# ## Run Simulation with Convergence Tracking

print("\nRunning simulation...")

model = VectorizedModel(
    network=network,
    n_experiments=10,
    uncertainty=0.001,
    agent_type="beta",
    tstep_stopping=True,  # Run for all steps
    compute_convergence=True,
)

model.run_simulation(number_of_steps=10000, show_bar=True)

print(f"\nSimulation completed: {model.n_steps} steps")
print(f"Conclusion (share believing truth): {model.conclusion:.4f}")

# ## Plot Convergence

# Extract belief change history
history = np.array(model.belief_change_history)
steps = np.arange(1, len(history) + 1)

# Create convergence plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(steps, history[:, 0], label='Theory 0 (Bad)', alpha=0.8, linewidth=1)
ax.plot(steps, history[:, 1], label='Theory 1 (Good)', alpha=0.8, linewidth=1)

ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('Mean Absolute Belief Change (Prior → Posterior)', fontsize=12)
ax.set_title('Convergence of Beta Agent Beliefs Over Time', fontsize=14)
ax.legend(fontsize=11)

# Use log scale for y-axis to better see convergence
ax.set_yscale('log')
ax.set_xlim(1, len(history))

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_plot_10000.png', dpi=150, bbox_inches='tight')

print("\nPlot saved to 'convergence_plot_10000.png'")

# Print final convergence values
print(f"\nFinal belief change (step {model.n_steps}):")
print(f"  Theory 0: {history[-1, 0]:.2e}")
print(f"  Theory 1: {history[-1, 1]:.2e}")
