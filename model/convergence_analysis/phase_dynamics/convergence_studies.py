#!/usr/bin/env python
# coding: utf-8

# This file is a script, not an importable module: the analysis below runs at
# module level, so importing it would execute a full simulation (minutes) and
# write plots into the working tree. Fail fast and say so instead.
if __name__ != "__main__":
    raise ImportError(
        f"{__name__} is an analysis script, not a library module. "
        f"Run it directly: python {__file__}"
    )

# # Convergence Studies for Vectorized Beta Agent
#
# Shows how the mean belief change (prior vs posterior) decreases over time steps,
# demonstrating convergence of the simulation.

import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import networkx as nx

from model.vectorized_model import VectorizedModel

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# ## Load Empirical Network

print("Loading pud_final.json network...")
network_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'networks', 'citation_data', 'pud_final.json')

with open(network_path, 'r') as f:
    network_data = json.load(f)

network = nx.node_link_graph(network_data, edges="links")
print(f"Network: {len(network.nodes())} nodes, {len(network.edges())} edges")

# ## Run Simulation with Convergence Tracking

print("\nRunning simulation...")

model = VectorizedModel(
    network=network,
    n_experiments=10,
    uncertainty=0.1,
    agent_type="beta",
    tolerance_stopping=True,
    tolerance=1e-6,
    compute_convergence=True,
)

model.run_simulation(number_of_steps=10**6, show_bar=True)

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
output_path = os.path.join(os.path.dirname(__file__), f'convergence_plot_{model.n_steps}steps.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"\nPlot saved to '{output_path}'")

# Print final convergence values
print(f"\nFinal belief change (step {model.n_steps}):")
print(f"  Theory 0: {history[-1, 0]:.2e}")
print(f"  Theory 1: {history[-1, 1]:.2e}")
