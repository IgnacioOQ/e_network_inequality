#!/usr/bin/env python
# coding: utf-8

# # Root Node Influence Analysis
#
# Testing hypothesis: The final epistemic state of the network is determined by
# root nodes' beliefs, weighted by their influence (number of descendants).

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from scipy import stats

from net_epistemology.core.vectorized_model import VectorizedModel
from net_epistemology.simulation.vectorized_simulation_functions import run_vectorized_simulation_with_params

plt.style.use('seaborn-v0_8-whitegrid')

# ## Load Network

print("Loading pud_final.json network...")
network_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'empirical_networks', 'pud_final.json')

with open(network_path, 'r') as f:
    network_data = json.load(f)

network = nx.node_link_graph(network_data, edges="links")
n_agents = len(network.nodes())
print(f"Network: {n_agents} nodes, {len(network.edges())} edges")

# Analyze network structure
in_degrees = dict(network.in_degree())
root_nodes = [n for n, d in in_degrees.items() if d == 0]
print(f"Number of root nodes: {len(root_nodes)}")

# ## Run Multiple Simulations

print("\nRunning simulations with root analysis...")

n_simulations = 100
results = []

for sim_idx in tqdm(range(n_simulations), desc="Simulations"):
    param_dict = {
        "network": network,
        "n_experiments": 10,
        "uncertainty": 0.001,
        "sim_index": sim_idx,
    }
    
    result = run_vectorized_simulation_with_params(
        param_dict,
        agent_type="beta",
        tstep_stopping=True,
        number_of_steps=5000,
        compute_root_analysis=True,
    )
    
    results.append(result)

# ## Extract Data for Analysis

# Extract key metrics
conclusions = np.array([r["share_of_correct_agents_at_convergence"] for r in results])
weighted_truth_shares = np.array([r["weighted_truth_share"] for r in results])
unweighted_truth_shares = np.array([r["unweighted_truth_share"] for r in results])
n_roots = results[0]["n_roots"]

print(f"\n{'='*60}")
print("Summary Statistics")
print(f"{'='*60}")
print(f"Number of roots: {n_roots}")
print(f"Network conclusion: {conclusions.mean():.4f} ± {conclusions.std():.4f}")
print(f"Weighted root truth share: {weighted_truth_shares.mean():.4f} ± {weighted_truth_shares.std():.4f}")
print(f"Unweighted root truth share: {unweighted_truth_shares.mean():.4f} ± {unweighted_truth_shares.std():.4f}")

# ## Correlation Analysis

# Pearson correlation
r_weighted, p_weighted = stats.pearsonr(weighted_truth_shares, conclusions)
r_unweighted, p_unweighted = stats.pearsonr(unweighted_truth_shares, conclusions)

print(f"\nCorrelation Analysis:")
print(f"  Weighted truth share vs conclusion: r={r_weighted:.4f}, p={p_weighted:.2e}")
print(f"  Unweighted truth share vs conclusion: r={r_unweighted:.4f}, p={p_unweighted:.2e}")

# ## Visualization

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Weighted truth share vs network conclusion
ax = axes[0]
ax.scatter(weighted_truth_shares, conclusions, alpha=0.6, edgecolor='black', linewidth=0.5)
# Add regression line
z = np.polyfit(weighted_truth_shares, conclusions, 1)
p = np.poly1d(z)
x_line = np.linspace(weighted_truth_shares.min(), weighted_truth_shares.max(), 100)
ax.plot(x_line, p(x_line), 'r--', label=f'r = {r_weighted:.3f}')
ax.set_xlabel('Weighted Root Truth Share', fontsize=11)
ax.set_ylabel('Network Conclusion (Share Correct)', fontsize=11)
ax.set_title('Root Influence (Weighted by Descendants)', fontsize=12)
ax.legend()

# 2. Unweighted truth share vs network conclusion
ax = axes[1]
ax.scatter(unweighted_truth_shares, conclusions, alpha=0.6, edgecolor='black', linewidth=0.5)
z = np.polyfit(unweighted_truth_shares, conclusions, 1)
p = np.poly1d(z)
x_line = np.linspace(unweighted_truth_shares.min(), unweighted_truth_shares.max(), 100)
ax.plot(x_line, p(x_line), 'r--', label=f'r = {r_unweighted:.3f}')
ax.set_xlabel('Unweighted Root Truth Share', fontsize=11)
ax.set_ylabel('Network Conclusion (Share Correct)', fontsize=11)
ax.set_title('Root Influence (Equal Weights)', fontsize=12)
ax.legend()

# 3. Distribution of descendant counts (from first simulation)
ax = axes[2]
ra = results[0]["root_analysis"]
desc_counts = ra["descendant_counts"]
ax.bar(range(len(desc_counts)), np.sort(desc_counts)[::-1], edgecolor='black', alpha=0.7)
ax.set_xlabel('Root Node (sorted by influence)', fontsize=11)
ax.set_ylabel('Number of Descendants', fontsize=11)
ax.set_title(f'Root Node Influence Distribution (n={len(desc_counts)})', fontsize=12)
ax.axhline(np.mean(desc_counts), color='red', linestyle='--', label=f'Mean: {np.mean(desc_counts):.1f}')
ax.legend()

plt.tight_layout()
plt.savefig('root_influence_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to 'root_influence_analysis.png'")

# ## Detailed Root Analysis

print(f"\n{'='*60}")
print("Root Node Details")
print(f"{'='*60}")

# Analyze descendant distribution
print(f"\nDescendant count distribution:")
print(f"  Min: {desc_counts.min()}")
print(f"  Max: {desc_counts.max()}")
print(f"  Mean: {desc_counts.mean():.1f}")
print(f"  Median: {np.median(desc_counts):.1f}")
print(f"  Total agents reached by roots: {desc_counts.sum()} (may overlap)")

# Gini coefficient of influence
def gini(x):
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x)) - (n + 1) * np.sum(x)) / (n * np.sum(x))

gini_coef = gini(desc_counts)
print(f"\nInfluence inequality (Gini coefficient): {gini_coef:.4f}")

# Top influential roots
print(f"\nTop 5 most influential roots:")
sorted_idx = np.argsort(desc_counts)[::-1]
for i in range(min(5, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"  Root {idx}: {desc_counts[idx]} descendants")

print(f"\n{'='*60}")
print("Analysis Complete")
print(f"{'='*60}")
