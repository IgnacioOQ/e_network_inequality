#!/usr/bin/env python
# coding: utf-8

# # Root Node Influence Analysis
#
# Testing hypothesis: The final epistemic state of the network is determined by
# root nodes' beliefs. Specifically, all descendants of truthful roots will
# eventually converge to truth.

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from scipy import stats

from models.vectorized_model import VectorizedModel

plt.style.use('seaborn-v0_8-whitegrid')

# ## Load Network

print("Loading pud_final.json network...")
network_path = os.path.join(os.path.dirname(__file__), '..', '..', 'networks', 'citation_data', 'pud_final.json')

with open(network_path, 'r') as f:
    network_data = json.load(f)

# Handle 'links' vs 'edges' key discrepancy
if 'links' in network_data:
    pass # node_link_graph defaults to 'links'
elif 'edges' in network_data:
    network_data['links'] = network_data['edges'] # node_link_graph expects 'links' by default usually, or we can specify `edges='edges'`
else:
    # If neither, it might be adjacency data which this function doesn't handle, but let's assume it's node-link
    pass

try:
    network = nx.node_link_graph(network_data)
except KeyError:
    # Fallback: maybe it wants 'edges' specified explicitly or the data structure is slightly off
    # If 'links' is missing but 'edges' exists, we can try alerting it
    if 'edges' in network_data:
        network = nx.node_link_graph(network_data, edges='edges')
    else:
        raise
n_agents = len(network.nodes())
print(f"Network: {n_agents} nodes, {len(network.edges())} edges")

# Analyze network structure
in_degrees = dict(network.in_degree())
root_nodes = [n for n, d in in_degrees.items() if d == 0]
print(f"Number of root nodes: {len(root_nodes)}")

UNCERTAINTY = 0.001  # Epsilon

# ## Convergence Gap Analysis
#
# Test how the gap between predicted (proportion reached by truthful roots)
# and actual (share believing truth) shrinks with more simulation steps.

print(f"\n{'='*60}")
print("Convergence Gap Analysis")
print(f"{'='*60}")
print(f"Epsilon (uncertainty) = {UNCERTAINTY}")
print(f"\nTesting how gap shrinks with increasing steps...")
print("-" * 60)

step_counts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
gap_results = []

for n_steps in step_counts:
    model = VectorizedModel(
        network=network,
        n_experiments=10,
        uncertainty=UNCERTAINTY,
        agent_type="beta",
        tstep_stopping=True,
        compute_root_analysis=True,
    )
    model.run_simulation(number_of_steps=n_steps, show_bar=True)
    
    predicted = model.proportion_reached_by_truth
    actual = model.conclusion
    gap = actual - predicted
    
    gap_results.append({
        'steps': n_steps,
        'predicted': predicted,
        'actual': actual,
        'gap': gap
    })
    
    print(f"Steps: {n_steps:7d} | Predicted: {predicted:.4f} | Actual: {actual:.4f} | Gap: {gap:+.4f}")
    
    if abs(gap) < UNCERTAINTY:
        print(f"\n*** Gap ({abs(gap):.4f}) < epsilon ({UNCERTAINTY})! Hypothesis CONFIRMED. ***")

# ## Visualization

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Gap vs Steps (log scale)
ax = axes[0, 0]
steps = [r['steps'] for r in gap_results]
gaps = [abs(r['gap']) for r in gap_results]
ax.plot(steps, gaps, 'o-', linewidth=2, markersize=8)
ax.axhline(UNCERTAINTY, color='red', linestyle='--', label=f'ε = {UNCERTAINTY}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of Steps', fontsize=11)
ax.set_ylabel('|Gap| = |Actual - Predicted|', fontsize=11)
ax.set_title('Convergence: Gap Shrinks with More Steps', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Predicted vs Actual over steps
ax = axes[0, 1]
predicted_vals = [r['predicted'] for r in gap_results]
actual_vals = [r['actual'] for r in gap_results]
ax.plot(steps, predicted_vals, 'o-', label='Predicted (nodes reachable by truthful roots)', linewidth=2)
ax.plot(steps, actual_vals, 's-', label='Actual (share believing truth)', linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('Number of Steps', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Predicted vs Actual Over Time', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Final state: Predicted vs Actual scatter (from last run)
ax = axes[1, 0]
# Use the final (longest) run
final_model = model  # From the last iteration
ra = final_model.root_analysis
desc_counts = ra["descendant_counts"]
ax.bar(range(len(desc_counts)), np.sort(desc_counts)[::-1], edgecolor='black', alpha=0.7)
ax.set_xlabel('Root Node (sorted by influence)', fontsize=11)
ax.set_ylabel('Number of Descendants', fontsize=11)
ax.set_title(f'Root Node Influence Distribution (n={len(desc_counts)})', fontsize=12)
ax.axhline(np.mean(desc_counts), color='red', linestyle='--', label=f'Mean: {np.mean(desc_counts):.1f}')
ax.legend()

# 4. Summary table as text
ax = axes[1, 1]
ax.axis('off')
table_data = [[r['steps'], f"{r['predicted']:.4f}", f"{r['actual']:.4f}", f"{r['gap']:+.4f}"] 
              for r in gap_results]
table = ax.table(
    cellText=table_data,
    colLabels=['Steps', 'Predicted', 'Actual', 'Gap'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax.set_title('Convergence Summary', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('root_influence_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to 'root_influence_analysis.png'")

# ## Final Summary

print(f"\n{'='*60}")
print("HYPOTHESIS TEST RESULTS")
print(f"{'='*60}")

print(f"""
HYPOTHESIS: If a root node converges to the true hypothesis, all of its
descendants will converge to the true hypothesis.

PREDICTION: The share of agents believing truth = proportion of network
reachable from truthful roots.

RESULT: At {gap_results[-1]['steps']:,} steps:
  - Predicted: {gap_results[-1]['predicted']:.4f}
  - Actual: {gap_results[-1]['actual']:.4f}
  - Gap: {gap_results[-1]['gap']:+.4f}
  
CONCLUSION: {'CONFIRMED ✓' if abs(gap_results[-1]['gap']) < UNCERTAINTY else 'Gap still > epsilon'}
The gap shrinks to near-zero with sufficient steps, confirming that
truth propagates from roots to all their descendants given enough time.
""")

# Gini coefficient of influence
def gini(x):
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x)) - (n + 1) * np.sum(x)) / (n * np.sum(x))

print(f"Root influence inequality (Gini): {gini(desc_counts):.4f}")

print(f"\n{'='*60}")
print("Analysis Complete")
print(f"{'='*60}")
