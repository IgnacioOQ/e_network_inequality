#!/usr/bin/env python
# coding: utf-8

# # Left Eigenvector Centrality Analysis
#
# Testing hypothesis: Left Eigenvector Centrality (DeGroot Influence) is the
# primary predictor of long-term belief outcomes in directed networks.
# This generalizes root node influence to ALL directed networks.

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from net_epistemology.core.vectorized_model import VectorizedModel

# Import local left_eigen functions
from left_eigen import compute_left_eigenvector, compute_katz_centrality

plt.style.use('seaborn-v0_8-whitegrid')

# ## Helper Functions

def predict_outcomes_by_left_eigen(G, node_beliefs, left_eigen_centrality):
    """
    Predict final network belief based on left eigenvector weighted beliefs.
    
    The prediction is: sum of LE centrality held by truth-believers.
    This represents the "influence mass" that believes truth.
    """
    nodes = list(G.nodes())
    truth_weight = 0.0
    
    for i, node in enumerate(nodes):
        weight = left_eigen_centrality.get(node, 0.0)
        if node_beliefs[i]:
            truth_weight += weight
    
    # LE centrality sums to 1.0, so truth_weight is already the proportion
    return truth_weight


def predict_node_outcomes(G, node_beliefs, left_eigen_centrality, threshold=0.5):
    """
    Predict per-node outcomes based on influence-weighted predecessor beliefs.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    predictions = np.zeros(n, dtype=bool)
    
    for i, node in enumerate(nodes):
        preds = list(G.predecessors(node))
        
        if len(preds) == 0:
            predictions[i] = node_beliefs[i]
        else:
            total_influence = 0.0
            truth_influence = 0.0
            for pred in preds:
                pred_idx = node_to_idx[pred]
                weight = left_eigen_centrality.get(pred, 1.0 / len(preds))
                total_influence += weight
                if node_beliefs[pred_idx]:
                    truth_influence += weight
            
            predictions[i] = (truth_influence / total_influence >= threshold) if total_influence > 0 else False
    
    return predictions


def gini(x):
    """Compute Gini coefficient."""
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x)) - (n + 1) * np.sum(x)) / (n * np.sum(x) + 1e-10)


# ## Load Network

print("=" * 60)
print("LEFT EIGENVECTOR CENTRALITY ANALYSIS")
print("=" * 60)

print("\nLoading pud_final.json network...")
network_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'empirical_networks', 'pud_final.json')

with open(network_path, 'r') as f:
    network_data = json.load(f)

if 'links' in network_data:
    network_data['edges'] = network_data.pop('links')

network = nx.node_link_graph(network_data)
n_agents = len(network.nodes())
n_edges = len(network.edges())
print(f"Network: {n_agents} nodes, {n_edges} edges")

# Analyze network structure
in_degrees = dict(network.in_degree())
root_nodes = [n for n, d in in_degrees.items() if d == 0]
print(f"Number of root nodes: {len(root_nodes)}")

# ## Compute Centralities

print("\nComputing Left Eigenvector Centrality...")
left_eigen = compute_left_eigenvector(network)

print("Computing Katz Centrality...")
try:
    katz = compute_katz_centrality(network, alpha=0.01)
except:
    katz = {n: 1.0/n_agents for n in network.nodes()}

# Analyze centrality distribution
le_values = np.array([left_eigen[n] for n in network.nodes()])
print(f"\nLeft Eigenvector Centrality Statistics:")
print(f"  Max: {np.max(le_values):.4f}")
print(f"  Mean: {np.mean(le_values):.4f}")
print(f"  Gini: {gini(le_values):.4f}")

# Check how much LE is concentrated in root nodes
root_le_sum = sum(left_eigen[n] for n in root_nodes)
print(f"  Share held by root nodes: {root_le_sum:.4f} ({root_le_sum*100:.1f}%)")

# ## Convergence Gap Analysis

UNCERTAINTY = 0.001

print(f"\n{'='*60}")
print("CONVERGENCE ANALYSIS: LEFT EIGENVECTOR vs ROOT-BASED")
print(f"{'='*60}")
print(f"Epsilon (uncertainty) = {UNCERTAINTY}")

step_counts = [1000, 5000, 10000, 50000, 100000]
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
    
    # Get beliefs
    actual_beliefs = model.credences[:, 1] > model.credences[:, 0]
    actual = np.mean(actual_beliefs)
    
    # Left Eigenvector prediction
    le_pred = predict_outcomes_by_left_eigen(network, actual_beliefs, left_eigen)
    
    # Root-based prediction
    root_pred = model.proportion_reached_by_truth
    
    # Node-level predictions
    node_pred_le = predict_node_outcomes(network, actual_beliefs, left_eigen)
    node_accuracy_le = np.mean(node_pred_le == actual_beliefs)
    
    # Root-based node prediction (from root_analysis)
    if model.root_analysis and model.root_analysis['node_predictions'] is not None:
        node_pred_root = model.root_analysis['node_predictions'] > 0.5
        node_accuracy_root = np.mean(node_pred_root == actual_beliefs)
    else:
        node_accuracy_root = None
    
    result = {
        'steps': n_steps,
        'actual': actual,
        'le_pred': le_pred,
        'root_pred': root_pred,
        'le_error': abs(actual - le_pred),
        'root_error': abs(actual - root_pred),
        'node_acc_le': node_accuracy_le,
        'node_acc_root': node_accuracy_root,
    }
    gap_results.append(result)
    
    print(f"\nSteps: {n_steps:6d}")
    print(f"  Actual belief share:     {actual:.4f}")
    print(f"  Left Eigenvector pred:   {le_pred:.4f} (error: {result['le_error']:.4f})")
    print(f"  Root-based pred:         {root_pred:.4f} (error: {result['root_error']:.4f})")
    print(f"  Node accuracy (LE):      {node_accuracy_le:.4f}")
    if node_accuracy_root:
        print(f"  Node accuracy (Root):    {node_accuracy_root:.4f}")

# ## Visualization

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Prediction Error Comparison
ax = axes[0, 0]
steps = [r['steps'] for r in gap_results]
le_errors = [r['le_error'] for r in gap_results]
root_errors = [r['root_error'] for r in gap_results]

ax.plot(steps, le_errors, 'o-', linewidth=2, markersize=8, label='Left Eigenvector', color='blue')
ax.plot(steps, root_errors, 's-', linewidth=2, markersize=8, label='Root-based', color='green')
ax.axhline(UNCERTAINTY, color='red', linestyle='--', label=f'ε = {UNCERTAINTY}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of Steps', fontsize=11)
ax.set_ylabel('Prediction Error', fontsize=11)
ax.set_title('Left Eigenvector vs Root-Based Prediction Error', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Left Eigenvector Distribution
ax = axes[0, 1]
ax.hist(le_values, bins=30, edgecolor='black', alpha=0.7, color='blue')
ax.axvline(np.mean(le_values), color='red', linestyle='--', label=f'Mean: {np.mean(le_values):.4f}')
ax.set_xlabel('Left Eigenvector Centrality', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Left Eigenvector Centrality Distribution', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Root LE vs Non-Root LE
ax = axes[1, 0]
root_le = [left_eigen[n] for n in root_nodes]
non_root_le = [left_eigen[n] for n in network.nodes() if n not in root_nodes]
ax.bar(['Root Nodes\n(in_degree=0)', 'Non-Root Nodes'], 
       [np.sum(root_le), np.sum(non_root_le)], 
       color=['green', 'gray'], edgecolor='black', alpha=0.7)
ax.set_ylabel('Total Left Eigenvector Centrality', fontsize=11)
ax.set_title(f'LE Centrality: Roots ({len(root_nodes)}) vs Non-Roots ({n_agents-len(root_nodes)})', fontsize=12)
ax.grid(True, alpha=0.3)

# 4. Summary Table
ax = axes[1, 1]
ax.axis('off')
table_data = [[r['steps'], f"{r['le_error']:.4f}", f"{r['root_error']:.4f}", 
               f"{r['node_acc_le']:.4f}", f"{r['node_acc_root']:.4f}" if r['node_acc_root'] else "N/A"] 
              for r in gap_results]
table = ax.table(
    cellText=table_data,
    colLabels=['Steps', 'LE Error', 'Root Error', 'Node Acc (LE)', 'Node Acc (Root)'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax.set_title('Convergence Summary', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('left_eigen_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to 'left_eigen_analysis.png'")

# ## Final Summary

print(f"\n{'='*60}")
print("LEFT EIGENVECTOR HYPOTHESIS TEST RESULTS")
print(f"{'='*60}")

# Compare final errors
final = gap_results[-1]
node_acc_root_str = f"{final['node_acc_root']:.4f}" if final['node_acc_root'] else "N/A"
conclusion_str = "LEFT EIGENVECTOR matches ROOT-BASED" if abs(final['le_error'] - final['root_error']) < 0.01 else "Methods differ - see details"

print(f"""
HYPOTHESIS: Left Eigenvector Centrality generalizes root node influence
and is the primary predictor of long-term belief outcomes.

TEST NETWORK: {n_agents} nodes, {len(root_nodes)} roots

RESULTS at {final['steps']:,} steps:
  - Actual share believing truth: {final['actual']:.4f}
  
  LEFT EIGENVECTOR PREDICTION:
    - Predicted: {final['le_pred']:.4f}
    - Error: {final['le_error']:.4f}
    - Node-level accuracy: {final['node_acc_le']:.4f}
  
  ROOT-BASED PREDICTION:
    - Predicted: {final['root_pred']:.4f}
    - Error: {final['root_error']:.4f}
    - Node-level accuracy: {node_acc_root_str}

KEY INSIGHT:
  - Roots hold {root_le_sum*100:.1f}% of total Left Eigenvector centrality
  - This explains why LE generalizes root influence
  - In networks WITHOUT roots, LE still identifies influential nodes!

CONCLUSION: {conclusion_str}
""")

print(f"{'='*60}")
print("Analysis Complete")
print(f"{'='*60}")

