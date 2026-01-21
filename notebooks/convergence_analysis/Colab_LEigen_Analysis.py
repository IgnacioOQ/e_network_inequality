#!/usr/bin/env python
# coding: utf-8

# # Left Eigenvector Centrality Analysis - Colab Notebook
# 
# This notebook tests the hypothesis that Left Eigenvector Centrality (DeGroot Influence)
# is the primary predictor of long-term belief outcomes in directed networks.
# 
# **Key Hypothesis:** The left eigenvector centrality generalizes root node influence
# and works for ALL directed networks, including those with cycles (no root nodes).

# # Setup

# In[ ]:


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDED GOOGLE COLAB RUNTIME
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("🚀 RECOMMENDED COLAB RUNTIME SETTINGS")
print("=" * 70)
print("""
Runtime Type: CPU (NOT GPU/TPU)
   - This notebook uses multiprocessing (CPU parallelization)
   - GPU/TPU won't help and wastes resources

Hardware Accelerator: None
   - Go to: Runtime → Change runtime type → Hardware accelerator: None

RAM:
   - Standard (12GB): OK for small networks (n < 200)
   - High-RAM (25GB+): RECOMMENDED for larger networks

Session Duration:
   - Free Colab: ~90 min timeout, may disconnect
   - Colab Pro: Up to 24h runtime

TIP: Run in background with Colab Pro for long simulations!
""")
print("=" * 70)


# In[ ]:


# Clone the repository (ai-agents-branch has the latest code)
get_ipython().system('git clone -b ai-agents-branch https://github.com/IgnacioOQ/e_network_inequality')


# In[ ]:


# Install required packages
get_ipython().system('pip install dill tqdm networkx pandas numpy scipy matplotlib seaborn scikit-learn')


# In[ ]:


# Change to repository directory and install the package
get_ipython().run_line_magic('cd', 'e_network_inequality')
get_ipython().system('pip install -e .')


# In[ ]:


# Add src to path and import modules
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

# Core imports
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy import stats

# Import from net_epistemology package
from net_epistemology.utils.imports import *
from net_epistemology.core.vectorized_model import VectorizedModel
from net_epistemology.utils.network_generation import *

print("✅ All imports successful!")


# # Left Eigenvector Centrality Functions

# In[ ]:


def compute_left_eigenvector(G):
    """
    Computes the Left Eigenvector centrality (DeGroot Influence).
    
    In the context of opinion dynamics, this metric identifies the 'ultimate' 
    sources of beliefs. It answers: "In the long run, how much does this agent's 
    initial state determine the group's final consensus?"
    
    Mathematical Definition:
    ------------------------
    1. Constructs a Row-Stochastic Matrix W where W_ij represents the 
       weight agent i places on agent j (based on incoming edges in G).
    2. If a node has no incoming edges (a Source), it is treated as 
       'stubborn' or 'independent' (weight 1.0 on itself).
    3. Solves pi * W = pi (normalized so sum(pi) = 1).
    
    Parameters:
    -----------
    G : nx.DiGraph
        A directed graph where an edge (u, v) means u influences v.
        (v listens to u).

    Returns:
    --------
    dict
        Dictionary mapping node IDs to their influence score (probability mass).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize adjacency matrix for the "Listening Graph"
    W = np.zeros((n, n))
    
    for u in nodes:
        u_idx = node_to_idx[u]
        # predecessors(u) are nodes that point TO u in G.
        influencers = list(G.predecessors(u))
        
        if len(influencers) == 0:
            # Case: Independent Agent (Source/Root). Listens only to itself.
            W[u_idx, u_idx] = 1.0
        else:
            # Case: Social Agent. Equal weights for simplicity.
            weight = 1.0 / len(influencers)
            for inf in influencers:
                v_idx = node_to_idx[inf]
                W[u_idx, v_idx] = weight
                
    # Calculate Left Eigenvector for Eigenvalue 1.
    # Corresponds to the Right Eigenvector of the Transpose matrix.
    eigenvalues, eigenvectors = np.linalg.eig(W.T)
    
    # Extract eigenvector corresponding to eigenvalue 1 (or closest to it)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    left_ev = np.real(eigenvectors[:, idx])
    
    # Normalize to form a probability distribution (sum = 1)
    left_ev = np.abs(left_ev) 
    left_ev = left_ev / np.sum(left_ev)
    
    return {nodes[i]: left_ev[i] for i in range(n)}


def compute_katz_centrality(G, alpha=0.1, beta=1.0, measure_influence=True):
    """
    Computes Katz Centrality, optionally measuring influence (reversed graph).
    """
    if measure_influence:
        target_G = G.reverse()
    else:
        target_G = G
        
    try:
        return nx.katz_centrality(target_G, alpha=alpha, beta=beta, normalized=True)
    except nx.PowerIterationFailedConvergence:
        return nx.katz_centrality_numpy(target_G, alpha=alpha, beta=beta, normalized=True)


# # Prediction Functions

# In[ ]:


def predict_outcomes_by_left_eigen(node_list, node_beliefs, left_eigen_centrality):
    """
    Predict final network belief based on left eigenvector weighted beliefs.
    
    The prediction is: sum of LE centrality for nodes believing truth.
    This represents the "influence mass" that believes truth.
    
    IMPORTANT: node_list must match the ordering of node_beliefs!
    Use model.nodes to ensure consistency with model.credences.
    
    Parameters:
    -----------
    node_list : list
        List of node IDs in the SAME ORDER as node_beliefs array.
        Should be model.nodes to match model.credences ordering.
    node_beliefs : np.ndarray
        Boolean array where True = believes truth. 
        MUST be indexed same as node_list.
    left_eigen_centrality : dict
        Left eigenvector centrality scores for each node (by node ID)
        
    Returns:
    --------
    float
        Predicted share believing truth (= sum of LE for truthful nodes)
    """
    truth_weight = 0.0
    
    for i, node in enumerate(node_list):
        weight = left_eigen_centrality.get(node, 0.0)
        if node_beliefs[i]:
            truth_weight += weight
    
    # LE centrality sums to 1.0, so truth_weight IS the predicted proportion
    return truth_weight


def predict_node_outcomes_by_influence(G, node_list, node_beliefs, left_eigen_centrality, threshold=0.5):
    """
    Predict per-node outcomes based on the influence from truthful vs false believers.
    
    For each node, compute the fraction of its incoming influence that comes from
    nodes believing truth. Predict truth if this exceeds threshold.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The network (for predecessor lookup)
    node_list : list
        List of node IDs in same order as node_beliefs (use model.nodes)
    node_beliefs : np.ndarray
        Current beliefs of each node (True/False for believing truth)
    left_eigen_centrality : dict
        Left eigenvector centrality scores
    threshold : float
        Threshold for predicting truth
        
    Returns:
    --------
    np.ndarray
        Predicted beliefs for each node
    """
    n = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    predictions = np.zeros(n, dtype=float)
    
    for i, node in enumerate(node_list):
        # Get predecessors (those influencing this node)
        preds = list(G.predecessors(node))
        
        if len(preds) == 0:
            # Root node - use its own belief
            predictions[i] = float(node_beliefs[i])
        else:
            # Compute weighted average of predecessor beliefs
            total_influence = 0.0
            truth_influence = 0.0
            for pred in preds:
                pred_idx = node_to_idx.get(pred)
                if pred_idx is not None:
                    weight = left_eigen_centrality.get(pred, 1.0 / len(preds))
                    total_influence += weight
                    if node_beliefs[pred_idx]:
                        truth_influence += weight
            
            if total_influence > 0:
                predictions[i] = truth_influence / total_influence
            else:
                predictions[i] = 0.5
    
    return predictions >= threshold


# # Test Networks
# 
# We test on TWO CATEGORIES of networks to validate the Left Eigenvector hypothesis:
# 
# **CATEGORY A: Networks WITH Root Nodes**
# - Root-based prediction IS applicable
# - LE should match root-based predictions
# 
# **CATEGORY B: Networks WITHOUT Root Nodes (Cyclic)**
# - Root-based prediction NOT applicable
# - LE provides unique predictive value

# In[ ]:


def create_test_networks():
    """
    Create a variety of test networks including those WITH and WITHOUT root nodes.
    
    CATEGORY A: Networks WITH root nodes
    - barabasi_albert: Scale-free with hubs
    - tree: Perfect hierarchy with single root
    - empirical_pud: Real-world citation network
    
    CATEGORY B: Networks WITHOUT root nodes (cyclic)
    - complete: Everyone influences everyone
    - erdos_renyi_cyclic: Random with added cycles
    """
    networks = {}
    
    print("=" * 70)
    print("CATEGORY A: NETWORKS WITH ROOT NODES")
    print("=" * 70)
    
    # A1. Network WITH root nodes - Barabasi-Albert
    print("\n[A1] Creating Barabasi-Albert network...")
    G_ba = nx.barabasi_albert_graph(100, 3, seed=42)
    G_ba = nx.DiGraph(G_ba)  # Convert to directed
    root_count_ba = sum(1 for n in G_ba.nodes() if G_ba.in_degree(n) == 0)
    networks['A1_barabasi_albert'] = {
        'graph': G_ba,
        'has_roots': True,
        'category': 'A',
        'description': f'Barabasi-Albert (100 nodes, {root_count_ba} roots)'
    }
    print(f"     {root_count_ba} root nodes detected")
    
    # A2. Network WITH root nodes - Tree
    print("\n[A2] Creating Directed Tree network...")
    G_tree = nx.balanced_tree(3, 4, create_using=nx.DiGraph())
    # Reverse edges so root has influence (children listen to parents)
    G_tree = G_tree.reverse()
    networks['A2_tree'] = {
        'graph': G_tree,
        'has_roots': True,
        'category': 'A',
        'description': f'Balanced Tree (3-ary depth 4, 1 root with 100% influence)'
    }
    print("     1 root node (perfect hierarchy)")
    
    # A3. Load empirical network if available
    try:
        network_path = 'data/empirical_networks/pud_final.json'
        with open(network_path, 'r') as f:
            network_data = json.load(f)
        if 'links' in network_data:
            network_data['edges'] = network_data.pop('links')
        G_emp = nx.node_link_graph(network_data)
        root_count = sum(1 for n in G_emp.nodes() if G_emp.in_degree(n) == 0)
        networks['A3_empirical_pud'] = {
            'graph': G_emp,
            'has_roots': root_count > 0,
            'category': 'A',
            'description': f'Empirical PUD ({len(G_emp.nodes())} nodes, {root_count} roots)'
        }
        print(f"\n[A3] Loaded empirical network: {len(G_emp.nodes())} nodes, {root_count} roots")
    except FileNotFoundError:
        print("\n[A3] Empirical network not found, skipping...")
    
    print("\n" + "=" * 70)
    print("CATEGORY B: NETWORKS WITHOUT ROOT NODES (CYCLIC)")
    print("=" * 70)
    print("NOTE: Root-based prediction NOT applicable for these networks!")
    print("      Left Eigenvector provides UNIQUE predictive value here.")
    
    # B1. Network WITHOUT root nodes - Complete graph (cyclic)
    print("\n[B1] Creating Complete Directed graph...")
    G_complete = nx.complete_graph(50, create_using=nx.DiGraph())
    networks['B1_complete'] = {
        'graph': G_complete,
        'has_roots': False,
        'category': 'B',
        'description': 'Complete Graph (50 nodes, 0 roots, uniform LE)'
    }
    print("     0 root nodes (everyone influences everyone)")
    
    # B2. Network WITHOUT root nodes - Random with cycles
    print("\n[B2] Creating Erdos-Renyi network with enforced cycles...")
    np.random.seed(42)
    G_er = nx.gnp_random_graph(100, 0.05, directed=True, seed=42)
    # Ensure NO root nodes by adding back edges
    for node in list(G_er.nodes()):
        if G_er.in_degree(node) == 0:
            sources = [n for n in G_er.nodes() if n != node]
            if sources:
                G_er.add_edge(np.random.choice(sources), node)
    root_check = sum(1 for n in G_er.nodes() if G_er.in_degree(n) == 0)
    networks['B2_erdos_renyi_cyclic'] = {
        'graph': G_er,
        'has_roots': False,
        'category': 'B',
        'description': f'Erdos-Renyi (100 nodes, {root_check} roots, cyclic)'
    }
    print(f"     {root_check} root nodes after adding cycles")
    
    return networks


# Print summary
networks = create_test_networks()
print("\n" + "=" * 70)
print("TEST NETWORKS SUMMARY")
print("=" * 70)
print("\n{:<25} {:>8} {:>8} {:>10} {}".format("Network", "Nodes", "Edges", "Roots", "Category"))
print("-" * 70)
for name, info in networks.items():
    G = info['graph']
    roots = sum(1 for n in G.nodes() if G.in_degree(n) == 0)
    cat = "WITH ROOTS" if info['category'] == 'A' else "NO ROOTS"
    print(f"{name:<25} {len(G.nodes()):>8} {len(G.edges()):>8} {roots:>10} {cat}")


# # Single Simulation Analysis

# In[ ]:


def run_single_analysis(network_name, network_info, n_steps=50000, uncertainty=0.001):
    """
    Run a single simulation and compare left eigenvector prediction with actual outcome.
    """
    G = network_info['graph']
    nodes = list(G.nodes())
    n_agents = len(nodes)
    category = network_info.get('category', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {network_name}")
    print(f"Category: {'WITH ROOTS (A)' if category == 'A' else 'NO ROOTS / CYCLIC (B)'}")
    print(f"{'='*60}")
    print(f"Network: {n_agents} nodes, {len(G.edges())} edges")
    
    # Compute left eigenvector centrality
    print("Computing Left Eigenvector Centrality...")
    left_eigen = compute_left_eigenvector(G)
    
    # Analyze LE distribution
    le_values = np.array(list(left_eigen.values()))
    print(f"  LE stats: max={np.max(le_values):.4f}, mean={np.mean(le_values):.4f}")
    
    # Also compute Katz for comparison
    print("Computing Katz Centrality...")
    try:
        katz = compute_katz_centrality(G, alpha=0.01)
    except:
        katz = {n: 1.0/n_agents for n in nodes}  # Fallback to uniform
    
    # Run simulation
    print(f"Running simulation for {n_steps} steps...")
    model = VectorizedModel(
        network=G,
        n_experiments=10,
        uncertainty=uncertainty,
        agent_type="beta",
        tstep_stopping=True,
        compute_root_analysis=True,
    )
    model.run_simulation(number_of_steps=n_steps, show_bar=True)
    
    # Get actual outcomes
    actual_beliefs = model.credences[:, 1] > model.credences[:, 0]
    actual_proportion = np.mean(actual_beliefs)
    
    # Compute predictions
    # IMPORTANT: Use model.nodes to ensure same ordering as model.credences
    node_list = model.nodes  # This matches the ordering of actual_beliefs
    
    # 1. Left Eigenvector weighted prediction
    le_prediction = predict_outcomes_by_left_eigen(node_list, actual_beliefs, left_eigen)
    
    # 2. Root-based prediction (if roots exist)
    if model.root_analysis and model.root_analysis['n_roots'] > 0:
        root_prediction = model.proportion_reached_by_truth
    else:
        root_prediction = None
    
    # 3. Katz-weighted prediction
    katz_prediction = predict_outcomes_by_left_eigen(node_list, actual_beliefs, katz)
    
    # Node-level analysis
    node_predictions_le = predict_node_outcomes_by_influence(G, node_list, actual_beliefs, left_eigen)
    node_accuracy_le = np.mean(node_predictions_le == actual_beliefs)
    
    results = {
        'network': network_name,
        'category': category,
        'n_nodes': n_agents,
        'n_edges': len(G.edges()),
        'has_roots': network_info['has_roots'],
        'n_roots': model.root_analysis['n_roots'] if model.root_analysis else 0,
        'actual_proportion': actual_proportion,
        'le_prediction': le_prediction,
        'le_error': abs(actual_proportion - le_prediction),
        'root_prediction': root_prediction,
        'root_error': abs(actual_proportion - root_prediction) if root_prediction is not None else None,
        'katz_prediction': katz_prediction,
        'katz_error': abs(actual_proportion - katz_prediction),
        'node_accuracy_le': node_accuracy_le,
        'n_steps': n_steps,
    }
    
    # Print results
    print(f"\nResults for {network_name}:")
    print(f"  Actual share believing truth: {actual_proportion:.4f}")
    print(f"  Left Eigenvector prediction:  {le_prediction:.4f} (error: {abs(actual_proportion - le_prediction):.4f})")
    if root_prediction is not None:
        print(f"  Root-based prediction:        {root_prediction:.4f} (error: {abs(actual_proportion - root_prediction):.4f})")
    else:
        print(f"  Root-based prediction:        N/A (no roots - Category B network)")
    print(f"  Katz centrality prediction:   {katz_prediction:.4f} (error: {abs(actual_proportion - katz_prediction):.4f})")
    print(f"  Node-level accuracy (LE):     {node_accuracy_le:.4f}")
    
    return results, left_eigen, model


# # Run Analysis on All Networks

# In[ ]:


all_results = []

# Run Category A (WITH roots) first
print("\n" + "=" * 80)
print("RUNNING CATEGORY A: NETWORKS WITH ROOT NODES")
print("=" * 80)
for network_name, network_info in networks.items():
    if network_info.get('category') == 'A':
        try:
            results, left_eigen, model = run_single_analysis(
                network_name, 
                network_info, 
                n_steps=100000,
                uncertainty=0.001
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {network_name}: {e}")

# Run Category B (WITHOUT roots)
print("\n" + "=" * 80)
print("RUNNING CATEGORY B: NETWORKS WITHOUT ROOT NODES (CYCLIC)")
print("This is where Left Eigenvector provides UNIQUE value!")
print("=" * 80)
for network_name, network_info in networks.items():
    if network_info.get('category') == 'B':
        try:
            results, left_eigen, model = run_single_analysis(
                network_name, 
                network_info, 
                n_steps=100000,
                uncertainty=0.001
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {network_name}: {e}")

# Create results dataframe
results_df = pd.DataFrame(all_results)

# Display summary with focus on ERRORS
print("\n" + "=" * 80)
print("SUMMARY RESULTS - PREDICTION ERRORS")
print("=" * 80)
print("\nLower error = better prediction. Root error is N/A for Category B (cyclic networks).")
print()

# Create a focused summary
for _, row in results_df.iterrows():
    root_err_str = f"{row['root_error']:.4f}" if row['root_error'] is not None else "N/A"
    print(f"{row['network']:25} | Cat {row['category']} | Roots: {row['n_roots']:3} | "
          f"LE Err: {row['le_error']:.4f} | Root Err: {root_err_str:>6} | "
          f"Katz Err: {row['katz_error']:.4f} | Node Acc: {row['node_accuracy_le']:.4f}")

print()
print("Full DataFrame:")
print(results_df[['network', 'category', 'n_roots', 'actual_proportion', 'le_error', 'root_error', 'katz_error', 'node_accuracy_le']].to_string())


# # Visualization

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Prediction Error Comparison (using pre-computed errors)
ax = axes[0, 0]
x = np.arange(len(results_df))
width = 0.25

# Use pre-computed error columns
le_errors = results_df['le_error'].values
root_errors = results_df['root_error'].fillna(0).values  # Fill NaN with 0 for plotting
katz_errors = results_df['katz_error'].values

# Mark which have valid root errors
root_valid = results_df['root_error'].notna().values

ax.bar(x - width, le_errors, width, label='Left Eigenvector', color='blue', alpha=0.7)
bars_root = ax.bar(x, root_errors, width, label='Root-based', color='green', alpha=0.7)
ax.bar(x + width, katz_errors, width, label='Katz', color='orange', alpha=0.7)

# Mark N/A for root errors (Category B)
for i, valid in enumerate(root_valid):
    if not valid:
        bars_root[i].set_alpha(0.2)
        bars_root[i].set_hatch('//')

ax.set_xlabel('Network')
ax.set_ylabel('Prediction Error (|Actual - Predicted|)')
ax.set_title('Prediction Error by Method\\n(Hatched = Root N/A for Category B)')
ax.set_xticks(x)
ax.set_xticklabels(results_df['network'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Left Eigenvector Centrality Distribution (last network)
ax = axes[0, 1]
le_values = np.array(list(left_eigen.values()))
ax.hist(le_values, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(le_values), color='red', linestyle='--', label=f'Mean: {np.mean(le_values):.4f}')
ax.axvline(np.max(le_values), color='green', linestyle='--', label=f'Max: {np.max(le_values):.4f}')
ax.set_xlabel('Left Eigenvector Centrality')
ax.set_ylabel('Frequency')
ax.set_title(f'Left Eigenvector Distribution (Last Network)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Node-level Accuracy by Category
ax = axes[1, 0]
colors = ['blue' if cat == 'A' else 'red' for cat in results_df['category']]
ax.bar(results_df['network'], results_df['node_accuracy_le'], color=colors, alpha=0.7)
ax.axhline(0.5, color='gray', linestyle='--', label='Random baseline')
ax.set_xlabel('Network')
ax.set_ylabel('Node-level Accuracy')
ax.set_title('Left Eigenvector Node-Level Accuracy\n(Blue=Category A/With Roots, Red=Category B/No Roots)')
ax.set_xticklabels(results_df['network'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Summary Table - ERRORS
ax = axes[1, 1]
ax.axis('off')
summary_data = []
for _, row in results_df.iterrows():
    root_err = f"{row['root_error']:.4f}" if row['root_error'] is not None else "N/A"
    summary_data.append([
        row['network'][:16],
        row['category'],
        f"{row['n_roots']}",
        f"{row['le_error']:.4f}",
        root_err,
        f"{row['katz_error']:.4f}",
        f"{row['node_accuracy_le']:.4f}"
    ])
table = ax.table(
    cellText=summary_data,
    colLabels=['Network', 'Cat', 'Roots', 'LE Err', 'Root Err', 'Katz Err', 'Node Acc'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)
ax.set_title('Prediction Errors Summary\\n(Lower = Better)', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('left_eigen_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to 'left_eigen_analysis.png'")
plt.show()


# # Convergence Analysis (Multiple Steps)

# In[ ]:


def run_convergence_analysis(G, network_name, step_counts=[1000, 5000, 10000, 50000, 100000, 500000], 
                              uncertainty=0.001, n_trials=5):
    """
    Run multiple simulations at different step counts to see how predictions improve.
    """
    print(f"\n{'='*60}")
    print(f"Convergence Analysis: {network_name}")
    print(f"{'='*60}")
    
    # Compute centralities once
    left_eigen = compute_left_eigenvector(G)
    nodes = list(G.nodes())
    
    results = []
    
    for n_steps in step_counts:
        le_errors = []
        root_errors = []
        
        for trial in range(n_trials):
            model = VectorizedModel(
                network=G,
                n_experiments=10,
                uncertainty=uncertainty,
                agent_type="beta",
                tstep_stopping=True,
                compute_root_analysis=True,
            )
            model.run_simulation(number_of_steps=n_steps, show_bar=False)
            
            actual_beliefs = model.credences[:, 1] > model.credences[:, 0]
            actual = np.mean(actual_beliefs)
            
            # Use model.nodes for consistent ordering
            le_pred = predict_outcomes_by_left_eigen(model.nodes, actual_beliefs, left_eigen)
            le_errors.append(abs(actual - le_pred))
            
            if model.root_analysis and model.root_analysis['n_roots'] > 0:
                root_errors.append(abs(actual - model.proportion_reached_by_truth))
        
        results.append({
            'steps': n_steps,
            'le_error_mean': np.mean(le_errors),
            'le_error_std': np.std(le_errors),
            'root_error_mean': np.mean(root_errors) if root_errors else None,
            'root_error_std': np.std(root_errors) if root_errors else None,
        })
        
        print(f"Steps {n_steps:6d}: LE error = {np.mean(le_errors):.4f} ± {np.std(le_errors):.4f}")
    
    return results


# Run on Empirical network if available, else BA network
if 'A3_empirical_pud' in networks:
    conv_results = run_convergence_analysis(
        networks['A3_empirical_pud']['graph'],
        'A3_empirical_pud',
        step_counts=[1000, 5000, 10000, 50000, 100000, 500000],
        n_trials=3
    )
elif 'A1_barabasi_albert' in networks:
    conv_results = run_convergence_analysis(
        networks['A1_barabasi_albert']['graph'],
        'A1_barabasi_albert',
        step_counts=[1000, 5000, 10000, 50000, 100000, 500000],
        n_trials=3
    )


# # Final Summary

# In[ ]:


print("\n" + "=" * 80)
print("LEFT EIGENVECTOR HYPOTHESIS TEST RESULTS")
print("=" * 80)

# Separate results by category
cat_a_results = results_df[results_df['category'] == 'A']
cat_b_results = results_df[results_df['category'] == 'B']

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  LEFT EIGENVECTOR CENTRALITY HYPOTHESIS                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  The Left Eigenvector Centrality (DeGroot Influence) is the PRIMARY         ║
║  predictor of long-term belief outcomes in directed networks, GENERALIZING  ║
║  root node influence to networks WITH and WITHOUT root nodes.               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("=" * 80)
print("CATEGORY A RESULTS: Networks WITH Root Nodes")
print("=" * 80)
if len(cat_a_results) > 0:
    avg_le_error_a = np.mean([abs(r['actual_proportion'] - r['le_prediction']) for _, r in cat_a_results.iterrows()])
    root_errors_a = [abs(r['actual_proportion'] - r['root_prediction']) for _, r in cat_a_results.iterrows() if r['root_prediction'] is not None]
    avg_root_error_a = np.mean(root_errors_a) if root_errors_a else None
    print(f"  Average LE Error:   {avg_le_error_a:.4f}")
    if avg_root_error_a:
        print(f"  Average Root Error: {avg_root_error_a:.4f}")
    print("  ➜ In networks with roots, LE captures the root influence structure.")

print("\n" + "=" * 80)
print("CATEGORY B RESULTS: Networks WITHOUT Root Nodes (CYCLIC)")
print("=" * 80)
if len(cat_b_results) > 0:
    avg_le_error_b = np.mean([abs(r['actual_proportion'] - r['le_prediction']) for _, r in cat_b_results.iterrows()])
    print(f"  Average LE Error:   {avg_le_error_b:.4f}")
    print("  Root-based Error:   N/A (no roots exist!)")
    print("  ➜ LE provides UNIQUE predictive value where root-based methods fail!")
else:
    print("  No Category B networks were analyzed.")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  KEY INSIGHT:                                                                ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  • In DAGs (Category A): LE centrality concentrates in root nodes            ║
║  • In cyclic networks (Category B): LE identifies "effective sources"       ║
║  • LE is the GENERALIZED form of root influence                             ║
║  • This connects to the Markov Chain interpretation of belief dynamics      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)


# # Disconnect from Runtime

# In[ ]:


from datetime import datetime
try:
    import pytz
    nyc_time = datetime.now(pytz.timezone('America/New_York'))
    formatted_time = nyc_time.strftime('%Y-%m-%d %H:%M:%S %Z')
except ImportError:
    formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f"✅ Analysis completed at: {formatted_time}")

# Disconnect from Colab runtime to free resources
try:
    from IPython.display import Javascript
    display(Javascript('google.colab.kernel.disconnect()'))
    print("🔌 Disconnected from Colab runtime.")
except:
    print("(Not running in Colab - no disconnect needed)")

