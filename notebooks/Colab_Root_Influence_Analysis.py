#!/usr/bin/env python
# coding: utf-8

# # Root Influence Analysis - Colab Notebook
# 
# This notebook runs parallel simulations with root influence analysis enabled.
# It tests the hypothesis that root nodes' beliefs determine the final epistemic state.

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
   - Standard (12GB): OK for small networks (n < 200) and few simulations
   - High-RAM (25GB+): RECOMMENDED for larger networks or many simulations
   - To enable: Runtime → Change runtime type → High-RAM (Colab Pro)

Session Duration:
   - Free Colab: ~90 min timeout, may disconnect
   - Colab Pro: Up to 24h runtime, background execution
   - RECOMMENDED: Colab Pro for long simulation runs

Estimated Runtime (100 simulations × 5 step counts × 3 networks):
   - ~30-60 minutes on standard Colab
   - Faster with Colab Pro (more cores)

TIP: Run in background with Colab Pro to avoid disconnects!
""")
print("=" * 70)


# In[ ]:


# Clone the repository (ai-agents-branch has the latest code)
get_ipython().system('git clone -b ai-agents-branch https://github.com/IgnacioOQ/e_network_inequality')


# In[ ]:


# Install required packages (including sklearn for AUC-ROC)
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
import random as rd
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy import stats

# Import from net_epistemology package
from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent, BayesAgent
from net_epistemology.core.model import Model
from net_epistemology.utils.network_utils import *
from net_epistemology.utils.network_generation import *
# Import only what we need from non-vectorized (for parameter generation)
from net_epistemology.simulation.simulation_functions import generate_parameters_aggregate
# Vectorized simulations for performance
from net_epistemology.simulation.vectorized_simulation_functions import *

print("✅ All imports successful!")


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')

dumping_path = '/content/drive/My Drive/Colab Projects/Data Driven ABMs/Data Sets/ignacio_playground/root_influence/'
# Create directory if it doesn't exist
os.makedirs(dumping_path, exist_ok=True)
print("Output Directory:", dumping_path)


# # Simulation Functions

# ## Parameter Generation

# In[ ]:


def generate_parameters_here(_,G,method='randomization'):
    """
    Generates parameters for a single simulation run, including network modification.
    """
    process_seed = int.from_bytes(os.urandom(4), byteorder='little')
    rd.seed(process_seed)

    # Randomly sample parameters for this group
    uncertainty = rd.uniform(.000001, .001)
    n_experiments = rd.randint(1000, 10000)

    # now we pick a random number
    # Re-instating the 1/3 limit as a potential fix for the
    # "Sample larger than population" ValueError seen in `equalize`.
    # This caps the number of edges to be modified.
    proportion_edges = rd.random() * (1/3)

    # Do randomization

    num_edges = G.number_of_edges()
    modified_network = G.copy() # Default to a copy

    if method == 'randomization':
      num_edges_to_randomize = int(num_edges * proportion_edges)
      modified_network = randomize_network(G, n_edges=num_edges_to_randomize)
    elif method == 'equalize':
      num_edges_to_randomize = int(num_edges * proportion_edges)
      modified_network = equalize(G, num_edges_to_randomize)
    elif method == 'densify':
      num_edges_to_add = int(num_edges * proportion_edges)
      modified_network = densify_fancy_speed_up(G,num_edges_to_add,target_degree_dist='original',keep_density_fixed=False)
    elif method == 'densify_fixed':
      num_edges_to_add = int(num_edges * proportion_edges)
      modified_network = densify_fancy_speed_up(G,num_edges_to_add,target_degree_dist='uniform',keep_density_fixed=True)
    elif method =='cluster':
      num_edges_to_add = int(num_edges * proportion_edges)
      modified_network = cluster_network(G,num_edges_to_add)
    elif method =='decluster':
      num_edges_to_randomize = int(num_edges * proportion_edges)
      modified_network = decluster_network(G,num_keys_to_randomize)
    else:
      print(f"Warning: Method '{method}' not recognized. No network modification applied.")

    result = generate_parameters_aggregate(modified_network, uncertainty=uncertainty, n_experiments=n_experiments,
                                           p_rewiring=proportion_edges)
    result['uncertainty'] = uncertainty
    result['n_experiments'] = n_experiments
    result['proportion_edges'] = proportion_edges
    return result


# ## PARAMETERS

# In[ ]:


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Output path
dumping_path = '/content/drive/My Drive/Colab Projects/Data Driven ABMs/Data Sets/ignacio_playground/root_influence/'

# Number of simulations per network TYPE (not per configuration)
N_SIMULATIONS_PER_TYPE = 100

# Maximum steps (fallback if AUC threshold not reached)
MAX_STEPS = 2000000  # 2 million steps max

# AUC-based stopping parameters
AUC_STOPPING = True
AUC_THRESHOLD = 0.95  # Stop when node-level AUC-ROC >= 0.95
AUC_CHECK_INTERVAL = 500  # Check AUC every 500 steps

# Representative network sizes
n_sizes = [50, 100, 200]

# Erdős-Rényi parameters (reduced - no randomization)
er_p_values = [0.05]  # Single representative value

# Watts-Strogatz parameters (small-world, no randomization)
ws_k_values = [4]  # Neighbors (must be even)
ws_p_values = [0.1, 0.3]  # Rewiring probability

# Barabási-Albert parameters (scale-free, no randomization)
ba_m_values = [2, 4]  # Edges per new node

# Methods for empirical network randomization
methods = ['randomization']

print("Parameters configured:")
print(f"  Simulations per network type: {N_SIMULATIONS_PER_TYPE}")
print(f"  Max steps: {MAX_STEPS:,}")
print(f"  AUC-based stopping: {AUC_STOPPING}")
print(f"  AUC threshold: {AUC_THRESHOLD} (stop when AUC >= {AUC_THRESHOLD})")
print(f"  AUC check interval: every {AUC_CHECK_INTERVAL} steps")
print(f"  Network sizes: {n_sizes}")


# ## Load Empirical Network

# In[ ]:


def load_empirical_network():
    """Load the pud_final.json empirical network."""
    import json
    
    network_path = 'data/empirical_networks/pud_final.json'
    print(f"Loading empirical network from: {network_path}")
    
    with open(network_path, 'r') as f:
        network_data = json.load(f)
    
    if 'links' in network_data and 'edges' not in network_data:
        network_data['edges'] = network_data['links']
    
    network = nx.node_link_graph(network_data, edges="edges")
    
    in_degrees = dict(network.in_degree())
    root_nodes = [n for n, d in in_degrees.items() if d == 0]
    
    print(f"Network: {len(network.nodes())} nodes, {len(network.edges())} edges")
    print(f"Root nodes: {len(root_nodes)}")
    
    return network


# ## Simplified Parameter Generation (No Randomization)

# In[ ]:


def generate_simple_params(G, uncertainty=None, n_experiments=None):
    """
    Generate parameters for a network WITHOUT randomization.
    Used for synthetic networks (ER, WS, BA).
    """
    import uuid
    
    if uncertainty is None:
        uncertainty = rd.uniform(0.000001, 0.001)
    if n_experiments is None:
        n_experiments = rd.randint(1000, 10000)
    
    params = {
        "unique_id": uuid.uuid4().hex,
        "n_agents": len(G.nodes()),
        "network": G,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        "proportion_edges": 0.0,  # No randomization
    }
    
    # Add network statistics
    stats = network_statistics(G)
    for stat in stats.keys():
        params[stat] = stats[stat]
    
    return params


def generate_randomized_params(_, G, method='randomization'):
    """
    Generate parameters for empirical network WITH randomization.
    Only used for pud_final.
    """
    import uuid
    
    process_seed = int.from_bytes(os.urandom(4), byteorder='little')
    rd.seed(process_seed)
    
    uncertainty = rd.uniform(0.000001, 0.001)
    n_experiments = rd.randint(1000, 10000)
    proportion_edges = rd.random() * (1/3)  # Up to 1/3 of edges modified
    
    num_edges = G.number_of_edges()
    modified_network = G.copy()
    
    if method == 'randomization':
        num_edges_to_randomize = int(num_edges * proportion_edges)
        modified_network = randomize_network(G, n_edges=num_edges_to_randomize)
    
    params = {
        "unique_id": uuid.uuid4().hex,
        "n_agents": len(modified_network.nodes()),
        "network": modified_network,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        "proportion_edges": float(proportion_edges),
    }
    
    stats = network_statistics(modified_network)
    for stat in stats.keys():
        params[stat] = stats[stat]
    
    return params


# ## Run Simulations Until Epsilon Convergence

# In[ ]:


def run_simulations_to_convergence(G, network_name, network_type, n_simulations, num_cores, 
                                    randomize=False, agent_type='beta'):
    """
    Run simulations until epsilon threshold is reached.
    
    Args:
        G: Network graph
        network_name: Name for output files
        network_type: 'er', 'ws', 'ba', or 'empirical'
        n_simulations: Number of simulations to run
        num_cores: CPU cores for parallel processing
        randomize: If True, randomize network (only for empirical)
        agent_type: 'beta' or 'bayes'
    
    Returns:
        DataFrame with results, or None if skipped
    """
    # Count root nodes (nodes with in-degree = 0)
    in_degrees = dict(G.in_degree())
    n_roots = sum(1 for d in in_degrees.values() if d == 0)
    
    print(f"\n{'='*60}")
    print(f"Network: {network_name} ({network_type})")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Root nodes: {n_roots}")
    print(f"Randomize: {randomize}, Simulations: {n_simulations}")
    print(f"{'='*60}")
    
    # Skip if no root nodes (strongly connected network)
    if n_roots == 0:
        print(f"⚠️  SKIPPING: Network has 0 root nodes (strongly connected).")
        print(f"    Root influence analysis requires at least 1 root node.")
        return None
    
    # Generate parameters
    if randomize:
        generate_params = partial(generate_randomized_params, G=G, method='randomization')
        param_list = []
        with Pool(num_cores) as pool:
            param_list = list(tqdm.tqdm(
                pool.imap_unordered(generate_params, range(n_simulations)),
                total=n_simulations,
                desc="Generating randomized params"
            ))
    else:
        # For non-randomized, generate params serially (same network)
        param_list = [generate_simple_params(G) for _ in range(n_simulations)]
    
    # Run simulations with AUC-based stopping
    run_sim = partial(
        run_vectorized_simulation_with_params,
        tolerance=0.01,  # Fallback tolerance
        tstep_stopping=False,  # Disable tstep stopping
        agent_type=agent_type,
        compute_root_analysis=True,
        number_of_steps=MAX_STEPS,
        auc_stopping=AUC_STOPPING,
        auc_threshold=AUC_THRESHOLD,
        auc_check_interval=AUC_CHECK_INTERVAL
    )
    
    results = []
    with Pool(num_cores) as pool:
        results = list(tqdm.tqdm(
            pool.imap_unordered(run_sim, param_list),
            total=len(param_list),
            desc=f"Running {network_name}"
        ))
    
    # Clean and save results
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() 
                   if isinstance(v, (int, float, str, bool, type(None)))}
        clean_r['network_type'] = network_type
        clean_r['network_name'] = network_name
        clean_r['randomized'] = randomize
        clean_results.append(clean_r)
    
    df = pd.DataFrame(clean_results)
    
    # Save results
    results_path = dumping_path + f"root_analysis_{network_name}.csv"
    df.to_csv(results_path, index=False)
    
    # Print summary
    print(f"\n--- Results Summary ---")
    print(f"Saved: {results_path}")
    print(f"Simulations: {len(df)}")
    if 'convergence_step' in df.columns:
        print(f"Avg convergence steps: {df['convergence_step'].mean():.0f}")
    if 'proportion_reached_by_truth' in df.columns and 'share_of_correct_agents_at_convergence' in df.columns:
        predicted = df['proportion_reached_by_truth'].mean()
        actual = df['share_of_correct_agents_at_convergence'].mean()
        gap = actual - predicted
        print(f"Predicted (root influence): {predicted:.4f}")
        print(f"Actual (share believing truth): {actual:.4f}")
        print(f"Gap: {gap:+.4f}")
    
    return df


# ## Main Simulation Runner

# In[ ]:


def root_analysis_main(n_simulations=100, agent_type='beta'):
    """
    Run root influence analysis on all network types:
    - Erdős-Rényi (no randomization)
    - Watts-Strogatz (no randomization)
    - Barabási-Albert (no randomization)
    - pud_final empirical (WITH randomization)
    
    100 simulations per network TYPE.
    """
    try:
        num_cores = cpu_count()
    except:
        num_cores = 1
    
    print(f"\n{'#'*70}")
    print("ROOT INFLUENCE ANALYSIS - REFINED")
    print(f"{'#'*70}")
    print(f"Cores: {num_cores}")
    print(f"Simulations per type: {n_simulations}")
    print(f"Agent type: {agent_type}")
    print(f"AUC threshold: {AUC_THRESHOLD} (stop when AUC >= {AUC_THRESHOLD})")
    print(f"AUC check interval: every {AUC_CHECK_INTERVAL} steps")
    print(f"Max steps: {MAX_STEPS:,}")
    
    all_results = []
    
    # ─── 1. ERDŐS-RÉNYI (no randomization) ───
    print(f"\n{'='*70}")
    print("PART 1: ERDŐS-RÉNYI NETWORKS (no randomization)")
    print(f"{'='*70}")
    
    for n in n_sizes:
        for p in er_p_values:
            G = nx.erdos_renyi_graph(n, p, directed=True)
            name = f"er_n{n}_p{p:.2f}"
            df = run_simulations_to_convergence(
                G, name, 'er', n_simulations, num_cores, 
                randomize=False, agent_type=agent_type
            )
            if df is not None:
                all_results.append(df)
    
    # ─── 2. WATTS-STROGATZ (no randomization) ───
    print(f"\n{'='*70}")
    print("PART 2: WATTS-STROGATZ NETWORKS (no randomization)")
    print(f"{'='*70}")
    
    for n in n_sizes:
        for k in ws_k_values:
            if k >= n:
                continue
            for p in ws_p_values:
                G_undirected = nx.watts_strogatz_graph(n, k, p)
                G = G_undirected.to_directed()
                name = f"ws_n{n}_k{k}_p{p:.2f}"
                df = run_simulations_to_convergence(
                    G, name, 'ws', n_simulations, num_cores,
                    randomize=False, agent_type=agent_type
                )
                if df is not None:
                    all_results.append(df)
    
    # ─── 3. BARABÁSI-ALBERT (no randomization) ───
    print(f"\n{'='*70}")
    print("PART 3: BARABÁSI-ALBERT NETWORKS (no randomization)")
    print(f"{'='*70}")
    
    for n in n_sizes:
        for m in ba_m_values:
            if m >= n:
                continue
            G_undirected = nx.barabasi_albert_graph(n, m)
            G = G_undirected.to_directed()
            name = f"ba_n{n}_m{m}"
            df = run_simulations_to_convergence(
                G, name, 'ba', n_simulations, num_cores,
                randomize=False, agent_type=agent_type
            )
            if df is not None:
                all_results.append(df)
    
    # ─── 4. EMPIRICAL NETWORK (WITH randomization) ───
    print(f"\n{'='*70}")
    print("PART 4: EMPIRICAL NETWORK - pud_final (WITH randomization)")
    print(f"{'='*70}")
    
    try:
        G_empirical = load_empirical_network()
        df = run_simulations_to_convergence(
            G_empirical, "pud_final_randomized", 'empirical', n_simulations, num_cores,
            randomize=True, agent_type=agent_type
        )
        if df is not None:
            all_results.append(df)
    except Exception as e:
        print(f"Error with empirical network: {e}")
        import traceback
        traceback.print_exc()
    
    # ─── Combine all results ───
    print(f"\n{'='*70}")
    print("COMBINING ALL RESULTS")
    print(f"{'='*70}")
    
    # Filter out None values  
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("⚠️  No valid results to combine!")
        return None
    
    combined_df = pd.concat(valid_results, ignore_index=True)
    combined_path = dumping_path + "root_analysis_all_networks.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined results saved: {combined_path}")
    print(f"Total simulations: {len(combined_df)}")
    
    return combined_df


# # Root Influence Plotting Functions

# In[ ]:


def root_influence_scatter(df, title_prefix=""):
    """
    Creates a scatter plot comparing predicted vs actual outcomes.
    """
    if 'proportion_reached_by_truth' not in df.columns:
        print("Warning: 'proportion_reached_by_truth' column not found.")
        return
    
    predicted = df['proportion_reached_by_truth']
    actual = df['share_of_correct_agents_at_convergence']
    
    correlation = stats.pearsonr(predicted, actual)[0]
    mae = np.mean(np.abs(predicted - actual))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter with diagonal
    ax = axes[0]
    ax.scatter(predicted, actual, alpha=0.5, edgecolors='black', linewidths=0.5)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Predicted (Root Influence)', fontsize=11)
    ax.set_ylabel('Actual (Share Believing Truth)', fontsize=11)
    ax.set_title(f'{title_prefix}Root Influence (r={correlation:.3f}, MAE={mae:.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gap distribution
    ax = axes[1]
    gaps = actual - predicted
    ax.hist(gaps, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(gaps.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {gaps.mean():.3f}')
    ax.set_xlabel('Gap (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Gap Distribution (std={gaps.std():.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Correlation: {correlation:.4f}, MAE: {mae:.4f}, Mean Gap: {gaps.mean():.4f}")


# In[ ]:


def plot_by_network_type(df):
    """Plot results grouped by network type."""
    if 'network_type' not in df.columns:
        print("'network_type' column not found.")
        return
    
    network_types = df['network_type'].unique()
    
    fig, axes = plt.subplots(1, len(network_types), figsize=(5*len(network_types), 5))
    if len(network_types) == 1:
        axes = [axes]
    
    for ax, ntype in zip(axes, network_types):
        subset = df[df['network_type'] == ntype]
        predicted = subset['proportion_reached_by_truth']
        actual = subset['share_of_correct_agents_at_convergence']
        
        ax.scatter(predicted, actual, alpha=0.5, edgecolors='black', linewidths=0.3)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
        
        r = stats.pearsonr(predicted, actual)[0] if len(predicted) > 2 else 0
        ax.set_title(f'{ntype.upper()} (r={r:.3f}, n={len(subset)})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_convergence_steps(df):
    """Plot distribution of convergence steps by network type."""
    if 'convergence_step' not in df.columns:
        print("'convergence_step' column not found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    network_types = df['network_type'].unique()
    for ntype in network_types:
        subset = df[df['network_type'] == ntype]
        ax.hist(subset['convergence_step'], bins=30, alpha=0.6, label=f'{ntype.upper()}')
    
    ax.set_xlabel('Convergence Steps', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Steps to Convergence by Network Type', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_node_accuracy(df):
    """Plot node-level accuracy metrics by network type."""
    if 'node_accuracy' not in df.columns:
        print("'node_accuracy' column not found.")
        return
    
    # Filter out None values
    df_valid = df[df['node_accuracy'].notna()]
    if len(df_valid) == 0:
        print("No valid node_accuracy values to plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Node accuracy distribution
    ax = axes[0]
    network_types = df_valid['network_type'].unique() if 'network_type' in df_valid.columns else ['all']
    for ntype in network_types:
        if 'network_type' in df_valid.columns:
            subset = df_valid[df_valid['network_type'] == ntype]
        else:
            subset = df_valid
        ax.hist(subset['node_accuracy'], bins=20, alpha=0.6, label=f'{ntype.upper()}')
    
    ax.set_xlabel('Node-Level Accuracy', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Node-Level Prediction Accuracy (mean={df_valid["node_accuracy"].mean():.4f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Node accuracy by network type (box plot)
    ax = axes[1]
    if 'network_type' in df_valid.columns and len(network_types) > 1:
        data_by_type = [df_valid[df_valid['network_type'] == nt]['node_accuracy'].dropna() for nt in network_types]
        ax.boxplot(data_by_type, labels=[nt.upper() for nt in network_types])
        ax.set_xlabel('Network Type', fontsize=11)
        ax.set_ylabel('Node Accuracy', fontsize=11)
        ax.set_title('Node Accuracy by Network Type', fontsize=12)
    else:
        ax.hist(df_valid['node_accuracy'], bins=30, edgecolor='black')
        ax.set_xlabel('Node Accuracy', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Node Accuracy Distribution', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n--- Node-Level Accuracy Summary ---")
    print(f"Mean node accuracy: {df_valid['node_accuracy'].mean():.4f}")
    if 'node_auc_roc' in df_valid.columns:
        auc_valid = df_valid['node_auc_roc'].dropna()
        if len(auc_valid) > 0:
            print(f"Mean AUC-ROC: {auc_valid.mean():.4f} (n={len(auc_valid)})")
        else:
            print("No valid AUC-ROC values (predictions may all be same class)")


# ## Main Plotting Function

# In[ ]:


def run_root_plotting():
    """Load combined results and generate all plots."""
    print("\n--- Loading Combined Results ---")
    
    combined_path = dumping_path + "root_analysis_all_networks.csv"
    
    try:
        df = pd.read_csv(combined_path)
        print(f"Loaded {len(df)} results from {combined_path}")
        print(f"Network types: {df['network_type'].unique() if 'network_type' in df.columns else 'N/A'}")
        
        # 1. Overall scatter plot
        print("\n--- Overall Root Influence Analysis ---")
        root_influence_scatter(df, title_prefix="All Networks: ")
        
        # 2. By network type
        print("\n--- By Network Type ---")
        plot_by_network_type(df)
        
        # 3. Convergence steps distribution
        print("\n--- Convergence Steps ---")
        plot_convergence_steps(df)
        
        # 4. Node-level accuracy (NEW)
        print("\n--- Node-Level Accuracy ---")
        plot_node_accuracy(df)
        
        # 5. Summary statistics (updated)
        print("\n--- Summary Statistics ---")
        if 'network_type' in df.columns:
            agg_dict = {
                'proportion_reached_by_truth': 'mean',
                'share_of_correct_agents_at_convergence': 'mean',
                'convergence_step': 'mean',
            }
            if 'node_accuracy' in df.columns:
                agg_dict['node_accuracy'] = 'mean'
            if 'node_auc_roc' in df.columns:
                agg_dict['node_auc_roc'] = lambda x: x.dropna().mean() if len(x.dropna()) > 0 else None
            
            summary = df.groupby('network_type').agg(agg_dict).round(4)
            print(summary)
        
    except FileNotFoundError:
        print(f"File not found: {combined_path}")
        print("Run root_analysis_main() first to generate results.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# # Run Simulations

# In[ ]:


# Run root analysis simulations
if __name__ == '__main__':
    root_analysis_main(
        n_simulations=N_SIMULATIONS_PER_TYPE,  # 100 per network type
        agent_type='beta'
    )


# # Load Results and Plot

# In[ ]:


run_root_plotting()


# # Disconnect from Runtime

# In[ ]:


from datetime import datetime
import pytz
from IPython.display import Javascript

nyc_time = datetime.now(pytz.timezone('America/New_York'))
formatted_time = nyc_time.strftime('%Y-%m-%d %H:%M:%S %Z')
print(f"✅ Disconnected from runtime at: {formatted_time}")
display(Javascript('google.colab.kernel.disconnect()'))
