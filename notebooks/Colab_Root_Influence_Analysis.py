#!/usr/bin/env python
# coding: utf-8

# # Root Influence Analysis - Colab Notebook
# 
# This notebook runs parallel simulations with root influence analysis enabled.
# It tests the hypothesis that root nodes' beliefs determine the final epistemic state.

# # Setup

# In[ ]:


# !git clone https://github.com/IgnacioOQ/e_network_inequality
get_ipython().system('git clone -b ai-agents-branch https://github.com/IgnacioOQ/e_network_inequality')


# In[ ]:


get_ipython().system('pip install dill')


# In[ ]:


get_ipython().run_line_magic('cd', 'e_network_inequality')


# In[ ]:


from imports import *
from agents import BetaAgent, BayesAgent
from model import Model
from network_utils import *
# notice this is not network_utilsv2
from network_generation import *
from simulation_functions import *
from vectorized_simulation_functions import *
from functools import partial
import hashlib
import gc
from multiprocessing import get_context
from scipy import stats


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')

dumping_path = '/content/drive/My Drive/Colab Projects/Data Driven ABMs/Data Sets/ignacio_playground/root_influence/'
# Create directory if it doesn't exist
import os
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


# PARAMETERS

# Define the path for saving results
dumping_path = '/content/drive/My Drive/Colab Projects/Data Driven ABMs/Data Sets/ignacio_playground/root_influence/'
methods = ['randomization'] # Kept only randomization as requested

# --- Define Network Parameters ---
# Moved to global scope so main() and plotting loop can access them
n_sizes = [200] # Example sizes

# ER parameters
er_p_values = [0.01, 0.05]    # Testing multiple edge probabilities

# WS parameters (k = initial neighbors, p = rewiring prob)
ws_k_values = [4, 6] # k must be even
ws_p_values = [0.01, 0.1]

# BA parameters (m = edges to attach from new node)
ba_m_values = [2, 4]


# ## Vectorized Simulation Method with Root Analysis

# In[ ]:


def run_vect_method_root_analysis(method, G_default, num_cores, n_simulations, network_name_prefix, combine_results=False, agent_type='beta'):
    """
    Runs the full parameter generation and simulation pipeline for a single method
    on a given network, WITH ROOT INFLUENCE ANALYSIS ENABLED.

    Args:
        method (str): The network modification method to test.
        G_default (nx.Graph): The base network to modify.
        num_cores (int): Number of CPU cores for pooling.
        n_simulations (int): Number of simulation runs.
        network_name_prefix (str): Base name for the network (e.g., "er_n100_p0.05").
        combine_results (bool): If True, append to existing results. If False, overwrite.
        agent_type (str): 'beta' or 'bayes'. Default 'beta' for root analysis.
    """

    # ─── Generate parameters ───
    generate_params = partial(generate_parameters_here, G=G_default, method=method)

    param_dict = []
    try:
        with Pool(num_cores) as pool:
            param_dict = list(tqdm.tqdm(
                pool.imap_unordered(generate_params, range(n_simulations)),
                total=n_simulations,
                desc=f"Generating params for {method}"
            ))
    except Exception as e:
        print(f"Error during parameter generation pool for method {method}: {e}")
        return

    if not param_dict:
        print(f"No parameters generated for method {method}. Skipping simulation.")
        return

    # ─── Run simulations WITH ROOT ANALYSIS ───
    run_simulation_wrapper = partial(
        run_vectorized_simulation_with_params, 
        tolerance=1e-5,
        tstep_stopping=True,
        agent_type=agent_type,
        compute_root_analysis=True,  # ENABLE ROOT ANALYSIS
        number_of_steps=50000  # More steps for convergence
    )

    simulation_results = []
    try:
        with Pool(num_cores) as pool:
            simulation_results = list(tqdm.tqdm(
                pool.imap_unordered(run_simulation_wrapper, param_dict),
                total=len(param_dict),
                desc=f"Running simulations for {method}"
            ))
    except Exception as e:
        print(f"Error during simulation pool for method {method}: {e}")
        simulation_results = [r for r in simulation_results if r is not None]
        if not simulation_results:
            print("No simulation results to save.")
            return

    # ─── Save results ───
    results_path = dumping_path + f"root_analysis_{network_name_prefix}_results_{method}.csv"
    
    # Filter out complex objects that can't be saved to CSV
    clean_results = []
    for r in simulation_results:
        clean_r = {k: v for k, v in r.items() 
                   if isinstance(v, (int, float, str, bool, type(None)))}
        clean_results.append(clean_r)
    
    new_results_df = pd.DataFrame(clean_results)

    try:
        if os.path.exists(results_path) and combine_results:
            existing_results_df = pd.read_csv(results_path)
            combined_results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
        else:
            combined_results_df = new_results_df

        combined_results_df.to_csv(results_path, index=False)
        print(f"Total results for {method} ({network_name_prefix}): {len(combined_results_df)}")
        print(f"Saved to: {results_path}")
        
        # Show columns
        print(f"Columns: {list(combined_results_df.columns)}")

    except pd.errors.EmptyDataError:
        print(f"Warning: Existing results file {results_path} was empty. Overwriting.")
        new_results_df.to_csv(results_path, index=False)
        print(f"Total results for {method} ({network_name_prefix}): {len(new_results_df)}")
    except Exception as e:
        print(f"Error saving results to {results_path}: {e}")


# ## Main Simulation Runner

# In[ ]:


def root_analysis_main(combine_results=False, agent_type='beta', n_simulations=100):
    """
    Main function to generate networks, run simulations with root analysis, and save results.
    """

    try:
        num_cores = cpu_count()
    except NotImplementedError:
        print("Warning: Could not determine number of cores. Defaulting to 1.")
        num_cores = 1

    print(f"Number of cores: {num_cores}")
    print(f"Agent type: {agent_type}")
    print(f"Number of simulations per network: {n_simulations}")

    # --- Loop 1: Erdős-Rényi ---
    for n in n_sizes:
        for p in er_p_values:
            network_name_prefix = f"er_n{n}_p{p:.2f}"
            print(f"\n{'='*60}")
            print(f"--- Starting ROOT ANALYSIS simulations for ER Network: {network_name_prefix} ---")
            print(f"{'='*60}")

            # Generate the network
            G_default = nx.erdos_renyi_graph(n, p, directed=True)
            print(f"Network: {G_default.number_of_nodes()} nodes, {G_default.number_of_edges()} edges")
            
            # Count root nodes
            in_degrees = dict(G_default.in_degree())
            root_nodes = [node for node, deg in in_degrees.items() if deg == 0]
            print(f"Root nodes: {len(root_nodes)}")

            for method in methods:
                run_vect_method_root_analysis(
                    method, G_default, num_cores, n_simulations, 
                    network_name_prefix, combine_results=combine_results, 
                    agent_type=agent_type
                )


# # Root Influence Plotting Functions

# In[ ]:


def root_influence_scatter(df, title_prefix=""):
    """
    Creates a scatter plot comparing predicted (proportion_reached_by_truth) 
    vs actual (share_of_correct_agents_at_convergence) outcomes.
    """
    if 'proportion_reached_by_truth' not in df.columns:
        print("Warning: 'proportion_reached_by_truth' column not found. Cannot create root influence plot.")
        return
    
    predicted = df['proportion_reached_by_truth']
    actual = df['share_of_correct_agents_at_convergence']
    
    # Compute statistics
    correlation = stats.pearsonr(predicted, actual)[0]
    mse = np.mean((predicted - actual)**2)
    mae = np.mean(np.abs(predicted - actual))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter with diagonal
    ax = axes[0]
    ax.scatter(predicted, actual, alpha=0.5, edgecolors='black', linewidths=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(predicted.min(), actual.min())
    max_val = max(predicted.max(), actual.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction (y=x)')
    
    ax.set_xlabel('Predicted: Proportion Reachable by Truthful Roots', fontsize=11)
    ax.set_ylabel('Actual: Share Believing Truth', fontsize=11)
    ax.set_title(f'{title_prefix}Root Influence Prediction\n(r={correlation:.3f}, MAE={mae:.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gap distribution
    ax = axes[1]
    gaps = actual - predicted
    ax.hist(gaps, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero gap')
    ax.axvline(gaps.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean gap: {gaps.mean():.3f}')
    ax.set_xlabel('Gap (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{title_prefix}Gap Distribution\n(std={gaps.std():.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"ROOT INFLUENCE ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Correlation (Pearson r): {correlation:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Gap: {gaps.mean():.4f}")
    print(f"Std Gap: {gaps.std():.4f}")
    print(f"{'='*50}")


# In[ ]:


def root_analysis_summary_plot(df, title_prefix=""):
    """
    Creates a comprehensive summary of root influence analysis results.
    """
    if 'n_roots' not in df.columns:
        print("Warning: 'n_roots' column not found. Root analysis may not have been computed.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predicted vs Actual
    ax = axes[0, 0]
    predicted = df['proportion_reached_by_truth']
    actual = df['share_of_correct_agents_at_convergence']
    ax.scatter(predicted, actual, alpha=0.5, c=df['n_roots'], cmap='viridis', edgecolors='black', linewidths=0.3)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax.set_xlabel('Predicted (Root Influence)', fontsize=11)
    ax.set_ylabel('Actual (Share Believing Truth)', fontsize=11)
    ax.set_title('Root Influence Prediction (colored by n_roots)', fontsize=12)
    plt.colorbar(ax.collections[0], ax=ax, label='Number of Roots')
    
    # Plot 2: n_roots distribution
    ax = axes[0, 1]
    ax.hist(df['n_roots'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Root Nodes', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Root Node Distribution (mean={df["n_roots"].mean():.1f})', fontsize=12)
    ax.axvline(df['n_roots'].mean(), color='red', linestyle='--', label=f'Mean: {df["n_roots"].mean():.1f}')
    ax.legend()
    
    # Plot 3: Weighted vs Unweighted truth share
    ax = axes[1, 0]
    if 'weighted_truth_share' in df.columns and 'unweighted_truth_share' in df.columns:
        ax.scatter(df['unweighted_truth_share'], df['weighted_truth_share'], alpha=0.5, edgecolors='black', linewidths=0.3)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
        ax.set_xlabel('Unweighted Truth Share (simple proportion)', fontsize=11)
        ax.set_ylabel('Weighted Truth Share (by descendants)', fontsize=11)
        ax.set_title('Weighted vs Unweighted Root Truth Share', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Weighted/Unweighted data not available', ha='center', va='center')
    
    # Plot 4: Gap vs network properties
    ax = axes[1, 1]
    gaps = actual - predicted
    ax.scatter(df['proportion_edges'], gaps, alpha=0.5, edgecolors='black', linewidths=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Proportion of Edges Modified', fontsize=11)
    ax.set_ylabel('Gap (Actual - Predicted)', fontsize=11)
    ax.set_title('Prediction Gap vs Network Modification', fontsize=12)
    
    plt.suptitle(f'{title_prefix}Root Influence Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ## Main Plotting Function

# In[ ]:


def run_root_plotting():
    """
    Loops through all network parameters and methods, loading results
    and generating root influence analysis plots.
    """
    print("\n--- Starting Root Influence Plotting Phase ---")

    if not methods:
        print("No methods specified. Skipping plotting.")
        return

    method = methods[0]

    # --- Plot Loop 1: Erdős-Rényi ---
    for n in n_sizes:
        for p in er_p_values:
            network_name_prefix = f"er_n{n}_p{p:.2f}"
            results_path = dumping_path + f"root_analysis_{network_name_prefix}_results_{method}.csv"

            try:
                print(f"\n{'='*60}")
                print(f"Loading results for: {network_name_prefix} ({method})")
                print(f"{'='*60}")
                
                combined_results_df = pd.read_csv(results_path)
                print(f"Loaded {len(combined_results_df)} results from path: {results_path}.")
                print(f"Columns: {list(combined_results_df.columns)}")
                
                # Standard plots
                scatter_plot(combined_results_df)
                scatter_plot(combined_results_df, target_variable="convergence_step")
                
                # Root influence specific plots
                root_influence_scatter(combined_results_df, title_prefix=f"{network_name_prefix}: ")
                root_analysis_summary_plot(combined_results_df, title_prefix=f"{network_name_prefix}: ")
                
                print('\n')
            except FileNotFoundError:
                print(f"File not found, skipping: {results_path}\n")
            except Exception as e:
                print(f"Error plotting {results_path}: {e}\n")
                import traceback
                traceback.print_exc()

    print("--- Plotting Phase Complete ---")


# # Run Simulations

# In[ ]:


# Run root analysis simulations
# Using a main guard for multiprocessing safety
if __name__ == '__main__':
    root_analysis_main(
        combine_results=False,
        agent_type='beta',  # Use Beta agents for root analysis
        n_simulations=500   # Adjust as needed
    )


# # Load Results and Plot

# In[ ]:


run_root_plotting()


# # Disconnect from Runtime

# In[ ]:


from datetime import datetime
import pytz
from IPython.display import Javascript

# Get current time in New York
nyc_time = datetime.now(pytz.timezone('America/New_York'))
formatted_time = nyc_time.strftime('%Y-%m-%d %H:%M:%S %Z')

# Print and log
print(f"✅ Disconnected from runtime at: {formatted_time}")

# Disconnect Colab runtime
display(Javascript('google.colab.kernel.disconnect()'))
