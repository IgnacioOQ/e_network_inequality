from utils.imports import *


# # Plotting Functions
# Plotting functions
def plot_network_degree_distribution(G, directed=True):
    # Compute density
    density = nx.density(G)
    print(f"Density of the network: {density}")
    if directed:
        degrees = np.array([degree for node, degree in G.out_degree()])
    else:
        degrees = np.array([degree for node, degree in G.degree()])
    # Create the histogram with a KDE
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(degrees, kde=False, bins=150, stat="count")
    # Calculate the mean
    mean_value = np.mean(degrees)
    print(mean_value)
    print(np.median(degrees))

    # Plot a vertical line at the mean value
    plt.axvline(mean_value, color="b", linestyle="--", linewidth=2)
    plt.text(
        mean_value + 0.1, plt.ylim()[1] * 0.9, f"Mean: {mean_value:.3f}", color="b"
    )
    # plt.text(mean_value + 0.1, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_value), color='b')

    plt.title("Network Out-Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.xticks(fontsize=8, rotation=20)
    plt.show()


def plot_loglog(G, directed=True, m=10):
    if directed:
        # Get the in-degree of all nodes
        out_degrees = [d for _, d in G.out_degree()]

        # Compute the histogram
        max_degree = max(out_degrees)
        degree_freq = [out_degrees.count(i) for i in range(max_degree + 1)]
    else:
        degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(8, 6))
    plt.loglog(degrees[m:], degree_freq[m:], "go-")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.xticks(fontsize=8, rotation=20)
    plt.title("Network Out-Degree Distribution Log-Log Plot")


def scatter_plot(df, target_variable="share_of_correct_agents_at_convergence"):
    # Select numerical columns excluding unique ID and target variable
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    numerical_columns.remove(
        target_variable
    )  # Remove target variable from independent variables

    # Generate scatter plots for each numerical column against the target variable
    num_plots = len(numerical_columns)
    fig, axes = plt.subplots(
        nrows=(num_plots + 1) // 2, ncols=2, figsize=(10, num_plots * 2)
    )
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        axes[i].scatter(df[column], df[target_variable], alpha=0.5)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel(target_variable)
        axes[i].set_title(f"{column} vs {target_variable}")
        axes[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# # Network Statistics


# Network statistics
def calculate_degree_gini(G, directed=True):
    if directed:
        degrees = [deg for _, deg in G.out_degree()]
    else:
        degrees = [deg for _, deg in G.degree()]
    # Sort the degrees in ascending order
    sorted_x = np.sort(np.array(degrees))
    n = len(np.array(degrees))
    cumx = np.cumsum(sorted_x, dtype=float)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    return gini


def find_reachability_dominator_set(G):
    """
    Finds a minimal reachability dominator set in a directed graph G.

    Parameters:
        G (nx.DiGraph): A directed graph.

    Returns:
        set: A set of nodes A such that every node in G is reachable from some node in A.
    """
    # Step 1: Compute strongly connected components
    sccs = list(nx.strongly_connected_components(G))

    # Step 2: Build the condensation graph
    C = nx.condensation(G, sccs)

    # Step 3: Find source SCCs (no incoming edges)
    source_sccs = [node for node in C.nodes if C.in_degree(node) == 0]

    # Step 4: Pick one representative node from each source SCC
    reachability_dominator_set = set()
    scc_list = C.graph["mapping"]  # maps node -> scc index
    inverse_scc_map = {}
    for node, scc_id in scc_list.items():
        inverse_scc_map.setdefault(scc_id, []).append(node)

    for source_scc in source_sccs:
        representative = inverse_scc_map[source_scc][0]  # pick one node from this SCC
        reachability_dominator_set.add(representative)

    return (
        len(reachability_dominator_set),
        len(reachability_dominator_set) / len(G),
        len(C),
        len(C) / len(G),
    )


def compute_left_eigenvector(G):
    """
    Computes the Left Eigenvector centrality (DeGroot Influence).

    In the context of opinion dynamics or influence networks, this metric
    identifies the 'ultimate' sources of beliefs. It answers the question:
    "In the long run, how much does this agent's initial state determine
    the group's final consensus?"

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
    # If G has edge u->v (u influences v), then v listens to u.
    W = np.zeros((n, n))

    for u in nodes:
        u_idx = node_to_idx[u]
        # precursors(u) are nodes that point TO u in G.
        # In an influence graph, these are the agents u listens to.
        influencers = list(G.predecessors(u))

        if len(influencers) == 0:
            # Case: Independent Agent (Source).
            # In DeGroot dynamics, they listen only to themselves.
            W[u_idx, u_idx] = 1.0
        else:
            # Case: Social Agent.
            # Assuming equal weights for simplicity.
            # (Can be modified to weight by reliability if data exists).
            weight = 1.0 / len(influencers)
            for inf in influencers:
                v_idx = node_to_idx[inf]
                W[u_idx, v_idx] = weight

    # Calculate Left Eigenvector for Eigenvalue 1.
    # Corresponds to the Right Eigenvector of the Transpose matrix.
    # W.T * v = 1 * v
    eigenvalues, eigenvectors = np.linalg.eig(W.T)

    # Extract eigenvector corresponding to eigenvalue 1 (or closest to it)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    left_ev = np.real(eigenvectors[:, idx])

    # Normalize to form a probability distribution (sum = 1)
    # Use absolute values to handle potential negative signs from solver (rare in stochastic matrices)
    left_ev = np.abs(left_ev)
    left_ev = left_ev / np.sum(left_ev)

    return {nodes[i]: left_ev[i] for i in range(n)}


def compute_katz_centrality(G, alpha=0.1, beta=1.0, measure_influence=True):
    """
    Computes Katz Centrality, optionally on the reversed graph to measure
    outgoing influence rather than incoming popularity.

    Parameters:
    -----------
    G : nx.DiGraph
        The influence network.
    alpha : float
        Attenuation factor. Smaller alpha means influence decays quickly
        over distance.
    beta : float
        Intrinsic weight. The baseline 'value' of an agent's own experiment.
    measure_influence : bool, default=True
        If True, reverses the graph before calculation.
        - True: Measures 'Influence' (how many people I reach). High for Sources.
        - False: Measures 'Prestige' (how many people reach me). High for Sinks.

    Returns:
    --------
    dict
        Dictionary of centrality scores.
    """
    if measure_influence:
        # We reverse the graph to track how the agent's 'beta' flows OUT to others.
        target_G = G.reverse()
    else:
        target_G = G

    try:
        return nx.katz_centrality(target_G, alpha=alpha, beta=beta, normalized=True)
    except nx.PowerIterationFailedConvergence:
        # Fallback for large/complex graphs: use numpy solver approach
        return nx.katz_centrality_numpy(
            target_G, alpha=alpha, beta=beta, normalized=True
        )


def network_statistics(G, directed=True):
    stats = {}

    # Average degree
    if directed:
        degrees = [deg for _, deg in G.out_degree()]
    else:
        degrees = [deg for _, deg in G.degree()]
    stats["average_degree"] = sum(degrees) / len(degrees)

    # Gini coefficient
    # print(degrees)
    stats["degree_gini_coefficient"] = calculate_degree_gini(G, directed=directed)

    # Compute clustering for each node
    # it allows us to use weights, which we neglect...
    clustering_values = nx.clustering(G)
    # Compute the average clustering coefficient manually
    average_clustering = sum(clustering_values.values()) / len(clustering_values)
    stats["approx_average_clustering_coefficient"] = average_clustering

    # commenting out unnecesary metrics to speed up computation
    # if directed:
    #     if nx.is_strongly_connected(G):
    #         stats['avg_path_length'] = nx.average_shortest_path_length(G)
    #     else:
    #         stats['avg_path_length'] = len(G.nodes)+1
    #         # largest_component = max(nx.weakly_connected_components(G), key=len)
    #         # subgraph = G.subgraph(largest_component)
    #         # stats['diameter'] = nx.diameter(subgraph)
    # else:
    #     if nx.is_connected(G):
    #         stats['avg_path_length'] = nx.average_shortest_path_length(G)
    #     else:
    #         stats['avg_path_length'] = len(G.nodes)+1
    #         # largest_component = max(nx.connected_components(G), key=len)
    #         # subgraph = G.subgraph(largest_component)
    #         # stats['diameter'] = nx.diameter(subgraph)

    # if directed:
    #     out_degrees = np.array([d for _, d in G.out_degree()])
    #     # out_degrees = np.array([d for _, d in graph.out_degree()])
    #     in_hist, _ = np.histogram(out_degrees, bins=range(np.max(out_degrees) + 2), density=True)
    #     # out_hist, _ = np.histogram(out_degrees, bins=range(np.max(out_degrees) + 2), density=True)
    #     out_entropy = -np.sum(in_hist[in_hist > 0] * np.log(in_hist[in_hist > 0]))
    #     # out_entropy = -np.sum(out_hist[out_hist > 0] * np.log(out_hist[out_hist > 0]))
    #     stats['degree_entropy'] = out_entropy
    # else:
    #     degrees = np.array([d for _, d in G.degree()])
    #     hist, _ = np.histogram(degrees, bins=range(np.max(degrees) + 2), density=True)
    #     entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))
    #     stats['degree_entropy'] = entropy

    # # Add additional metrics as needed here, e.g., centrality measures
    # stats['reachability_dominator_set_size'] = find_reachability_dominator_set(G)[0]
    # stats['reachability_dominator_set_ratio'] = find_reachability_dominator_set(G)[1]
    # stats['condensation_graph_size'] = find_reachability_dominator_set(G)[2]
    # stats['condensation_graph_ratio'] = find_reachability_dominator_set(G)[3]
    return stats


# # Variation Methods
# ## Helper Functions


def get_triangles(net: nx.DiGraph):
    """Return the list of all triangles in a directed graph G."""
    triangles = []
    for clique in nx.enumerate_all_cliques(net.to_undirected()):
        if len(clique) <= 3:
            if len(clique) == 3:
                triangles.append(clique)
        else:
            return triangles
    return triangles