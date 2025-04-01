from imports import *  

# Plotting functions
def plot_network_degree_distribution(G, directed=True, title='title'):
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
    plt.axvline(mean_value, color='b', linestyle='--', linewidth=2)
    plt.text(mean_value + 0.1, plt.ylim()[1] * 0.9, f'Mean: {mean_value}', color='b')
    # plt.text(mean_value + 0.1, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_value), color='b')

    plt.title('Timeline Smooth Histogram for: ' + title)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.xticks(fontsize=8,rotation=20)
    plt.show()
    
def plot_loglog(G,directed=True,m=10):
    if directed:
        # Get the in-degree of all nodes
        out_degrees = [d for _, d in G.out_degree()]

        # Compute the histogram
        max_degree = max(out_degrees)
        degree_freq = [out_degrees.count(i) for i in range(max_degree + 1)]
    else:
        degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(12, 8))
    plt.loglog(degrees[m:], degree_freq[m:],'go-')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Log-Log plot of the degree distribution')


# Cleaning Functions
def remove_duplicate_nodes_directed(G, consider='both'):
    """
    Remove duplicate nodes in a directed graph based on neighbors.

    Parameters:
    - G: The directed graph (DiGraph).
    - consider: 'both', 'outgoing', or 'incoming' to specify which neighbors to consider for duplication.
    """
    # Dictionary to map neighbor sets to a representative node
    neighbor_dict = {}

    for node in list(G.nodes):
        if consider == 'outgoing':
            # Use only outgoing neighbors
            neighbors = frozenset(G.successors(node))
        elif consider == 'incoming':
            # Use only incoming neighbors
            neighbors = frozenset(G.predecessors(node))
        else:
            # Use both incoming and outgoing neighbors as a single set
            outgoing_neighbors = frozenset(G.successors(node))
            incoming_neighbors = frozenset(G.predecessors(node))
            # Union of incoming and outgoing neighbors
            neighbors = incoming_neighbors.union(outgoing_neighbors)
        if neighbors in neighbor_dict:
            # Remove the current node if it's a duplicate
            G.remove_node(node)
        else:
            # Keep the node as a representative for this neighbor configuration
            neighbor_dict[neighbors] = node

    return G

def remove_duplicate_nodes_undirected(G, consider='both'):
    """
    Remove duplicate nodes in a directed graph based on neighbors.

    Parameters:
    - G: The directed graph (DiGraph).
    - consider: 'both', 'outgoing', or 'incoming' to specify which neighbors to consider for duplication.
    """
    # Dictionary to map neighbor sets to a representative node
    neighbor_dict = {}

    for node in list(G.nodes):
        if consider == 'outgoing':
            # Use only outgoing neighbors
            neighbors = frozenset(G.successors(node))
        elif consider == 'incoming':
            # Use only incoming neighbors
            neighbors = frozenset(G.predecessors(node))
        else:
            # Use both incoming and outgoing neighbors as a single set
            outgoing_neighbors = frozenset(G.successors(node))
            incoming_neighbors = frozenset(G.predecessors(node))
            # Union of incoming and outgoing neighbors
            neighbors = incoming_neighbors.union(outgoing_neighbors)
        if neighbors in neighbor_dict:
            # Remove the current node if it's a duplicate
            G.remove_node(node)
        else:
            # Keep the node as a representative for this neighbor configuration
            neighbor_dict[neighbors] = node

    return G

def get_connected_component(G):
  # Extract largest component:
  largest_cc = max(nx.weakly_connected_components(G), key=len)
#   G = G.subgraph(largest_cc)
  return G.subgraph(largest_cc)

# Network statistics
def calculate_degree_gini(G, directed = True):
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
    scc_list = C.graph['mapping']  # maps node -> scc index
    inverse_scc_map = {}
    for node, scc_id in scc_list.items():
        inverse_scc_map.setdefault(scc_id, []).append(node)

    for source_scc in source_sccs:
        representative = inverse_scc_map[source_scc][0]  # pick one node from this SCC
        reachability_dominator_set.add(representative)

    return len(reachability_dominator_set), len(reachability_dominator_set)/len(G.nodes())

def network_statistics(G, directed = True):
    stats = {}

    # Average degree
    if directed:
        degrees = [deg for _, deg in G.out_degree()]
    else:
        degrees = [deg for _, deg in G.degree()]
    stats['average_degree'] = sum(degrees) / len(degrees)

    # Gini coefficient
    #print(degrees)
    stats['degree_gini_coefficient'] = calculate_degree_gini(G, directed=directed)

    # Compute clustering for each node
    # it allows us to use weights, which we neglect...
    clustering_values = nx.clustering(G)
    # Compute the average clustering coefficient manually
    average_clustering = sum(clustering_values.values()) / len(clustering_values)
    stats['approx_average_clustering_coefficient'] = average_clustering

    if directed:    
        if nx.is_strongly_connected(G):
            stats['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            stats['avg_path_length'] = len(G.nodes)+1
            # largest_component = max(nx.weakly_connected_components(G), key=len)
            # subgraph = G.subgraph(largest_component)
            # stats['diameter'] = nx.diameter(subgraph)
    else:
        if nx.is_connected(G):
            stats['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            stats['avg_path_length'] = len(G.nodes)+1
            # largest_component = max(nx.connected_components(G), key=len)
            # subgraph = G.subgraph(largest_component)
            # stats['diameter'] = nx.diameter(subgraph)

    if directed:
        out_degrees = np.array([d for _, d in G.out_degree()])
        # out_degrees = np.array([d for _, d in graph.out_degree()])
        in_hist, _ = np.histogram(out_degrees, bins=range(np.max(out_degrees) + 2), density=True)
        # out_hist, _ = np.histogram(out_degrees, bins=range(np.max(out_degrees) + 2), density=True)
        out_entropy = -np.sum(in_hist[in_hist > 0] * np.log(in_hist[in_hist > 0]))
        # out_entropy = -np.sum(out_hist[out_hist > 0] * np.log(out_hist[out_hist > 0]))
        stats['degree_entropy'] = out_entropy
    else:
        degrees = np.array([d for _, d in G.degree()])
        hist, _ = np.histogram(degrees, bins=range(np.max(degrees) + 2), density=True)
        entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))
        stats['degree_entropy'] = entropy

    # Add additional metrics as needed here, e.g., centrality measures
    stats['reachability_dominator_set_size'] = find_reachability_dominator_set(G)[0]
    stats['reachability_dominator_set_ratio'] = find_reachability_dominator_set(G)[1]
    H = nx.condensation(G)
    stats['condensation_graph_size'] = len(H.nodes())
    return stats

def scatter_plot(df, target_variable="share_of_correct_agents_at_convergence"):
     # Select numerical columns excluding unique ID and target variable
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    numerical_columns.remove(target_variable)  # Remove target variable from independent variables

    # Generate scatter plots for each numerical column against the target variable
    num_plots = len(numerical_columns)
    fig, axes = plt.subplots(nrows=(num_plots + 1) // 2, ncols=2, figsize=(10, num_plots * 2))
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