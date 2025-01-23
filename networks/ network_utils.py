import networkx as nx


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
  G = G.subgraph(largest_cc)
  return G




def calculate_degree_gini(degrees):
    # Sort the degrees in ascending order
    sorted_degrees = sorted(degrees)
    n = len(degrees)

    # Calculate the cumulative sum of the sorted degrees
    cumulative_degrees = sum(sorted_degrees)

    # Calculate the Gini coefficient
    gini_numerator = 0
    for i, degree in enumerate(sorted_degrees):
        gini_numerator += (i + 1) * degree

    gini_denominator = n * cumulative_degrees

    # Gini formula
    gini_coefficient = (2 * gini_numerator) / gini_denominator - (n + 1) / n

    return gini_coefficient




def network_statistics(G):
    stats = {}

    # Number of nodes and edges#
#    stats['number_of_nodes'] = G.number_of_nodes()
#    stats['number_of_edges'] = G.number_of_edges()

    # Average degree
    degrees = [deg for _, deg in G.degree()]
    stats['average_degree'] = sum(degrees) / len(degrees)

    # Gini coefficient
    #print(degrees)
    stats['degree_gini_coefficient'] = calculate_degree_gini(degrees)

    # Approximate average clustering coefficient
    stats['approx_average_clustering_coefficient'] = nx.average_clustering(G)#, trials=50000)

    # Calculate the diameter (approximate)
    if nx.is_connected(G):
        stats['diameter'] = nx.diameter(G)
    else:
        largest_component = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_component)
        stats['diameter'] = nx.diameter(subgraph)

    # Add additional metrics as needed here, e.g., centrality measures

    return stats

