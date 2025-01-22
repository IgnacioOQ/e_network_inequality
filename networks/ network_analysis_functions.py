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