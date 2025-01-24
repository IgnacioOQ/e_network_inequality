from imports import *

def barabasi_albert_directed(n, m):
    """
    Implements the Barabási-Albert model for directed networks.

    Parameters:
        n (int): Total number of nodes in the network.
        m (int): Number of directed edges each new node creates. Must be <= total nodes at any time.

    Returns:
        G (networkx.DiGraph): A directed scale-free network.
    """
    # Ensure valid input
    if m < 1 or m >= n:
        raise ValueError("m must be >= 1 and < n")

    # Create a directed graph
    G = nx.DiGraph()

    # Start with an initial connected directed graph of m nodes
    for i in range(m):
        G.add_node(i)
        for j in range(i):
            G.add_edge(j, i)  # Initial directed edges

    # Add the remaining nodes to the graph
    for new_node in range(m, n):
        # Add the new node
        G.add_node(new_node)

        # Calculate the total in-degree of all existing nodes
        total_in_degree = sum(dict(G.in_degree()).values())

        # Create a list of existing nodes to connect to
        targets = set()
        while len(targets) < m:
            # Preferential attachment: choose a node with probability proportional to its in-degree
            if total_in_degree == 0:
                # If total in-degree is zero, connect randomly
                target = random.choice(list(G.nodes()))
            else:
                # Select node based on preferential attachment
                target = random.choices(
                    list(G.nodes()),
                    weights=[G.in_degree(node) + 1 for node in G.nodes()],  # +1 to avoid zero probability
                    k=1
                )[0]

            # Add the target to the set (ensures unique connections)
            targets.add(target)

        # Add directed edges from the new node to the selected targets
        for target in targets:
            G.add_edge(new_node, target)

    return G