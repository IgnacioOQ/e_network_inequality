from imports import *

# This function randomizes a network by rewiring edges with a probability p_rewiring
# it does so in two distinct steps: first identify edges to remove, then add new edges
def randomize_network(G, p_rewiring):
    # Check if the graph is directed
    is_directed = G.is_directed()

    # Get edges and nodes
    edges = list(G.edges()).copy()
    random.shuffle(edges)
    edges_set = set(edges)
    new_edges_set = edges_set.copy()
    nodes = list(G.nodes()).copy()

    to_remove_set = set()
    for old_edge in edges:
        if random.random() < p_rewiring:  # p probability to rewire an edge
            to_remove_set.add(old_edge)
            new_edges_set.remove(old_edge)

    # Generate a new edges
    for edge in to_remove_set:
        new_edge = (random.choice(nodes), random.choice(nodes))
        if not is_directed:
            new_edge = tuple(sorted(new_edge))  # Ensure (u, v) == (v, u) for undirected graphs

        # Avoid duplicate edges and self-loops
        while (new_edge in new_edges_set) or (new_edge[0] == new_edge[1]):
            new_edge = (random.choice(nodes), random.choice(nodes))
            if not is_directed:
                new_edge = tuple(sorted(new_edge))

        new_edges_set.add(new_edge)

    # Create a new graph with updated edges
    G_new = G.copy()
    G_new.remove_edges_from(to_remove_set)
    G_new.add_edges_from(new_edges_set)

    return G_new

def randomize_networkv2(G, p_rewiring):
    # Check if the graph is directed
    is_directed = G.is_directed()

    # Get edges and nodes
    edges = list(G.edges()).copy()
    random.shuffle(edges)
    edges_set = set(edges)
    new_edges_set = edges_set.copy()
    nodes = list(G.nodes()).copy()

    to_remove_set = set()
    for old_edge in edges:
        if random.random() < p_rewiring:  # p probability to rewire an edge
            to_remove_set.add(old_edge)
            new_edges_set.remove(old_edge)

            # Generate a new edge
            new_edge = (random.choice(nodes), random.choice(nodes))
            if not is_directed:
                new_edge = tuple(sorted(new_edge))  # Ensure (u, v) == (v, u) for undirected graphs

            # Avoid duplicate edges and self-loops
            while (new_edge in new_edges_set) or (new_edge[0] == new_edge[1]):
                new_edge = (random.choice(nodes), random.choice(nodes))
                if not is_directed:
                    new_edge = tuple(sorted(new_edge))

            new_edges_set.add(new_edge)

    # Create a new graph with updated edges
    G_new = G.copy()
    G_new.remove_edges_from(to_remove_set)
    G_new.add_edges_from(new_edges_set)

    return G_new

