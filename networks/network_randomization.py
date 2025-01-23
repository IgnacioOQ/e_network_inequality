import networkx as nx
import random

def randomize_network(G, p_rewiring):
    edges = list(G.edges()).copy()
    random.shuffle(edges)
    edges_set = set(edges)
    new_edges_set = edges_set
    nodes = list(G.nodes()).copy()
    #new_edges_set = set()
    to_remove_set = set()
    for old_edge in edges:
        if random.random() < p_rewiring:  # p probability to rewire an edge
            to_remove_set.add(old_edge)
            new_edges_set.remove(old_edge)
            new_edge = (random.choice(nodes),rd.choice(nodes))
            while (new_edge in new_edges_set) or (new_edge[0] == new_edge[1]):
                new_edge = (random.choice(nodes),rd.choice(nodes))
            new_edges_set.add(new_edge)
    # Update the graph with new edges
    G_new = G.copy() # not doing this because it takes up memory
    G_new.remove_edges_from(list(to_remove_set))
    G_new.add_edges_from(list(new_edges_set))
    return G_new