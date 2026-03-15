from functools import partial

from utils.imports import *

# Helper functions for variation methods


def _sample_edge_from_degree_dist(
    net: nx.DiGraph,
    nodes: list,
    indegrees: dict,
    outdegrees: dict,
    attempts: int = 10,
) -> tuple | None:
    """Sample a new edge based on degree distribution weights. Note: may return None."""
    out_weights = [outdegrees[node] for node in nodes]
    if all(w == 0 for w in out_weights):
        out_weights = np.ones(len(nodes))
    in_weights = [indegrees[node] for node in nodes]
    if all(w == 0 for w in in_weights):
        in_weights = np.ones(len(nodes))

    sources = random.choices(
        nodes,
        weights=out_weights,
        k=attempts,
    )
    targets = random.choices(
        nodes,
        weights=in_weights,
        k=attempts,
    )
    sample_edges = [
        (source, target)
        for source in sources
        for target in targets
        if source != target and not net.has_edge(source, target)
    ]
    if sample_edges == []:
        return None
    return random.choice(sample_edges)


def _remove_edge_avoiding_isolates(graph: nx.DiGraph) -> nx.DiGraph:
    """Remove one random edge while avoiding creating isolated nodes."""
    graph_dummy = copy.deepcopy(graph)
    degree_1_nodes = [
        node
        for node in graph.nodes()
        if (graph.in_degree(node) + graph.out_degree(node)) == 1
    ]
    graph_dummy.remove_edge(
        *random.choice(
            [
                edge
                for edge in graph.edges()
                if edge[0] not in degree_1_nodes and edge[1] not in degree_1_nodes
            ]
        )
    )
    return graph_dummy


def _update_clustering(
    clustering_coefs: dict, network: nx.DiGraph, new_edge, clustering_sum: float
) -> tuple[dict, float]:
    """
    Incrementally update clustering coefficients after adding new_edge.

    The key: when edge (u, v) is added, the affected nodes are:
    1. u and v themselves (their degree changed)
    2. all neighbors of u (because they now potentially have a new
       triangle if they're also connected to v)
    3. all neighbors of v (because they now potentially have a new
       triangle if they're also connected to u)

    Returns the updated clustering_coefs dict and the new sum of coefficients.
    """
    u, v = new_edge

    # Get all neighbors (predecessors + successors for directed graphs)
    neighbors_u = set(network.predecessors(u)) | set(network.successors(u))
    neighbors_v = set(network.predecessors(v)) | set(network.successors(v))

    # All affected nodes: u, v, and the union of their neighborhoods
    affected_nodes = {u, v} | neighbors_u | neighbors_v

    # Update only affected nodes and track the change in sum
    for node in affected_nodes:
        old_val = clustering_coefs.get(node, 0.0)
        new_val = nx.clustering(network, node)
        clustering_coefs[node] = new_val
        clustering_sum += new_val - old_val

    return clustering_coefs, clustering_sum


# Main variation method functions


def generate_network_variant(
    net: nx.DiGraph,
    n_edges: int,
    target_degree_dist: str = "original",
    target_average_clustering: float | None = None,
    keep_density_fixed: bool = False,
) -> nx.DiGraph:
    """
    Generates a variant of a directed network. Option to fix the density.
    Option to target a specific degree distribution and clustering coefficient.
    Priority is given to targeting the specified clustering coefficient.

    Parameters
    ----------
    net : nx.DiGraph
        The original directed network to densify.
    n_edges : int
        The number of edges to add.
    target_degree_dist : str, optional
        The target degree distribution for new edges.
        "original" preserves the original degree distribution,
        "uniform" assigns equal probability to all nodes. Default is "original".
    target_average_clustering : float, optional
        The desired average clustering coefficient. If None, uses the original network's clustering.

    Returns
    -------
    nx.DiGraph
        A new directed network with increased density and optionally modified clustering/degree distribution.
    """
    # Set target average clustering if input is None
    if target_average_clustering is None:
        target_average_clustering = np.average(list(nx.clustering(net).values()))

    # Set target degree distribution
    if target_degree_dist == "original":
        target_out_degrees = dict(net.out_degree())
        target_in_degrees = dict(net.in_degree())
    elif target_degree_dist == "uniform":
        target_out_degrees = {node: 1 for node in net.nodes()}
        target_in_degrees = {node: 1 for node in net.nodes()}
    else:
        raise ValueError("target_degree_dist must be 'original' or 'uniform'")

    # Create a copy of the original network
    variant = copy.deepcopy(net)
    uniform_degrees = {n: 1 for n in variant.nodes()}

    # Create clustering dictionary and compute initial sum for incremental updates
    clustering_dict: dict = nx.clustering(variant)
    clustering_sum = sum(clustering_dict.values())
    n_nodes = variant.number_of_nodes()

    # Remove edges (conditional)
    if keep_density_fixed:
        num_edges_to_remove = min(n_edges, variant.number_of_edges())
        for _ in range(num_edges_to_remove):
            variant = _remove_edge_avoiding_isolates(variant)
        # Recompute clustering after edge removal
        clustering_dict = nx.clustering(variant)
        clustering_sum = sum(clustering_dict.values())

    # Main loop for adding edges
    n_edges_added = 0
    new_average_clustering = clustering_sum / n_nodes
    while n_edges_added < n_edges:
        # Choose sampling strategy based on clustering target
        if new_average_clustering < target_average_clustering:
            node = random.choice(list(variant.nodes()))
            neighbors = list(variant.predecessors(node)) + list(
                variant.successors(node)
            )
            new_edge = _sample_edge_from_degree_dist(
                variant,
                neighbors,
                uniform_degrees,
                uniform_degrees,
            )
        else:
            new_edge = _sample_edge_from_degree_dist(
                variant,
                list(variant.nodes()),
                target_in_degrees,
                target_out_degrees,
            )

        # Add edge and update clustering (shared logic)
        if new_edge:
            variant.add_edge(*new_edge)
            n_edges_added += 1
            clustering_dict, clustering_sum = _update_clustering(
                clustering_dict, variant, new_edge, clustering_sum
            )
            new_average_clustering = clustering_sum / n_nodes
    return variant


generate_densify_variant = partial(
    generate_network_variant, target_degree_dist="original", keep_density_fixed=False
)

generate_equalize_variant = partial(
    generate_network_variant, target_degree_dist="uniform", keep_density_fixed=True
)


# ARCHIVE

# Randomization


def randomize_network(G, n_edges: int):
    is_directed = G.is_directed()

    nodes = list(G.nodes())

    # Canonicalize existing edges if undirected
    raw_edges = list(G.edges())
    edges = raw_edges if is_directed else [tuple(sorted(e)) for e in raw_edges]

    random.shuffle(edges)
    new_edges_set = set(edges)

    # Choose edges to remove (already canonicalized if undirected)
    to_remove_set = set(random.sample(edges, k=n_edges))
    new_edges_set.difference_update(to_remove_set)  # <- fixes issue #1 and #2

    # Generate replacement edges (simple rejection is fine for sparse graphs)
    for _ in to_remove_set:
        u, v = random.choice(nodes), random.choice(nodes)
        if not is_directed:
            u, v = sorted((u, v))
        while (u == v) or ((u, v) in new_edges_set):
            u, v = random.choice(nodes), random.choice(nodes)
            if not is_directed:
                u, v = sorted((u, v))
        new_edges_set.add((u, v))

    # Rebuild the edge set on a copy
    G_new = copy.deepcopy(G)
    G_new.clear_edges()
    G_new.add_edges_from(new_edges_set)
    return G_new
