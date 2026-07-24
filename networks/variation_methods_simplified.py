from functools import partial

from utils.imports import copy, np, nx, random

# Helper functions for variation methods


def _sample_edge_from_degree_dist(
    net: nx.DiGraph,
    nodes: list,
    indegrees: dict,
    outdegrees: dict,
    attempts: int = 100,
    _out_weights: list | None = None,
    _in_weights: list | None = None,
) -> tuple | None:
    """Sample a new edge based on degree distribution weights. Note: may return None.

    Parameters
    ----------
    _out_weights, _in_weights : list, optional
        Precomputed weight lists for `nodes`. If provided, skips recomputation.
    """
    if _out_weights is None:
        out_weights = [outdegrees[node] for node in nodes]
        if all(w == 0 for w in out_weights):
            out_weights = list(np.ones(len(nodes)))
    else:
        out_weights = _out_weights

    if _in_weights is None:
        in_weights = [indegrees[node] for node in nodes]
        if all(w == 0 for w in in_weights):
            in_weights = list(np.ones(len(nodes)))
    else:
        in_weights = _in_weights

    sources = random.choices(nodes, weights=out_weights, k=attempts)  # type: ignore
    targets = random.choices(nodes, weights=in_weights, k=attempts)  # type: ignore
    sample_edges = [
        edge
        for edge in zip(sources, targets)
        if edge[0] != edge[1] and not net.has_edge(*edge)
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
    target_clustering: float | None = None,
    keep_density_fixed: bool = False,
) -> tuple:
    """
    Generates a variant of a directed network. Option to fix the density.
    Option to target a specific degree distribution and clustering coefficient.

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
    target_clustering : float, optional
        The desired average clustering coefficient. If None, uses the original
        network's clustering.
    p_conditional : float, optional
        Probability in [0, 1] of using the conditional (sequential) sampling
        approach in the degree branch. With probability 1 - p_conditional the
        independent approach is used instead. Default is 1.0.
    interim_clustering : bool, optional
        If True, the main loop samples within neighbourhoods whenever
        clustering falls below target, to raise it. If False, the degree
        branch always runs freely. Default is True.

    Returns
    -------
    tuple[nx.DiGraph, int, int, int, int]
        (variant, n_clustering_edges, n_degree_edges, n_post_rewires)
    """

    # Set target average clustering if input is None
    target_clustering = float(
        np.average(list(nx.clustering(net).values()))  # type: ignore
        if target_clustering is None
        else target_clustering
    )

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
    # uniform_degrees = {n: 1 for n in variant.nodes()}

    # Create clustering dictionary and compute initial sum for incremental updates
    clustering_dict: dict = nx.clustering(variant)  # type: ignore
    clustering_sum = sum(clustering_dict.values())
    n_nodes = variant.number_of_nodes()

    # Remove edges (conditional)
    if keep_density_fixed:
        num_edges_to_remove = min(n_edges, variant.number_of_edges())
        for _ in range(num_edges_to_remove):
            variant = _remove_edge_avoiding_isolates(variant)
        # Recompute clustering after edge removal
        clustering_dict = nx.clustering(variant)  # type: ignore
        clustering_sum = sum(clustering_dict.values())

    # Precompute node list and weight arrays once (nodes never change, degrees are fixed)
    all_nodes = list(variant.nodes())
    all_out_weights = [target_out_degrees[node] for node in all_nodes]
    if all(w == 0 for w in all_out_weights):
        all_out_weights = list(np.ones(len(all_nodes)))
    all_in_weights = [target_in_degrees[node] for node in all_nodes]
    if all(w == 0 for w in all_in_weights):
        all_in_weights = list(np.ones(len(all_nodes)))

    # Main loop for adding edges
    n_edges_added = 0
    n_clustering_edges = 0
    n_degree_edges = 0
    new_average_clustering = clustering_sum / n_nodes
    while n_edges_added < n_edges:
        if new_average_clustering < target_clustering:
            # Below target and interim_clustering enabled
            # ⇒ sample within neighbourhood to raise clustering
            node = random.choice(all_nodes)
            neighbors = list(variant.predecessors(node)) + list(
                variant.successors(node)
            )
            new_edge = _sample_edge_from_degree_dist(
                variant,
                neighbors,
                target_in_degrees,
                target_out_degrees,
            )
            branch = "clustering"
        else:
            # At/above target_clustering ⇒ degree-distribution sampling
            new_edge = _sample_edge_from_degree_dist(
                variant,
                all_nodes,
                target_in_degrees,
                target_out_degrees,
                _out_weights=all_out_weights,
                _in_weights=all_in_weights,
            )
            branch = "degree"

        # Add edge and update clustering (shared logic)
        if new_edge:
            variant.add_edge(*new_edge)
            n_edges_added += 1
            if branch == "clustering":
                n_clustering_edges += 1
            elif branch == "degree":
                n_degree_edges += 1
            clustering_dict, clustering_sum = _update_clustering(
                clustering_dict, variant, new_edge, clustering_sum
            )
            new_average_clustering = clustering_sum / n_nodes
    return (
        variant,
        n_clustering_edges,
        n_degree_edges,
    )


# generate_densify_variant = partial(
#     generate_network_variant,
#     target_degree_dist="original",
#     keep_density_fixed=False,
# )

generate_equalize_variant = partial(
    generate_network_variant,
    target_degree_dist="uniform",
    keep_density_fixed=True,
)