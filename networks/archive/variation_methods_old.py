from functools import partial

from utils.imports import copy, np, nx, random

# Helper functions for variation methods


def _sample_edge_from_degree_dist(
    net: nx.DiGraph,
    nodes: list,
    indegrees: dict,
    outdegrees: dict,
    attempts: int = 100,
    conditional: bool = True,
    _out_weights: list | None = None,
    _in_weights: list | None = None,
) -> tuple | None:
    """Sample a new edge based on degree distribution weights. Note: may return None.

    Parameters
    ----------
    conditional : bool
        If False, uses batch zip sampling: sources and targets are drawn independently
        and then zipped. If True (default), uses conditional sampling: for each sampled
        source, the target is drawn only from nodes not already connected to that source,
        preserving preferential attachment more faithfully.
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

    if conditional:
        for _ in range(attempts):
            (source,) = random.choices(nodes, weights=out_weights, k=1)  # type: ignore
            existing_out = set(net.successors(source))
            available_targets = [
                v for v in nodes if v != source and v not in existing_out
            ]
            if not available_targets:
                continue
            avail_in_weights = [indegrees[v] for v in available_targets]
            if all(w == 0 for w in avail_in_weights):
                avail_in_weights = list(np.ones(len(available_targets)))
            (target,) = random.choices(available_targets, weights=avail_in_weights, k=1)  # type: ignore
            return (source, target)
        return None
    else:
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


def _rewire_for_clustering(
    variant: nx.DiGraph,
    target_clustering: float,
    clustering_tolerance: float,
    max_rewires: int,
    clustering_dict: dict,
    clustering_sum: float,
    batch: int = 100,
) -> tuple[nx.DiGraph, dict, float, int]:
    """Degree-preserving edge swaps to bring clustering within the tolerance band.

    Each swap picks two edges (u, v) and (x, y) and replaces them with (u, y)
    and (x, v), preserving every node's degree exactly. The swap is accepted
    only if it moves clustering closer to the target. The loop stops as soon as
    clustering lies within [target - tolerance, target + tolerance], or after
    max_rewires attempts, whichever comes first.

    Edge selection is directed rather than random: when declustering, the first
    edge is biased toward triangle-participating edges and the second is chosen
    to maximise net triangles destroyed (triangles destroyed minus triangles
    created). The mirror strategy is used when clustering is below target.

    Returns the updated (variant, clustering_dict, clustering_sum, n_accepted).
    """

    def nbrs(node: object) -> set:
        return set(variant.predecessors(node)) | set(variant.successors(node))

    n_nodes = variant.number_of_nodes()
    edges = list(variant.edges())
    n_accepted = 0

    # Build triangle-edge sets once; updated lazily after each accepted swap.
    # tri_set:     edges (u,v) where N(u) ∩ N(v) ≠ ∅  (edge is in a triangle)
    # non_tri_set: edges (u,v) where N(u) ∩ N(v) = ∅
    tri_set: set = set()
    non_tri_set: set = set()
    for edge in edges:
        (tri_set if nbrs(edge[0]) & nbrs(edge[1]) else non_tri_set).add(edge)

    for _ in range(max_rewires):
        if abs(clustering_sum / n_nodes - target_clustering) <= clustering_tolerance:
            break
        if len(edges) < 2:
            break

        current_clustering = clustering_sum / n_nodes
        need_decrease = current_clustering > target_clustering

        # First-edge selection
        if need_decrease:
            tri_indices = [i for i, edge in enumerate(edges) if edge in tri_set]
            idx1 = (
                random.choice(tri_indices)
                if tri_indices
                else random.randrange(len(edges))
            )
        else:
            non_tri_indices = [i for i, edge in enumerate(edges) if edge in non_tri_set]
            idx1 = (
                random.choice(non_tri_indices)
                if non_tri_indices
                else random.randrange(len(edges))
            )
        u, v = edges[idx1]
        nbrs_u, nbrs_v = nbrs(u), nbrs(v)
        # Triangles destroyed by removing (u, v) — constant for this first-edge pick
        destroyed_uv = len((nbrs_u - {v}) & (nbrs_v - {u}))

        # Batch-scored second-edge selection
        candidate_indices = random.sample(range(len(edges)), min(batch, len(edges)))
        best_idx2: int | None = None
        best_score: int | None = None

        for idx2 in candidate_indices:
            if idx2 == idx1:
                continue
            x, y = edges[idx2]
            if u == y or x == v or u == x or v == y:
                continue
            if variant.has_edge(u, y) or variant.has_edge(x, v):
                continue
            # Net score = triangles created by (u,y) and (x,v) minus triangles
            # destroyed by removing (u,v) and (x,y).
            nbrs_x, nbrs_y = nbrs(x), nbrs(y)
            created = len(nbrs_u & nbrs_y) + len(nbrs_x & nbrs_v)
            destroyed = destroyed_uv + len((nbrs_x - {y}) & (nbrs_y - {x}))
            score = created - destroyed
            if need_decrease:
                if best_score is None or score < best_score:
                    best_score, best_idx2 = score, idx2
            else:
                if best_score is None or score > best_score:
                    best_score, best_idx2 = score, idx2

        if best_idx2 is None:
            continue
        idx2 = best_idx2
        x, y = edges[idx2]

        # Attempt swap
        variant.remove_edge(u, v)
        variant.remove_edge(x, y)
        variant.add_edge(u, y)
        variant.add_edge(x, v)

        affected = {u, v, x, y}
        for node in (u, v, x, y):
            affected |= set(variant.predecessors(node)) | set(variant.successors(node))
        old_vals = {node: clustering_dict.get(node, 0.0) for node in affected}
        new_vals = {node: nx.clustering(variant, node) for node in affected}
        delta = sum(new_vals[n] - old_vals[n] for n in affected)
        new_clustering = (clustering_sum + delta) / n_nodes

        if abs(new_clustering - target_clustering) < abs(
            current_clustering - target_clustering
        ):
            clustering_dict.update(new_vals)
            clustering_sum += delta
            edges[idx1] = (u, y)
            edges[idx2] = (x, v)
            n_accepted += 1
            # Lazy update of triangle sets
            for old in [(u, v), (x, y)]:
                tri_set.discard(old)
                non_tri_set.discard(old)
            for new in [(u, y), (x, v)]:
                (tri_set if nbrs(new[0]) & nbrs(new[1]) else non_tri_set).add(new)
        else:
            variant.remove_edge(u, y)
            variant.remove_edge(x, v)
            variant.add_edge(u, v)
            variant.add_edge(x, y)

    return variant, clustering_dict, clustering_sum, n_accepted


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
    p_conditional: float = 1.0,
    interim_clustering: bool = True,
    rewiring_tolerance: float = 0.0,
    max_post_rewires: int = 0,
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
    rewiring_tolerance : float, optional
        Absolute tolerance used as the stopping criterion for the post-step
        rewiring. The rewiring stops once clustering is within
        [target - tol, target + tol]. Default is 0.0.
    max_post_rewires : int, optional
        Maximum number of degree-preserving edge swaps to attempt after the main
        loop. The rewiring stops early if clustering falls within the tolerance
        band. Default is 0 (no post-step rewiring).

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
        if new_average_clustering < target_clustering and interim_clustering:
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
            # At/above target or interim_clustering disabled
            # ⇒ degree-distribution sampling
            use_conditional = random.random() < p_conditional
            new_edge = _sample_edge_from_degree_dist(
                variant,
                all_nodes,
                target_in_degrees,
                target_out_degrees,
                _out_weights=all_out_weights,
                _in_weights=all_in_weights,
                conditional=use_conditional,
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
    # Post rewiring (optional):
    # Degree-preserving rewiring to correct residual clustering error
    n_post_rewires = 0
    if max_post_rewires > 0:
        variant, clustering_dict, clustering_sum, n_post_rewires = (
            _rewire_for_clustering(
                variant,
                target_clustering,
                rewiring_tolerance,
                max_post_rewires,
                clustering_dict,
                clustering_sum,
            )
        )
    return (
        variant,
        n_clustering_edges,
        n_degree_edges,
        n_post_rewires,
    )


generate_densify_variant = partial(
    generate_network_variant,
    target_degree_dist="original",
    keep_density_fixed=False,
)

generate_equalize_variant = partial(
    generate_network_variant,
    target_degree_dist="uniform",
    keep_density_fixed=True,
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
