from utils.imports import *


def _rng(seed):
    """The RNG to draw from.

    ``seed=None`` returns the ``random`` module itself, so a caller that seeds
    the global RNG (``random.seed(7)``) before calling still gets a reproducible
    graph — which is how every existing call site works. Passing a seed returns
    an isolated generator instead, which is what a worker process wants: under
    multiprocessing the global RNG is inherited on fork, so several workers can
    otherwise draw the identical "random" network without anything failing.
    """
    return random if seed is None else random.Random(seed)


def barabasi_albert_directed(n, m, seed=None):
    """
    Implements the Barabási-Albert model for directed networks.

    Parameters:
        n (int): Total number of nodes in the network.
        m (int): Number of directed edges each new node creates. Must be <= total nodes at any time.
        seed (int | None): If given, draw from an isolated RNG seeded with it.
            If None (default), draw from the global `random` module.

    Returns:
        G (networkx.DiGraph): A directed scale-free network, free of self-loops.
    """
    # Ensure valid input
    if m < 1 or m >= n:
        raise ValueError("m must be >= 1 and < n")

    rng = _rng(seed)

    # Create a directed graph
    G = nx.DiGraph()

    # Start with an initial connected directed graph of m nodes
    for i in range(m):
        G.add_node(i)
        for j in range(i):
            G.add_edge(j, i)  # Initial directed edges

    # Add the remaining nodes to the graph
    for new_node in range(m, n):
        # Snapshot the candidate targets BEFORE the new node joins the graph.
        # Sampling from G.nodes() after adding it lets the node be drawn as its
        # own target, which produced a self-loop on essentially every call —
        # silently, since nothing here rejected one and the graph is otherwise
        # well-formed. Networks loaded elsewhere in this project are asserted to
        # have no self-loops, so a generated one is an inconsistency, not a
        # harmless quirk.
        existing = list(G.nodes())

        # Preferential attachment: probability proportional to out-degree.
        # Out-degree, not in-degree: edges here run from cited to citing, so a
        # heavily cited node is one with many outgoing edges.
        weights = [G.out_degree(node) + 1 for node in existing]  # +1 to avoid zero probability

        # Add the new node
        G.add_node(new_node)

        # Choose m distinct targets among the pre-existing nodes
        targets = set()
        while len(targets) < m:
            targets.add(rng.choices(existing, weights=weights, k=1)[0])

        # Add directed edges from the targets to the new node
        # edges go from cited to citing, so from target to new_node
        for target in targets:
            G.add_edge(target, new_node)

    return G


def directed_watts_strogatz(n, k, p, seed=None):
    """
    Generates a directed Watts-Strogatz small-world network.

    Parameters:
    n (int): Number of nodes
    k (int): Each node is initially connected to k nearest neighbors
    p (float): Probability of rewiring an edge
    seed (int | None): If given, draw from an isolated RNG seeded with it.
        If None (default), draw from the global `random` module.

    Returns:
    nx.DiGraph: A directed Watts-Strogatz network, free of self-loops
    """
    rng = _rng(seed)

    # Step 1: Create a directed ring lattice
    G = nx.DiGraph()
    nodes = list(range(n))

    for i in range(n):
        for j in range(1, k // 2 + 1):  # k//2 neighbors in each direction
            neighbor = (i + j) % n
            G.add_edge(i, neighbor)  # Forward direction
            G.add_edge(neighbor, i)  # Backward direction (ensuring directed edges)

    # Step 2: Rewire edges with probability p
    edges = list(G.edges())  # Get the initial edges
    for edge in edges:
        u, v = edge
        if rng.random() < p:
            # Pick the replacement BEFORE removing the old edge, from the set of
            # legal targets. The previous version removed first and then retried
            # random draws until one was legal, which never terminates once u is
            # already connected to every other node — reachable whenever k
            # approaches n. Enumerating instead makes "no legal target" a case
            # to handle rather than a hang.
            candidates = [w for w in nodes if w != u and not G.has_edge(u, w)]
            if not candidates:
                continue  # nothing legal to rewire to; keep the existing edge

            G.remove_edge(u, v)  # Remove old edge
            G.add_edge(u, rng.choice(candidates))  # Add new directed edge

    return G
