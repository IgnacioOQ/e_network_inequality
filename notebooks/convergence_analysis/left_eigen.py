import networkx as nx
import numpy as np
import scipy.linalg

def compute_left_eigenvector(G):
    """
    Computes the Left Eigenvector centrality (DeGroot Influence).
    
    In the context of opinion dynamics or influence networks, this metric 
    identifies the 'ultimate' sources of beliefs. It answers the question: 
    "In the long run, how much does this agent's initial state determine 
    the group's final consensus?"
    
    Mathematical Definition:
    ------------------------
    1. Constructs a Row-Stochastic Matrix W where W_ij represents the 
       weight agent i places on agent j (based on incoming edges in G).
    2. If a node has no incoming edges (a Source), it is treated as 
       'stubborn' or 'independent' (weight 1.0 on itself).
    3. Solves pi * W = pi (normalized so sum(pi) = 1).
    
    Parameters:
    -----------
    G : nx.DiGraph
        A directed graph where an edge (u, v) means u influences v.
        (v listens to u).

    Returns:
    --------
    dict
        Dictionary mapping node IDs to their influence score (probability mass).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize adjacency matrix for the "Listening Graph"
    # If G has edge u->v (u influences v), then v listens to u.
    W = np.zeros((n, n))
    
    for u in nodes:
        u_idx = node_to_idx[u]
        # precursors(u) are nodes that point TO u in G.
        # In an influence graph, these are the agents u listens to.
        influencers = list(G.predecessors(u))
        
        if len(influencers) == 0:
            # Case: Independent Agent (Source).
            # In DeGroot dynamics, they listen only to themselves.
            W[u_idx, u_idx] = 1.0
        else:
            # Case: Social Agent.
            # Assuming equal weights for simplicity. 
            # (Can be modified to weight by reliability if data exists).
            weight = 1.0 / len(influencers)
            for inf in influencers:
                v_idx = node_to_idx[inf]
                W[u_idx, v_idx] = weight
                
    # Calculate Left Eigenvector for Eigenvalue 1.
    # Corresponds to the Right Eigenvector of the Transpose matrix.
    # W.T * v = 1 * v
    eigenvalues, eigenvectors = np.linalg.eig(W.T)
    
    # Extract eigenvector corresponding to eigenvalue 1 (or closest to it)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    left_ev = np.real(eigenvectors[:, idx])
    
    # Normalize to form a probability distribution (sum = 1)
    # Use absolute values to handle potential negative signs from solver (rare in stochastic matrices)
    left_ev = np.abs(left_ev) 
    left_ev = left_ev / np.sum(left_ev)
    
    return {nodes[i]: left_ev[i] for i in range(n)}


def compute_katz_centrality(G, alpha=0.1, beta=1.0, measure_influence=True):
    """
    Computes Katz Centrality, optionally on the reversed graph to measure 
    outgoing influence rather than incoming popularity.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The influence network.
    alpha : float
        Attenuation factor. Smaller alpha means influence decays quickly 
        over distance.
    beta : float
        Intrinsic weight. The baseline 'value' of an agent's own experiment.
    measure_influence : bool, default=True
        If True, reverses the graph before calculation. 
        - True: Measures 'Influence' (how many people I reach). High for Sources.
        - False: Measures 'Prestige' (how many people reach me). High for Sinks.

    Returns:
    --------
    dict
        Dictionary of centrality scores.
    """
    if measure_influence:
        # We reverse the graph to track how the agent's 'beta' flows OUT to others.
        target_G = G.reverse()
    else:
        target_G = G
        
    try:
        return nx.katz_centrality(target_G, alpha=alpha, beta=beta, normalized=True)
    except nx.PowerIterationFailedConvergence:
        # Fallback for large/complex graphs: use numpy solver approach
        return nx.katz_centrality_numpy(target_G, alpha=alpha, beta=beta, normalized=True)