import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from itertools import count
from heapq import heappush, heappop
from collections import defaultdict
import time
import pandas as pd
from datashader.bundling import hammer_bundle  # New import for hammer bundling

###############################################################################
# Minimal AbstractBundling base class (refactored from .abstractBundling import)
###############################################################################
class AbstractBundling:
    def __init__(self, G: nx.Graph):
        self.G = G

    def bundle(self):
        raise NotImplementedError("Subclasses should implement 'bundle'.")

###############################################################################
# Simple SplineC placeholder (refactoring out the nx2ipe dependency)
###############################################################################
class SplineC:
    def __init__(self, points):
        self.points = points

###############################################################################
# A base SpannerBundling class that SpannerBundlingNoSP depends on
###############################################################################
class SpannerBundling(AbstractBundling):
    """
    S-EPB. Implementation
    
    weightFactor: kappa value that sets the bundling strength
    distortion: t value that sets the maximum allowed stretch/distortion
    numWorkers: number of workers that process biconnected components 
    """
    def __init__(self, G: nx.Graph, weightFactor=2, distortion=2, numWorkers=1):
        super().__init__(G)
        self.distortion = distortion
        self.weightFactor = weightFactor
        self.mode = "greedy"
        self.name = None
        self.numWorkers = numWorkers

    @property
    def name(self):
        return f"SEPB_d_{self.distortion}_w_{self.weightFactor}_{self.mode}"

    @name.setter
    def name(self, value):
        self._name = value

    def bundle(self):
        # Default does nothing
        return 0.0

    def process(self, component):
        # Default does nothing
        pass

    def spanner(self, g, k):
        # Default does nothing
        return None

###############################################################################
# The requested SpannerBundlingNoSP class
###############################################################################
class SpannerBundlingNoSP(SpannerBundling):
    """
    S-EPB where instead of computing single source shortest paths we reuse
    shortest paths during the spanner construction. 
    """
    def __init__(self, G: nx.Graph, weightFactor=2, distortion=2):
        super().__init__(G)
        self.distortion = distortion
        self.weightFactor = weightFactor
        self.mode = "reuse"

    def bundle(self):
        """
        Executes the bundling process on all biconnected components.
        Returns the total time for bundling.
        """
        t_start = time.process_time()

        if nx.is_directed(self.G):
            # Convert to undirected for the biconnected components
            GG = self.G.to_undirected(as_view=True)
            components = nx.biconnected_components(GG)
        else:
            components = nx.biconnected_components(self.G)

        to_process = []
        for nodes in components:
            if len(nodes) > 2:
                subg = self.G.subgraph(nodes).copy()
                to_process.append(subg)

        # Sort the components from largest to smallest
        to_process = sorted(to_process, key=lambda x: len(x.nodes()), reverse=True)

        # Process each component
        for comp in to_process:
            self.process(comp)

        return time.process_time() - t_start

    def process(self, component):
        """
        Process a component: build a spanner, then for each edge not in
        the spanner, store a 'path' and create a Spline if possible.
        """
        T = self.spanner(component, self.distortion)

        # Mark edges in T as 'Spanning'
        for u, v, data in T.edges(data=True):
            data["weight"] = np.power(data["dist"], self.weightFactor)

        for u, v in T.edges():
            self.G[u][v]["Layer"] = "Spanning"
            self.G[u][v]["Stroke"] = "blue"

        # For edges not in T, build a spline from the stored path
        for u, v, data in component.edges(data=True):
            if T.has_edge(u, v):
                continue

            path = data.get("path", [])
            if len(path) < 1:
                continue

            spline_points = []
            current = path[0]
            for nxt in path[1:-1]:
                x = component.nodes[nxt].get("X", component.nodes[nxt].get("x", 0))
                y = component.nodes[nxt].get("Y", component.nodes[nxt].get("y", 0))
                spline_points.append((x, y))
                current = nxt

            self.G[u][v]["Spline"] = SplineC(spline_points)
            self.G[u][v]["Layer"] = "Bundled"
            self.G[u][v]["Stroke"] = "purple"

        return

    def spanner(self, g, k):
        """
        Create a spanner and store the shortest path in edge['path'] when the
        edge is not added to the spanner.
        """
        if nx.is_directed(g):
            spanner = nx.DiGraph()
        else:
            spanner = nx.Graph()

        edges = sorted(g.edges(data=True), key=lambda t: t[2].get("dist", 1))

        for u, v, data in edges:
            if u not in spanner.nodes:
                spanner.add_edge(u, v, dist=data["dist"])
                continue
            if v not in spanner.nodes:
                spanner.add_edge(u, v, dist=data["dist"])
                continue

            pred, pathLength = nx.dijkstra_predecessor_and_distance(
                spanner, u, weight="dist", cutoff=k * data["dist"]
            )

            # If v is in pathLength, we store the path in data['path']
            if v in pathLength:
                # reconstruct path from v back to u
                path = []
                nxt = v
                while nxt != u:
                    path.append(nxt)
                    nxt = pred[nxt][0]
                # remove the first node (==v) because we typically want just intermediate
                path = path[1:]
                path.reverse()

                data["path"] = path
            else:
                spanner.add_edge(u, v, dist=data["dist"])

        return spanner

###############################################################################
# Function to plot only the bundled edges (with optional color gradient)
###############################################################################
def plot_bundled_edges_only(G, edge_gradient=False, node_colors=None, ax=None, **plot_kwargs):
    """
    Plots only the edges whose 'Layer' is 'Bundled' (or user-defined).
    Nodes are plotted for reference in black.
    
    Parameters:
        G: NetworkX graph
        title: Plot title
        edge_gradient: If True, color edges with gradient
        node_colors: Dictionary of node colors
        ax: Optional matplotlib axis to plot on. If None, creates new figure.
        **plot_kwargs: Additional keyword arguments passed to LineCollection
    """
    # Use provided axis or create new one
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

    # 1. Extract positions
    pos = {}
    for node, data in G.nodes(data=True):
        x = data.get('X', data.get('x', 0))
        y = data.get('Y', data.get('y', 0))
        pos[node] = (x, y)

    # 2. Assign or retrieve node colors. If your graph doesn't already have
    #    some color-coded attribute, you can define them here. 
    #    For example, let's just fix them to green for demonstration:
    # node_colors = {}
    # for node in G.nodes():
    #     node_colors[node] = (0.0, 0.5, 0.0, 1.0)  # RGBA

    # 3. Build up segments (and possibly per-segment colors) for the edges
    def binomial(n, k):
        """Compute the binomial coefficient (n choose k)."""
        coeff = 1
        for i in range(1, k + 1):
            coeff *= (n - i + 1) / i
        return coeff

    def approxBezier(points, n=50):
        """
        Compute and return n points along a Bezier curve defined by control points.
        """
        X, Y = [], []
        m = len(points) - 1
        binom_vals = [binomial(m, i) for i in range(m + 1)]
        t_values = np.linspace(0, 1, n)
        for t in t_values:
            pX, pY = 0.0, 0.0
            for i, p in enumerate(points):
                coeff = binom_vals[i] * ((1 - t) ** (m - i)) * (t ** i)
                pX += coeff * p[0]
                pY += coeff * p[1]
            X.append(pX)
            Y.append(pY)
        return np.column_stack([X, Y])

    edge_segments = []
    edge_colors = []

    for u, v, data in G.edges(data=True):
        if data.get("Layer", None) != "Bundled":
            # Skip edges not marked as bundled
            continue

        # (a) Gather the control points
        if "Spline" in data and data["Spline"] is not None:
            spline_obj = data["Spline"]
            control_points = list(spline_obj.points)
            # Add the start/end for completeness
            control_points = [pos[u]] + control_points + [pos[v]]
        else:
            # fallback to a straight line
            control_points = [pos[u], pos[v]]

        # (b) Approximate a curve from these control points
        #     We always subdivide if edge_gradient is True. 
        #     If not gradient-based, only subdivide for an actual curve.
        do_subdivide = edge_gradient or (len(control_points) > 2)
        if do_subdivide:
            curve_points = approxBezier(control_points, n=50)
        else:
            curve_points = np.array(control_points)

        # (c) If we're using gradient, we break it into small segments, each with a color
        if edge_gradient:
            c_u = np.array(node_colors[u])  # RGBA for source node
            c_v = np.array(node_colors[v])  # RGBA for target node
            num_pts = len(curve_points)
            for i in range(num_pts - 1):
                p0 = curve_points[i]
                p1 = curve_points[i + 1]
                # fraction along the curve
                t = i / max(1, (num_pts - 2))  
                seg_color = (1 - t) * c_u + t * c_v  # linear interpolation in RGBA
                edge_segments.append([p0, p1])
                edge_colors.append(seg_color)
        else:
            # Single color for the entire edge
            if len(curve_points) > 1:
                edge_segments.append([curve_points[0], curve_points[-1]])
                edge_colors.append((0.5, 0.0, 0.5, 0.9))  # purple RGBA

    # 4. Plot
    # Remove the plt.figure() call since we're using the provided axis

    # Set default values for LineCollection
    lc_kwargs = {
        'linewidths': 1,
        'alpha': 0.9
    }
    
    # If colors weren't explicitly passed and we calculated edge_colors, use them
    if 'colors' not in plot_kwargs and edge_colors:
        lc_kwargs['colors'] = edge_colors
    
    # Update with user-provided kwargs
    lc_kwargs.update(plot_kwargs)
    
    # Create the LineCollection with all parameters
    lc = LineCollection(edge_segments, **lc_kwargs)
    ax.add_collection(lc)

    # The nodes in black
    # node_positions = np.array([pos[n] for n in G.nodes()])
    # ax.scatter(node_positions[:, 0], node_positions[:, 1], color="black", s=20, alpha=0.8)

    # ax.set_aspect('equal')
    # Remove plt.show() since we want to allow further additions to the plot

###############################################################################
# Convenience function to run SpannerBundlingNoSP on a graph and plot results
###############################################################################
def run_and_plot_spanner_bundling_no_sp(G, weightFactor=2, distortion=2, edge_gradient=False, node_colors=None, ax=None, **plot_kwargs):
    """
    Create an instance of SpannerBundlingNoSP, run .bundle(), and
    plot only the bundled edges. Pass edge_gradient=True to see 
    color-gradient edges.
    
    Additional keyword arguments are passed to the LineCollection for edge styling.
    """
    bundler = SpannerBundlingNoSP(G, weightFactor=weightFactor, distortion=distortion)
    bundler.bundle()
    plot_bundled_edges_only(G, 
                          edge_gradient=edge_gradient,
                          node_colors=node_colors,
                          ax=ax,
                          **plot_kwargs)

def run_hammer_bundling(G, accuracy=500, advect_iterations=50, batch_size=20000, 
                       decay=0.01, initial_bandwidth=1.1, iterations=4, 
                       max_segment_length=0.016, min_segment_length=0.008, 
                       tension=1.2):
    """
    Run hammer bundling on a NetworkX graph and return the bundled paths.
    """
    # Create nodes DataFrame
    nodes = []
    node_to_index = {}
    for i, (node, attr) in enumerate(G.nodes(data=True)):
        x = attr.get('X', attr.get('x', 0))
        y = attr.get('Y', attr.get('y', 0))
        nodes.append({'node': node, 'x': x, 'y': y})
        node_to_index[node] = i
    nodes_df = pd.DataFrame(nodes)
    
    # Create edges DataFrame
    edges = []
    for u, v in G.edges():
        edges.append({'source': node_to_index[u], 'target': node_to_index[v]})
    edges_df = pd.DataFrame(edges)
    
    # Apply hammer bundling
    bundled_paths = hammer_bundle(nodes_df, edges_df,
                                accuracy=accuracy,
                                advect_iterations=advect_iterations,
                                batch_size=batch_size,
                                decay=decay,
                                initial_bandwidth=initial_bandwidth,
                                iterations=iterations,
                                max_segment_length=max_segment_length,
                                min_segment_length=min_segment_length,
                                tension=tension)
    
    # Convert bundled paths to a format compatible with our plotting function
    paths = []
    current_path = []
    edge_index = 0
    
    for _, row in bundled_paths.iterrows():
        if pd.isna(row['x']) or pd.isna(row['y']):
            if current_path:
                # Get source and target nodes for this edge
                source_idx = edges_df.iloc[edge_index]['source']
                target_idx = edges_df.iloc[edge_index]['target']
                source_node = nodes_df.iloc[source_idx]['node']
                target_node = nodes_df.iloc[target_idx]['node']
                
                paths.append((source_node, target_node, current_path))
                current_path = []
                edge_index += 1
        else:
            current_path.append((row['x'], row['y']))
    
    if current_path:  # Handle the last path
        source_idx = edges_df.iloc[edge_index]['source']
        target_idx = edges_df.iloc[edge_index]['target']
        source_node = nodes_df.iloc[source_idx]['node']
        target_node = nodes_df.iloc[target_idx]['node']
        paths.append((source_node, target_node, current_path))
    
    return paths

def plot_bundled_edges(G, bundled_paths, edge_gradient=False, node_colors=None, ax=None, **plot_kwargs):
    """
    Generic plotting function that works with both bundling methods.
    
    Parameters:
        G: NetworkX graph
        bundled_paths: List of (source, target, path_points) tuples
        edge_gradient: If True, color edges with gradient
        node_colors: Dictionary of node colors
        ax: Optional matplotlib axis
        **plot_kwargs: Additional styling arguments
    """
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

    def approxBezier(points, n=50):
        """Compute points along a Bezier curve."""
        points = np.array(points)
        t = np.linspace(0, 1, n)
        return np.array([(1-t)*points[:-1] + t*points[1:] for t in t]).reshape(-1, 2)

    edge_segments = []
    edge_colors = []

    for source, target, path_points in bundled_paths:
        points = np.array(path_points)
        
        if edge_gradient:
            # Create segments with gradient colors
            c_u = np.array(node_colors[source])
            c_v = np.array(node_colors[target])
            num_pts = len(points)
            
            for i in range(num_pts - 1):
                p0, p1 = points[i], points[i + 1]
                t = i / max(1, (num_pts - 2))
                seg_color = (1 - t) * c_u + t * c_v
                edge_segments.append([p0, p1])
                edge_colors.append(seg_color)
        else:
            # Single color for the entire path
            for i in range(len(points) - 1):
                edge_segments.append([points[i], points[i + 1]])
                edge_colors.append((0.5, 0.0, 0.5, 0.9))

    # Plot edges
    lc_kwargs = {'linewidths': 1, 'alpha': 0.9}
    if edge_colors:
        lc_kwargs['colors'] = edge_colors
    lc_kwargs.update(plot_kwargs)
    
    lc = LineCollection(edge_segments, **lc_kwargs)
    ax.add_collection(lc)
    ax.autoscale()

def run_and_plot_bundling(G, method='hammer', edge_gradient=False, node_colors=None, ax=None, 
                         bundling_params=None, **plot_kwargs):
    """
    Unified function to run and plot different bundling methods.
    
    Parameters:
        G: NetworkX graph
        method: 'spanner' or 'hammer'
        bundling_params: dict of parameters specific to the bundling method
        Other parameters same as plot_bundled_edges
    """
    bundling_params = bundling_params or {}
    
    if method == 'spanner':
        bundler = SpannerBundlingNoSP(G, **bundling_params)
        bundler.bundle()
        
        # Extract bundled paths from SpannerBundling format
        bundled_paths = []
        for u, v, data in G.edges(data=True):
            if data.get("Layer") == "Bundled" and "Spline" in data:
                spline_points = data["Spline"].points
                pos_u = (G.nodes[u].get('X', G.nodes[u].get('x', 0)), 
                        G.nodes[u].get('Y', G.nodes[u].get('y', 0)))
                pos_v = (G.nodes[v].get('X', G.nodes[v].get('x', 0)), 
                        G.nodes[v].get('Y', G.nodes[v].get('y', 0)))
                path = [pos_u] + list(spline_points) + [pos_v]
                bundled_paths.append((u, v, path))
                
    elif method == 'hammer':
        bundled_paths = run_hammer_bundling(G, **bundling_params)
    else:
        raise ValueError(f"Unknown bundling method: {method}")
    
    plot_bundled_edges(G, bundled_paths, edge_gradient, node_colors, ax, **plot_kwargs)