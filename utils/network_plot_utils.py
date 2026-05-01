"""
Standalone helpers to turn a pandas DataFrame that contains an OpenAlex
query (or any citation list) into a bundled citation graph.

Typical use inside a notebook
-----------------------------
from network_plot_utils import create_citation_graph, draw_citation_graph

G = create_citation_graph(kuramoto_df)           # build NetworkX graph
fig, ax = draw_citation_graph(G)                 # plot with bundling

# or reuse an existing axis (e. g. the one returned by datamapplot)
fig, base_ax = datamapplot.create_plot(...)[0:2]
draw_citation_graph(G, ax=base_ax, bundle_edges=True)
"""

from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Optional, Sequence

# the tiny (pure-python) library used in the HF-app
from NetworkInequality.edgebundling import run_and_plot_bundling

def _rgba_from_hex(hex_color: str) -> Tuple[float, float, float, float]:
    """'#rrggbb' ➜ (r, g, b, 1.0) in 0-1 range – handy for edge gradients."""
    return tuple(int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (
        1.0,
    )


# --------------------------------------------------------------------------- #
#   Public API
# --------------------------------------------------------------------------- #
def create_citation_graph(
    df: pd.DataFrame,
    *,
    id_col: str = "id",
    x_col: str = "x",
    y_col: str = "y",
    refs_col: str = "referenced_works",
    pub_year_col: str = "publication_year",
    color_col: str = "color",
    cmap: Optional[mcolors.Colormap] = None,
) -> nx.Graph:
    """
    Build an UNDIRECTED NetworkX graph from `df`.

    Each node gets:
    * X / Y           – 2-D UMAP coordinates
    * publication_year
    * color           – '#rrggbb' (computed if absent)

    Edges are inserted **only** between works that are *both* present in the
    provided DataFrame.
    """
    g_df = df.copy()

    # --------------------------------------------------------------------- #
    # Ensure that we have a colour for every node
    # --------------------------------------------------------------------- #
    if color_col not in g_df.columns:
        if cmap is None:
            # defer import because it is *slow* on CPU-only machines
            import colormaps

            cmap = colormaps.haline
        years = pd.to_numeric(g_df[pub_year_col])
        norm = mcolors.Normalize(vmin=years.min(), vmax=years.max())
        g_df[color_col] = [mcolors.to_hex(cmap(norm(y))) for y in years]

    # --------------------------------------------------------------------- #
    # Create the actual graph
    # --------------------------------------------------------------------- #
    G = nx.Graph()
    for _, row in g_df.iterrows():
        G.add_node(
            row[id_col],
            X=row[x_col],
            Y=row[y_col],
            publication_year=row[pub_year_col],
            color=row[color_col],
        )

    id_set = set(g_df[id_col].values)
    for _, row in g_df.iterrows():
        src = row[id_col]
        refs = row[refs_col]

        # harmonise the various formats that come out of pyalex / CSV
        if pd.isna(refs):
            continue
        if isinstance(refs, str):
            refs = [r.strip() for r in refs.split(",")]
        if not isinstance(refs, (list, tuple, pd.Series)):
            continue

        for ref in refs:
            if ref in id_set and src != ref:
                G.add_edge(src, ref)

    return G


def draw_citation_graph(
    G: nx.Graph,
    *,
    ax: Optional[plt.Axes] = None,
    bundle_edges: bool = True,
    min_max_coordinates: Optional[Sequence[float]] = None,
    linewidths: float = 0.8,
    alpha: float = 0.5,
    edge_color: str = "#f98e31",
    node_size: int = 0,
    hammer_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Parameters
    ----------
    G
        Graph created by :func:`create_citation_graph`.
    ax
        Existing axis you want to reuse.  If *None*, a fresh figure/axis is
        created (20 × 20 inch).
    bundle_edges
        Switch between "raw" straight edges and **force-directed edge bundling**.
    min_max_coordinates
        Optional `[xmin, xmax, ymin, ymax]` to keep the plot perfectly aligned
        with an already shown UMAP background.
    linewidths, alpha, edge_color, node_size
        forwarded to Matplotlib / the bundling routine.
    hammer_kwargs
        Optional dictionary of parameters to pass to the hammer bundling algorithm.
        Default values are:
        {
            'accuracy': 500,           # Grid size for density estimation
            'advect_iterations': 50,   # Number of iterations for advection
            'batch_size': 20000,       # Number of edges to process at once
            'decay': 0.01,            # Rate of decay for the gradient force
            'initial_bandwidth': 1.1,  # Initial bandwidth for the kernel
            'iterations': 4,           # Number of bundling iterations
            'max_segment_length': 0.016, # Maximum length of edge segments
            'min_segment_length': 0.008, # Minimum length of edge segments
            'tension': 1.2            # Controls how tightly edges bundle
        }

    Returns
    -------
    (fig, ax)
        The figure / axis that has been drawn on.
    """
    # --------------------------------------------------------------------- #
    # Figure / axis handling
    # --------------------------------------------------------------------- #
    created_new_fig = False
    if ax is None:
        created_new_fig = True
        fig, ax = plt.subplots(figsize=(20, 20))
    else:
        fig = ax.figure

    pos = {n: (G.nodes[n]["X"], G.nodes[n]["Y"]) for n in G.nodes()}

    # --------------------------------------------------------------------- #
    # Actual drawing
    # --------------------------------------------------------------------- #
    if bundle_edges:
        # RGB-look-up for every node (needed for edge-gradient)
        node_rgba = {n: _rgba_from_hex(G.nodes[n]["color"]) for n in G.nodes()}

        # the bundling code can take into account a pre-computed distance
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            G.edges[u, v]["dist"] = np.hypot(x1 - x2, y1 - y2)

        # Create bundling parameters dictionary
        bundling_params = hammer_kwargs or {}

        run_and_plot_bundling(
            G,
            method="hammer",
            ax=ax,
            edge_gradient=True,
            node_colors=node_rgba,
            linewidths=linewidths,
            alpha=alpha,
            bundling_params=bundling_params
        )
    else:  # fall back on a plain, fast straight-line drawing
        nx.draw(
            G,
            pos=pos,
            with_labels=False,
            node_size=node_size,
            edge_color=edge_color,
            alpha=alpha,
            ax=ax,
        )

    # --------------------------------------------------------------------- #
    # Cosmetic fixes
    # --------------------------------------------------------------------- #
    ax.set_aspect("equal")
    ax.axis("off")

    if min_max_coordinates is not None:
        ax.set_xlim(min_max_coordinates[0], min_max_coordinates[1])
        ax.set_ylim(min_max_coordinates[2], min_max_coordinates[3])

    if created_new_fig:
        plt.tight_layout()

    return fig, ax