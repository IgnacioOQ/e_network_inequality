"""Directed simulated annealing workflow for constrained network variation.

Version 1 targets simple, directed, unweighted graphs and mirrors the
undirected notebook workflow with directed definitions:

- density = m / (n * (n - 1))
- degree-gini over a chosen directed degree mode, defaulting to out-degree
- directed transitivity over ordered two-paths

Directed clustering definition
------------------------------
The clustering statistic in this module is a directed global transitivity:

    clustering = closed_directed_triplets / directed_triplets

where a directed triplet is an ordered two-path u -> v -> w with u, v, w
distinct, and it is closed when the shortcut edge u -> w exists.

This choice keeps the coefficient in [0, 1] and allows exact local updates.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import multiprocessing as mp
import os
from pathlib import Path
from typing import Mapping, Sequence

import math
import pickle
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse as sp
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from utils.sa_network_variation import (
    DEFAULT_SUFFIXES,
    bootstrap_ci,
    compute_overall_correlation_matrix,
    compute_overall_correlation_table,
    compute_residualized_correlation_matrix,
    compute_residualized_correlation_table,
    compute_target_intervention_diagnostics,
    degree_gini_from_degrees,
    fisher_z_average,
    load_graph_object,
    partial_spearman,
    plot_correlation_heatmap,
    plot_delta_scatter,
    plot_edit_distance_vs_target_delta,
    plot_final_pairplot,
    plot_requested_vs_achieved,
    plot_target_intervention_diagnostic_heatmap,
    plot_target_intervention_scatter,
    plot_target_delta_vs_edit_distance,
    regression_with_edit_distance,
    resolve_source_paths,
    safe_spearman,
    safe_spearman_matrix,
    save_results_table,
    safe_divide,
)


STAT_NAMES = ("density", "degree_gini", "clustering")
MOVE_KINDS = (
    "add_edge",
    "delete_edge",
    "source_rewire",
    "target_rewire",
    "double_edge_swap",
    "density_repair",
    "edge_repair",
    "degree_preserving_repair",
)
VALID_GINI_MODES = {"out", "in"}
VALID_CONNECTIVITY_MODES = {None, "weak", "strong"}
VALID_TARGET_GRID_MODES = {"pilot", "centered", "explicit"}
VALID_STATE_MATRIX_FORMATS = {"csr", "dense"}
VALID_IN_BAND_LOSS_MODES = {"exact", "zero", "target_zero_preserve_exact"}
VALID_FEASIBLE_SELECTION_MODES = {"weighted", "preserve_first"}


def directed_edge(u: int, v: int) -> tuple[int, int]:
    """Return the canonical directed edge representation."""
    if u == v:
        raise ValueError("self-loops are not allowed in a simple directed graph")
    return (u, v)


def _read_directed_edgelist_dataframe(path: Path) -> nx.DiGraph:
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] < 2:
        raise ValueError(f"{path} does not contain two edge-list columns")
    source_col, target_col = df.columns[:2]
    return nx.from_pandas_edgelist(
        df,
        source=source_col,
        target=target_col,
        create_using=nx.DiGraph,
    )


def normalize_to_simple_directed(
    graph_obj,
    *,
    connectivity_mode: str | None = "weak",
    keep_largest_component: bool = True,
    drop_isolates: bool = False,
) -> nx.DiGraph:
    """Normalize a graph-like input to a simple directed graph."""

    if isinstance(graph_obj, pd.DataFrame):
        graph = nx.from_pandas_edgelist(
            graph_obj,
            source=graph_obj.columns[0],
            target=graph_obj.columns[1],
            create_using=nx.DiGraph,
        )
    elif isinstance(graph_obj, nx.MultiDiGraph):
        graph = nx.DiGraph(graph_obj)
    elif isinstance(graph_obj, nx.MultiGraph):
        graph = nx.DiGraph()
        graph.add_nodes_from(graph_obj.nodes(data=True))
        graph.add_edges_from((u, v) for u, v in graph_obj.edges())
    elif isinstance(graph_obj, nx.DiGraph):
        graph = nx.DiGraph(graph_obj)
    elif isinstance(graph_obj, nx.Graph):
        graph = nx.DiGraph()
        graph.add_nodes_from(graph_obj.nodes(data=True))
        graph.add_edges_from(graph_obj.edges())
    else:
        raise TypeError(f"Unsupported graph object type: {type(graph_obj)!r}")

    graph.remove_edges_from(nx.selfloop_edges(graph))
    if drop_isolates:
        isolates = [
            node
            for node in graph.nodes()
            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0
        ]
        graph.remove_nodes_from(isolates)

    if graph.number_of_nodes() == 0:
        raise ValueError("normalized graph has zero nodes")

    if keep_largest_component and connectivity_mode in {"weak", "strong"}:
        if connectivity_mode == "weak" and not nx.is_weakly_connected(graph):
            largest = max(nx.weakly_connected_components(graph), key=len)
            graph = graph.subgraph(largest).copy()
        elif connectivity_mode == "strong" and not nx.is_strongly_connected(graph):
            largest = max(nx.strongly_connected_components(graph), key=len)
            graph = graph.subgraph(largest).copy()

    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute="original_label"
    )
    return graph


@dataclass(slots=True)
class Move:
    kind: str
    remove: tuple[tuple[int, int], ...] = ()
    add: tuple[tuple[int, int], ...] = ()

    def signature(
        self,
    ) -> tuple[str, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        return (
            self.kind,
            tuple(sorted(self.remove)),
            tuple(sorted(self.add)),
        )


@dataclass
class SAConfig:
    target_stats: tuple[str, ...] = STAT_NAMES
    degree_gini_mode: str = "out"
    connectivity_mode: str | None = "weak"
    keep_largest_component: bool = True
    drop_isolates: bool = False
    target_grid_mode: str = "pilot"
    centered_grid_half_widths: dict[str, float] = field(
        default_factory=lambda: {
            "density": 0.002,
            "degree_gini": 0.01,
            "clustering": 0.02,
        }
    )
    explicit_grid_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "density": (0.001, 0.1),
            "degree_gini": (0.4, 0.99),
            "clustering": (0.05, 0.95),
        }
    )
    target_tolerances: dict[str, float] = field(
        default_factory=lambda: {
            "density": 5e-4,
            "degree_gini": 5e-3,
            "clustering": 5e-3,
        }
    )
    preserve_tolerances: dict[str, float] = field(
        default_factory=lambda: {
            "density": 5e-4,
            "degree_gini": 5e-3,
            "clustering": 5e-3,
        }
    )
    pilot_runs_per_direction: int = 6
    pilot_steps: int = 1500
    attain_steps: int = 12000
    compress_steps: int = 6000
    temperature_block_size: int = 200
    cooling_alpha: float = 0.97
    stall_blocks: int = 12
    pilot_temperature_samples: int = 64
    initial_acceptance_rate: float = 0.7
    targeted_move_prob: float = 0.7
    best_of_k: int = 6
    candidate_trials: int = 96
    node_selection_batch: int = 16
    density_repair_swap_prob: float = 0.25
    n_target_grid: int = 25
    n_seeds: int = 40
    grid_quantiles: tuple[float, float] = (0.05, 0.95)
    random_seed: int = 7
    attain_target_weight: float = 5.0
    attain_preserve_weight: float = 2.5
    attain_distance_weight: float = 0.2
    compress_distance_weight: float = 6.0
    compress_target_weight: float = 0.2
    compress_preserve_weight: float = 0.8
    in_band_loss_mode: str = "target_zero_preserve_exact"
    feasible_preserve_weight: float = 50.0
    feasible_selection_mode: str = "weighted"
    hard_constraint_penalty: float = 1e6
    output_dir: Path = Path("NetworkInequality/sa_variation_outputs_directed")
    save_selected_graphs: bool = False
    max_saved_graphs_per_group: int = 2
    stop_attain_on_first_feasible: bool = True
    attain_no_feasible_proposal_limit: int = 0
    attain_no_feasible_proposal_limits: dict[str, int] = field(default_factory=dict)
    compress_no_improvement_patience: int = 400
    show_progress: bool = True
    progress_desc: str = "SA batch runs"
    record_acceptance_trace: bool = False
    trace_accept_stride: int = 1
    record_state_matrices: bool = False
    state_matrix_stride: int = 1
    state_matrix_format: str = "csr"
    save_run_matrices: bool = False
    n_jobs: int = 1
    parallel_start_method: str | None = None

    def __post_init__(self) -> None:
        if self.degree_gini_mode not in VALID_GINI_MODES:
            raise ValueError(
                f"degree_gini_mode must be one of {tuple(sorted(VALID_GINI_MODES))}"
            )
        if self.connectivity_mode not in VALID_CONNECTIVITY_MODES:
            raise ValueError(
                "connectivity_mode must be one of (None, 'weak', 'strong')"
            )
        if self.target_grid_mode not in VALID_TARGET_GRID_MODES:
            raise ValueError(
                f"target_grid_mode must be one of {tuple(sorted(VALID_TARGET_GRID_MODES))}"
            )
        if self.trace_accept_stride < 1:
            raise ValueError("trace_accept_stride must be >= 1")
        if self.state_matrix_stride < 1:
            raise ValueError("state_matrix_stride must be >= 1")
        if self.state_matrix_format not in VALID_STATE_MATRIX_FORMATS:
            raise ValueError(
                f"state_matrix_format must be one of {tuple(sorted(VALID_STATE_MATRIX_FORMATS))}"
            )
        if self.in_band_loss_mode not in VALID_IN_BAND_LOSS_MODES:
            raise ValueError(
                "in_band_loss_mode must be one of "
                f"{tuple(sorted(VALID_IN_BAND_LOSS_MODES))}"
            )
        if self.feasible_preserve_weight < 0:
            raise ValueError("feasible_preserve_weight must be >= 0")
        if self.feasible_selection_mode not in VALID_FEASIBLE_SELECTION_MODES:
            raise ValueError(
                "feasible_selection_mode must be one of "
                f"{tuple(sorted(VALID_FEASIBLE_SELECTION_MODES))}"
            )
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be nonzero; use 1 for serial or -1 for all cores")
        if (
            self.parallel_start_method is not None
            and self.parallel_start_method not in mp.get_all_start_methods()
        ):
            raise ValueError(
                "parallel_start_method must be one of "
                f"{tuple(mp.get_all_start_methods())}, or None"
            )
        if self.attain_no_feasible_proposal_limit < 0:
            raise ValueError("attain_no_feasible_proposal_limit must be >= 0")
        for stat_name, limit in self.attain_no_feasible_proposal_limits.items():
            if stat_name not in STAT_NAMES:
                raise ValueError(f"Unknown target stat in attain_no_feasible_proposal_limits: {stat_name}")
            if int(limit) < 0:
                raise ValueError("attain_no_feasible_proposal_limits values must be >= 0")
        if self.compress_no_improvement_patience < 0:
            raise ValueError("compress_no_improvement_patience must be >= 0")

    def tolerance_scale(self, stat_name: str, *, target: bool) -> float:
        tolerance_map = self.target_tolerances if target else self.preserve_tolerances
        return max(float(tolerance_map[stat_name]), 1e-9)

    def band_loss_mode(self, *, target: bool) -> str:
        if self.in_band_loss_mode == "target_zero_preserve_exact":
            return "zero" if target else "exact"
        return self.in_band_loss_mode

    def no_feasible_proposal_limit(self, target_name: str) -> int:
        return int(
            self.attain_no_feasible_proposal_limits.get(
                target_name,
                self.attain_no_feasible_proposal_limit,
            )
        )


def load_source_graph(path: str | Path, cfg: SAConfig) -> nx.DiGraph:
    graph_obj = load_graph_object(path)
    if isinstance(graph_obj, pd.DataFrame):
        graph = nx.from_pandas_edgelist(
            graph_obj,
            source=graph_obj.columns[0],
            target=graph_obj.columns[1],
            create_using=nx.DiGraph,
        )
        graph_obj = graph
    graph = normalize_to_simple_directed(
        graph_obj,
        connectivity_mode=cfg.connectivity_mode,
        keep_largest_component=cfg.keep_largest_component,
        drop_isolates=cfg.drop_isolates,
    )
    if cfg.connectivity_mode == "weak" and not nx.is_weakly_connected(graph):
        raise ValueError(f"{path} is not weakly connected after normalization")
    if cfg.connectivity_mode == "strong" and not nx.is_strongly_connected(graph):
        raise ValueError(f"{path} is not strongly connected after normalization")
    return graph


def _intersection_count(left: set[int], right: set[int]) -> int:
    if len(left) > len(right):
        left, right = right, left
    return sum(1 for node in left if node in right)


def compute_directed_triplets(
    out_adj: Sequence[set[int]], in_adj: Sequence[set[int]]
) -> int:
    total = 0
    for node in range(len(out_adj)):
        total += len(in_adj[node]) * len(out_adj[node]) - _intersection_count(
            in_adj[node], out_adj[node]
        )
    return int(total)


def compute_closed_directed_triplets(
    out_adj: Sequence[set[int]], in_adj: Sequence[set[int]]
) -> int:
    total = 0
    for middle in range(len(out_adj)):
        in_middle = in_adj[middle]
        if not in_middle:
            continue
        for target in out_adj[middle]:
            total += _intersection_count(in_middle, in_adj[target])
    return int(total)


@dataclass
class GraphState:
    n: int
    out_adj: list[set[int]]
    in_adj: list[set[int]]
    edges: list[tuple[int, int]]
    edge_positions: dict[tuple[int, int], int]
    out_degrees: np.ndarray
    in_degrees: np.ndarray
    m: int
    closed_triplets: int
    triplets: int
    original_edge_set: frozenset[tuple[int, int]]
    diff_edges: set[tuple[int, int]]
    node_labels: tuple[object, ...]
    node_attrs: tuple[dict[str, object], ...]
    original_edge_attrs: dict[tuple[int, int], dict[str, object]]
    degree_gini_mode: str
    _degree_gini_cache: float | None = None

    @classmethod
    def from_networkx(
        cls,
        graph: nx.DiGraph,
        *,
        degree_gini_mode: str = "out",
        original_edge_set: frozenset[tuple[int, int]] | None = None,
    ) -> "GraphState":
        n = graph.number_of_nodes()
        out_adj = [set() for _ in range(n)]
        in_adj = [set() for _ in range(n)]
        edges: list[tuple[int, int]] = []
        edge_positions: dict[tuple[int, int], int] = {}
        out_degrees = np.zeros(n, dtype=np.int64)
        in_degrees = np.zeros(n, dtype=np.int64)

        for u, v in graph.edges():
            edge = directed_edge(int(u), int(v))
            edge_positions[edge] = len(edges)
            edges.append(edge)
            out_adj[edge[0]].add(edge[1])
            in_adj[edge[1]].add(edge[0])
            out_degrees[edge[0]] += 1
            in_degrees[edge[1]] += 1

        triplets = compute_directed_triplets(out_adj, in_adj)
        closed_triplets = compute_closed_directed_triplets(out_adj, in_adj)
        original = frozenset(edges) if original_edge_set is None else original_edge_set
        diff_edges = set(set(edges).symmetric_difference(original))
        node_labels = tuple(graph.nodes[i].get("original_label", i) for i in range(n))
        node_attrs = tuple(dict(graph.nodes[i]) for i in range(n))
        original_edge_attrs = {
            directed_edge(int(u), int(v)): dict(data)
            for u, v, data in graph.edges(data=True)
        }

        state = cls(
            n=n,
            out_adj=out_adj,
            in_adj=in_adj,
            edges=edges,
            edge_positions=edge_positions,
            out_degrees=out_degrees,
            in_degrees=in_degrees,
            m=len(edges),
            closed_triplets=closed_triplets,
            triplets=triplets,
            original_edge_set=original,
            diff_edges=diff_edges,
            node_labels=node_labels,
            node_attrs=node_attrs,
            original_edge_attrs=original_edge_attrs,
            degree_gini_mode=degree_gini_mode,
        )
        state._degree_gini_cache = degree_gini_from_degrees(state.degree_vector())
        return state

    def clone(self) -> "GraphState":
        return GraphState(
            n=self.n,
            out_adj=[set(neighbors) for neighbors in self.out_adj],
            in_adj=[set(neighbors) for neighbors in self.in_adj],
            edges=list(self.edges),
            edge_positions=dict(self.edge_positions),
            out_degrees=self.out_degrees.copy(),
            in_degrees=self.in_degrees.copy(),
            m=self.m,
            closed_triplets=self.closed_triplets,
            triplets=self.triplets,
            original_edge_set=self.original_edge_set,
            diff_edges=set(self.diff_edges),
            node_labels=self.node_labels,
            node_attrs=tuple(dict(attrs) for attrs in self.node_attrs),
            original_edge_attrs={
                edge: dict(attrs) for edge, attrs in self.original_edge_attrs.items()
            },
            degree_gini_mode=self.degree_gini_mode,
            _degree_gini_cache=self._degree_gini_cache,
        )

    def degree_vector(self) -> np.ndarray:
        if self.degree_gini_mode == "out":
            return self.out_degrees
        if self.degree_gini_mode == "in":
            return self.in_degrees
        raise ValueError(f"Unsupported degree_gini_mode: {self.degree_gini_mode}")

    def has_edge(self, u: int, v: int) -> bool:
        return directed_edge(u, v) in self.edge_positions

    def shortcut_count(self, u: int, v: int) -> int:
        return _intersection_count(self.out_adj[u], self.in_adj[v])

    def in_common_count(self, u: int, v: int) -> int:
        return _intersection_count(self.in_adj[u], self.in_adj[v])

    def out_common_count(self, u: int, v: int) -> int:
        return _intersection_count(self.out_adj[u], self.out_adj[v])

    def edge_closure_contribution(self, u: int, v: int) -> int:
        return self.shortcut_count(u, v) + self.in_common_count(u, v) + self.out_common_count(u, v)

    def _add_edge_index(self, edge: tuple[int, int]) -> None:
        self.edge_positions[edge] = len(self.edges)
        self.edges.append(edge)

    def _remove_edge_index(self, edge: tuple[int, int]) -> None:
        idx = self.edge_positions.pop(edge)
        last = self.edges.pop()
        if idx < len(self.edges):
            self.edges[idx] = last
            self.edge_positions[last] = idx

    def _toggle_diff_edge(self, edge: tuple[int, int]) -> None:
        if edge in self.diff_edges:
            self.diff_edges.remove(edge)
        else:
            self.diff_edges.add(edge)

    def add_edge(self, u: int, v: int) -> None:
        edge = directed_edge(u, v)
        if edge in self.edge_positions:
            raise ValueError(f"edge already exists: {edge}")

        reciprocal = int(u in self.out_adj[v])
        delta_triplets = len(self.in_adj[u]) + len(self.out_adj[v]) - 2 * reciprocal
        delta_closed = (
            self.shortcut_count(u, v)
            + self.in_common_count(u, v)
            + self.out_common_count(u, v)
        )

        self.triplets += delta_triplets
        self.closed_triplets += delta_closed
        self.out_adj[u].add(v)
        self.in_adj[v].add(u)
        self.out_degrees[u] += 1
        self.in_degrees[v] += 1
        self._add_edge_index(edge)
        self.m += 1
        self._toggle_diff_edge(edge)
        self._degree_gini_cache = None

    def delete_edge(self, u: int, v: int) -> None:
        edge = directed_edge(u, v)
        if edge not in self.edge_positions:
            raise ValueError(f"edge does not exist: {edge}")

        reciprocal = int(u in self.out_adj[v])
        delta_triplets = len(self.in_adj[u]) + len(self.out_adj[v]) - 2 * reciprocal
        delta_closed = (
            self.shortcut_count(u, v)
            + self.in_common_count(u, v)
            + self.out_common_count(u, v)
        )

        self.triplets -= delta_triplets
        self.closed_triplets -= delta_closed
        self.out_adj[u].remove(v)
        self.in_adj[v].remove(u)
        self.out_degrees[u] -= 1
        self.in_degrees[v] -= 1
        self._remove_edge_index(edge)
        self.m -= 1
        self._toggle_diff_edge(edge)
        self._degree_gini_cache = None

    def degree_gini(self) -> float:
        if self._degree_gini_cache is None:
            self._degree_gini_cache = degree_gini_from_degrees(self.degree_vector())
        return float(self._degree_gini_cache)

    def density(self) -> float:
        denom = self.n * (self.n - 1)
        return safe_divide(self.m, denom)

    def clustering(self) -> float:
        return safe_divide(self.closed_triplets, self.triplets)

    def edit_distance(self) -> int:
        return len(self.diff_edges)

    def stats(self) -> dict[str, float]:
        return {
            "density": self.density(),
            "degree_gini": self.degree_gini(),
            "clustering": self.clustering(),
        }

    def adjacency_matrix(
        self,
        *,
        matrix_format: str = "csr",
        dtype=np.uint8,
    ):
        """Return the current directed adjacency matrix in integer node order."""

        if matrix_format not in VALID_STATE_MATRIX_FORMATS:
            raise ValueError(
                f"matrix_format must be one of {tuple(sorted(VALID_STATE_MATRIX_FORMATS))}"
            )
        if matrix_format == "dense":
            matrix = np.zeros((self.n, self.n), dtype=dtype)
            if self.edges:
                rows, cols = zip(*self.edges)
                matrix[np.asarray(rows), np.asarray(cols)] = 1
            return matrix

        if self.edges:
            rows, cols = zip(*self.edges)
            row_index = np.asarray(rows, dtype=np.int64)
            col_index = np.asarray(cols, dtype=np.int64)
        else:
            row_index = np.asarray([], dtype=np.int64)
            col_index = np.asarray([], dtype=np.int64)
        data = np.ones(len(row_index), dtype=dtype)
        return sp.csr_matrix(
            (data, (row_index, col_index)),
            shape=(self.n, self.n),
            dtype=dtype,
        )

    def is_weakly_connected(self) -> bool:
        if self.n <= 1:
            return True
        visited = [False] * self.n
        visited[0] = True
        seen = 1
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in self.out_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    seen += 1
                    if seen == self.n:
                        return True
                    stack.append(neighbor)
            for neighbor in self.in_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    seen += 1
                    if seen == self.n:
                        return True
                    stack.append(neighbor)
        return False

    def is_strongly_connected(self) -> bool:
        if self.n <= 1:
            return True

        def reaches_all(forward: bool) -> bool:
            visited = [False] * self.n
            visited[0] = True
            seen = 1
            stack = [0]
            while stack:
                node = stack.pop()
                neighbors = self.out_adj[node] if forward else self.in_adj[node]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        seen += 1
                        if seen == self.n:
                            return True
                        stack.append(neighbor)
            return False

        return reaches_all(True) and reaches_all(False)

    def connectivity_ok(self, mode: str | None) -> bool:
        if mode is None:
            return True
        if mode == "weak":
            return self.is_weakly_connected()
        if mode == "strong":
            return self.is_strongly_connected()
        raise ValueError(f"Unknown connectivity_mode: {mode}")

    def to_networkx(self, *, restore_labels: bool = True) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in range(self.n):
            label = self.node_labels[node] if restore_labels else node
            attrs = dict(self.node_attrs[node]) if node < len(self.node_attrs) else {}
            attrs["original_label"] = self.node_labels[node]
            graph.add_node(label, **attrs)
        for u, v in self.edges:
            left = self.node_labels[u] if restore_labels else u
            right = self.node_labels[v] if restore_labels else v
            edge = directed_edge(u, v)
            attrs = dict(self.original_edge_attrs.get(edge, {}))
            attrs["was_original_edge"] = edge in self.original_edge_set
            attrs["is_added_edge"] = edge not in self.original_edge_set
            graph.add_edge(left, right, **attrs)
        return graph


def move_is_legal(state: GraphState, move: Move) -> bool:
    remove_set = set(move.remove)
    add_set = set(move.add)
    if remove_set & add_set:
        return False
    if len(remove_set) != len(move.remove) or len(add_set) != len(move.add):
        return False

    for edge in remove_set:
        if edge not in state.edge_positions:
            return False

    for edge in add_set:
        if edge[0] == edge[1]:
            return False
        if edge in state.edge_positions and edge not in remove_set:
            return False

    return True


def apply_move(state: GraphState, move: Move) -> None:
    for edge in move.remove:
        state.delete_edge(*edge)
    for edge in move.add:
        state.add_edge(*edge)


def undo_move(state: GraphState, move: Move) -> None:
    for edge in reversed(move.add):
        state.delete_edge(*edge)
    for edge in reversed(move.remove):
        state.add_edge(*edge)


def hard_violation_penalty(state: GraphState, cfg: SAConfig) -> float:
    if not state.connectivity_ok(cfg.connectivity_mode):
        return cfg.hard_constraint_penalty
    return 0.0


def move_may_break_connectivity(move: Move, cfg: SAConfig) -> bool:
    return cfg.connectivity_mode is not None and bool(move.remove)


def is_feasible(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    *,
    assume_hard_ok: bool = False,
) -> bool:
    current = state.stats()
    if abs(current[target_name] - target_value) > cfg.target_tolerances[target_name]:
        return False
    for stat_name in STAT_NAMES:
        if stat_name == target_name:
            continue
        if (
            abs(current[stat_name] - baseline_stats[stat_name])
            > cfg.preserve_tolerances[stat_name]
        ):
            return False
    if assume_hard_ok:
        return True
    return hard_violation_penalty(state, cfg) == 0.0


def _scaled_error(error: float, tolerance: float) -> float:
    return float(error) / max(float(tolerance), 1e-9)


def _squared(value: float) -> float:
    return float(value) * float(value)


def _outside_band_error(error: float, tolerance: float) -> float:
    return max(0.0, float(error) - float(tolerance))


def _outside_band_scaled_error(error: float, tolerance: float) -> float:
    return _scaled_error(_outside_band_error(error, tolerance), tolerance)


def _scaled_loss(error: float, tolerance: float) -> float:
    return _squared(_scaled_error(error, tolerance))


def _outside_band_scaled_loss(error: float, tolerance: float) -> float:
    return _squared(_outside_band_scaled_error(error, tolerance))


def _in_band_scaled_loss(
    error: float,
    tolerance: float,
    cfg: SAConfig,
    *,
    target: bool,
) -> float:
    if cfg.band_loss_mode(target=target) == "zero":
        return _outside_band_scaled_loss(error, tolerance)
    return _scaled_loss(error, tolerance)


def _in_band_raw_error(
    error: float,
    tolerance: float,
    cfg: SAConfig,
    *,
    target: bool,
) -> float:
    if cfg.band_loss_mode(target=target) == "zero":
        return _outside_band_error(error, tolerance)
    return float(error)


def objective_loss_parts(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
) -> dict[str, float]:
    current = state.stats()
    target_error = abs(current[target_name] - target_value)
    target_loss = _in_band_scaled_loss(
        target_error,
        cfg.target_tolerances[target_name],
        cfg,
        target=True,
    )

    preserve_violation = 0.0
    preserve_center = 0.0
    preserve_raw_total = 0.0
    max_preserve_scaled_error = 0.0
    for stat_name in STAT_NAMES:
        if stat_name == target_name:
            continue
        error = abs(current[stat_name] - baseline_stats[stat_name])
        tolerance = cfg.preserve_tolerances[stat_name]
        scaled_error = _scaled_error(error, tolerance)
        preserve_violation += _outside_band_scaled_loss(error, tolerance)
        preserve_center += _squared(scaled_error)
        preserve_raw_total += error
        max_preserve_scaled_error = max(max_preserve_scaled_error, scaled_error)

    edit_distance = float(state.edit_distance())
    edit_loss = safe_divide(edit_distance, max(1, len(state.original_edge_set)))
    return {
        "target_error": float(target_error),
        "target_loss": float(target_loss),
        "preserve_violation": float(preserve_violation),
        "preserve_center": float(preserve_center),
        "preserve_raw_total": float(preserve_raw_total),
        "max_preserve_scaled_error": float(max_preserve_scaled_error),
        "edit_distance": float(edit_distance),
        "edit_loss": float(edit_loss),
    }


def attainment_energy(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
) -> float:
    parts = objective_loss_parts(state, baseline_stats, target_name, target_value, cfg)
    return (
        cfg.attain_target_weight * parts["target_loss"]
        + cfg.attain_preserve_weight
        * (parts["preserve_violation"] + parts["preserve_center"])
        + cfg.attain_distance_weight * parts["edit_loss"]
    )


def compression_energy(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
) -> float:
    parts = objective_loss_parts(state, baseline_stats, target_name, target_value, cfg)
    return (
        cfg.compress_target_weight * parts["target_loss"]
        + cfg.compress_preserve_weight * parts["preserve_center"]
        + cfg.compress_distance_weight * parts["edit_loss"]
    )


def feasible_lexicographic_key(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
) -> tuple[float, ...]:
    parts = objective_loss_parts(state, baseline_stats, target_name, target_value, cfg)
    current = state.stats()
    target_raw_error = _in_band_raw_error(
        abs(current[target_name] - target_value),
        cfg.target_tolerances[target_name],
        cfg,
        target=True,
    )
    if cfg.feasible_selection_mode == "preserve_first":
        return (
            float(parts["preserve_center"]),
            float(parts["edit_loss"]),
            float(parts["edit_distance"]),
            float(parts["target_loss"]),
            float(parts["preserve_violation"]),
        )
    if cfg.feasible_preserve_weight > 0.0:
        score = (
            cfg.feasible_preserve_weight * parts["preserve_center"]
            + cfg.compress_distance_weight * parts["edit_loss"]
        )
        return (
            float(score),
            float(parts["preserve_center"]),
            float(parts["edit_loss"]),
            float(parts["edit_distance"]),
            float(parts["target_loss"]),
            float(parts["preserve_violation"]),
        )
    return (
        float(parts["edit_distance"]),
        float(target_raw_error),
        float(parts["preserve_center"]),
    )


def _sample_edge(
    state: GraphState, rng: np.random.Generator
) -> tuple[int, int] | None:
    if not state.edges:
        return None
    idx = int(rng.integers(len(state.edges)))
    return state.edges[idx]


def _sample_nonedge(
    state: GraphState,
    rng: np.random.Generator,
    max_trials: int,
    forbidden_sources: set[int] | None = None,
    forbidden_targets: set[int] | None = None,
) -> tuple[int, int] | None:
    forbidden_sources = forbidden_sources or set()
    forbidden_targets = forbidden_targets or set()
    for _ in range(max_trials):
        u = int(rng.integers(state.n))
        v = int(rng.integers(state.n))
        if u == v or u in forbidden_sources or v in forbidden_targets:
            continue
        edge = directed_edge(u, v)
        if edge not in state.edge_positions:
            return edge
    return None


def _sample_extreme_node(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    high: bool,
    must_be_positive: bool,
    exclude: set[int] | None,
    batch_size: int,
) -> int | None:
    exclude = exclude or set()
    candidates: list[int] = []
    for _ in range(batch_size * 3):
        node = int(rng.integers(len(values)))
        if node in exclude:
            continue
        if must_be_positive and values[node] <= 0:
            continue
        candidates.append(node)
        if len(candidates) >= batch_size:
            break
    if not candidates:
        return None
    key = lambda node: int(values[node])
    return max(candidates, key=key) if high else min(candidates, key=key)


def _sample_high_contribution_edge(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> tuple[int, int] | None:
    best_edge = None
    best_score = -1
    for _ in range(cfg.node_selection_batch):
        edge = _sample_edge(state, rng)
        if edge is None:
            break
        score = state.edge_closure_contribution(*edge)
        if score > best_score:
            best_score = score
            best_edge = edge
    return best_edge


def sample_random_swap(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> Move | None:
    for _ in range(cfg.candidate_trials):
        first = _sample_edge(state, rng)
        second = _sample_edge(state, rng)
        if first is None or second is None or first == second:
            continue
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4:
            continue
        move = Move(
            kind="double_edge_swap",
            remove=(first, second),
            add=(directed_edge(a, d), directed_edge(c, b)),
        )
        if move_is_legal(state, move):
            return move
    return None


def sample_targeted_clustering_swap(
    state: GraphState,
    target_value: float,
    rng: np.random.Generator,
    cfg: SAConfig,
) -> Move | None:
    increase = target_value >= state.clustering()

    if increase:
        for _ in range(cfg.candidate_trials):
            middle = int(rng.integers(state.n))
            if not state.in_adj[middle] or not state.out_adj[middle]:
                continue
            source = tuple(state.in_adj[middle])[int(rng.integers(len(state.in_adj[middle])))]
            target = tuple(state.out_adj[middle])[int(rng.integers(len(state.out_adj[middle])))]
            if source == target or state.has_edge(source, target):
                continue

            source_out = tuple(
                node
                for node in state.out_adj[source]
                if node not in {middle, target}
            )
            target_in = tuple(
                node
                for node in state.in_adj[target]
                if node not in {source, middle}
            )
            if not source_out or not target_in:
                continue

            for _inner in range(8):
                spill = source_out[int(rng.integers(len(source_out)))]
                donor = target_in[int(rng.integers(len(target_in)))]
                if len({source, middle, target, spill, donor}) < 5:
                    continue
                move = Move(
                    kind="double_edge_swap",
                    remove=(directed_edge(source, spill), directed_edge(donor, target)),
                    add=(directed_edge(source, target), directed_edge(donor, spill)),
                )
                if move_is_legal(state, move):
                    return move

    high_edge = _sample_high_contribution_edge(state, rng, cfg)
    if high_edge is None:
        return sample_random_swap(state, rng, cfg)

    for _ in range(cfg.candidate_trials):
        other = _sample_edge(state, rng)
        if other is None or other == high_edge:
            continue
        a, b = high_edge
        c, d = other
        if len({a, b, c, d}) < 4:
            continue
        move = Move(
            kind="double_edge_swap",
            remove=(high_edge, other),
            add=(directed_edge(a, d), directed_edge(c, b)),
        )
        if move_is_legal(state, move):
            return move

    return sample_random_swap(state, rng, cfg)


def sample_random_rewire(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> Move | None:
    for _ in range(cfg.candidate_trials):
        edge = _sample_edge(state, rng)
        if edge is None:
            return None
        u, v = edge
        if state.degree_gini_mode == "out":
            new_source = int(rng.integers(state.n))
            if new_source in {u, v}:
                continue
            move = Move(
                kind="source_rewire",
                remove=(edge,),
                add=(directed_edge(new_source, v),),
            )
        else:
            new_target = int(rng.integers(state.n))
            if new_target in {u, v}:
                continue
            move = Move(
                kind="target_rewire",
                remove=(edge,),
                add=(directed_edge(u, new_target),),
            )
        if move_is_legal(state, move):
            return move
    return None


def sample_targeted_degree_gini_rewire(
    state: GraphState,
    target_value: float,
    rng: np.random.Generator,
    cfg: SAConfig,
) -> Move | None:
    current = state.degree_gini()
    increase = target_value >= current

    if state.degree_gini_mode == "out":
        for _ in range(cfg.candidate_trials):
            moving = _sample_extreme_node(
                state.out_degrees,
                rng,
                high=not increase,
                must_be_positive=True,
                exclude=None,
                batch_size=cfg.node_selection_batch,
            )
            new_source = _sample_extreme_node(
                state.out_degrees,
                rng,
                high=increase,
                must_be_positive=False,
                exclude={moving} if moving is not None else None,
                batch_size=cfg.node_selection_batch,
            )
            if moving is None or new_source is None or moving == new_source:
                continue
            targets = tuple(state.out_adj[moving])
            if not targets:
                continue
            fixed_target = targets[int(rng.integers(len(targets)))]
            if new_source == fixed_target:
                continue
            move = Move(
                kind="source_rewire",
                remove=(directed_edge(moving, fixed_target),),
                add=(directed_edge(new_source, fixed_target),),
            )
            if move_is_legal(state, move):
                return move
    else:
        for _ in range(cfg.candidate_trials):
            moving = _sample_extreme_node(
                state.in_degrees,
                rng,
                high=not increase,
                must_be_positive=True,
                exclude=None,
                batch_size=cfg.node_selection_batch,
            )
            new_target = _sample_extreme_node(
                state.in_degrees,
                rng,
                high=increase,
                must_be_positive=False,
                exclude={moving} if moving is not None else None,
                batch_size=cfg.node_selection_batch,
            )
            if moving is None or new_target is None or moving == new_target:
                continue
            sources = tuple(state.in_adj[moving])
            if not sources:
                continue
            fixed_source = sources[int(rng.integers(len(sources)))]
            if fixed_source == new_target:
                continue
            move = Move(
                kind="target_rewire",
                remove=(directed_edge(fixed_source, moving),),
                add=(directed_edge(fixed_source, new_target),),
            )
            if move_is_legal(state, move):
                return move

    return sample_random_rewire(state, rng, cfg)


def sample_density_repair_move(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> Move | None:
    added_edges = [
        edge
        for edge in state.diff_edges
        if edge not in state.original_edge_set and edge in state.edge_positions
    ]
    missing_original = [
        edge
        for edge in state.diff_edges
        if edge in state.original_edge_set and edge not in state.edge_positions
    ]
    if not added_edges or not missing_original:
        return None

    for _ in range(cfg.candidate_trials):
        remove_edge = added_edges[int(rng.integers(len(added_edges)))]
        add_edge = missing_original[int(rng.integers(len(missing_original)))]
        move = Move(
            kind="density_repair",
            remove=(remove_edge,),
            add=(add_edge,),
        )
        if move_is_legal(state, move):
            return move
    return None


def _diff_edge_parts(
    state: GraphState,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    added_edges = tuple(
        edge
        for edge in state.diff_edges
        if edge not in state.original_edge_set and edge in state.edge_positions
    )
    missing_original = tuple(
        edge
        for edge in state.diff_edges
        if edge in state.original_edge_set and edge not in state.edge_positions
    )
    return added_edges, missing_original


def sample_edge_repair_move(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> Move | None:
    added_edges, missing_original = _diff_edge_parts(state)
    if not added_edges or not missing_original:
        return None

    for _ in range(cfg.candidate_trials):
        remove_edge = added_edges[int(rng.integers(len(added_edges)))]
        add_edge = missing_original[int(rng.integers(len(missing_original)))]
        move = Move(kind="edge_repair", remove=(remove_edge,), add=(add_edge,))
        if move_is_legal(state, move):
            return move
    return None


def sample_degree_preserving_repair_move(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> Move | None:
    added_edges, missing_original = _diff_edge_parts(state)
    if len(added_edges) < 2 or len(missing_original) < 2:
        return None
    missing_set = set(missing_original)

    for _ in range(cfg.candidate_trials):
        first = added_edges[int(rng.integers(len(added_edges)))]
        second = added_edges[int(rng.integers(len(added_edges)))]
        if first == second:
            continue
        try:
            add_first = directed_edge(first[0], second[1])
            add_second = directed_edge(second[0], first[1])
        except ValueError:
            continue
        if add_first not in missing_set or add_second not in missing_set:
            continue
        move = Move(
            kind="degree_preserving_repair",
            remove=(first, second),
            add=(add_first, add_second),
        )
        if move_is_legal(state, move):
            return move
    return None


def sample_compression_repair_move(
    state: GraphState,
    target_name: str,
    rng: np.random.Generator,
    cfg: SAConfig,
) -> Move | None:
    if target_name == "clustering":
        repair = sample_degree_preserving_repair_move(state, rng, cfg)
        if repair is not None:
            return repair
    return sample_edge_repair_move(state, rng, cfg)


def sample_random_density_move(
    state: GraphState,
    target_value: float,
    rng: np.random.Generator,
    cfg: SAConfig,
) -> Move | None:
    prefer_add = target_value >= state.density()
    if prefer_add:
        edge = _sample_nonedge(state, rng, cfg.candidate_trials)
        if edge is not None:
            return Move(kind="add_edge", add=(edge,))
        edge = _sample_edge(state, rng)
        return None if edge is None else Move(kind="delete_edge", remove=(edge,))

    edge = _sample_edge(state, rng)
    if edge is not None:
        return Move(kind="delete_edge", remove=(edge,))
    edge = _sample_nonedge(state, rng, cfg.candidate_trials)
    return None if edge is None else Move(kind="add_edge", add=(edge,))


def sample_targeted_density_move(
    state: GraphState,
    target_value: float,
    rng: np.random.Generator,
    cfg: SAConfig,
) -> Move | None:
    if rng.random() < cfg.density_repair_swap_prob:
        repair = sample_density_repair_move(state, rng, cfg)
        if repair is not None:
            return repair
    return sample_random_density_move(state, target_value, rng, cfg)


def _sampler_for_target(target_name: str, *, targeted: bool):
    if target_name == "clustering":
        return sample_targeted_clustering_swap if targeted else sample_random_swap
    if target_name == "degree_gini":
        return sample_targeted_degree_gini_rewire if targeted else sample_random_rewire
    if target_name == "density":
        return sample_targeted_density_move if targeted else sample_random_density_move
    raise ValueError(f"Unknown target statistic: {target_name}")


def _sample_move_once(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    rng: np.random.Generator,
    *,
    targeted: bool,
) -> Move | None:
    sampler = _sampler_for_target(target_name, targeted=targeted)
    if target_name == "clustering":
        return sampler(state, target_value, rng, cfg) if targeted else sampler(state, rng, cfg)
    if target_name == "degree_gini":
        return sampler(state, target_value, rng, cfg) if targeted else sampler(state, rng, cfg)
    if target_name == "density":
        return sampler(state, target_value, rng, cfg)
    raise ValueError(f"Unknown target statistic: {target_name}")


def sample_local_move_stat_changes(
    state0: GraphState,
    baseline_stats: dict[str, float],
    cfg: SAConfig,
    *,
    n_samples_per_target: int = 512,
    seed: int | None = None,
    include_targeted: bool = True,
    include_random: bool = True,
    require_hard_ok: bool = True,
    max_attempts_multiplier: int = 50,
) -> pd.DataFrame:
    """Sample one-move statistic changes around the baseline state.

    This is a cheap empirical scale diagnostic. It samples the same local move
    kernels used by annealing, applies each sampled move to a fresh clone of the
    baseline, records the one-move statistic deltas, and undoes the move.
    """

    if n_samples_per_target < 1:
        raise ValueError("n_samples_per_target must be >= 1")
    if not include_targeted and not include_random:
        raise ValueError("At least one of include_targeted/include_random must be True")
    if max_attempts_multiplier < 1:
        raise ValueError("max_attempts_multiplier must be >= 1")

    rng = np.random.default_rng(cfg.random_seed if seed is None else seed)
    rows: list[dict[str, float | int | str | bool]] = []

    for target_name in cfg.target_stats:
        collected = 0
        attempts = 0
        max_attempts = n_samples_per_target * max_attempts_multiplier
        state = state0.clone()

        while collected < n_samples_per_target and attempts < max_attempts:
            attempts += 1
            if include_targeted and include_random:
                targeted = bool(rng.random() < cfg.targeted_move_prob)
            else:
                targeted = bool(include_targeted)

            if targeted:
                target_value = float(rng.choice(np.array([0.0, 1.0], dtype=float)))
            else:
                target_value = float(baseline_stats[target_name])

            move = _sample_move_once(
                state,
                baseline_stats,
                target_name,
                target_value,
                cfg,
                rng,
                targeted=targeted,
            )
            if move is None or not move_is_legal(state, move):
                continue

            apply_move(state, move)
            hard_ok = hard_violation_penalty(state, cfg) == 0.0
            if require_hard_ok and not hard_ok:
                undo_move(state, move)
                continue

            current = state.stats()
            row: dict[str, float | int | str | bool] = {
                "target_name": target_name,
                "targeted": targeted,
                "target_value_for_sampler": target_value,
                "move_kind": move.kind,
                "hard_ok": hard_ok,
                "edit_delta": int(state.edit_distance()),
            }
            for stat_name in STAT_NAMES:
                delta = float(current[stat_name] - baseline_stats[stat_name])
                row[f"delta_{stat_name}"] = delta
                row[f"abs_delta_{stat_name}"] = abs(delta)
            rows.append(row)
            collected += 1
            undo_move(state, move)

    return pd.DataFrame(rows)


def calibrate_tolerances_from_local_moves(
    state0: GraphState,
    baseline_stats: dict[str, float],
    cfg: SAConfig,
    *,
    n_samples_per_target: int = 512,
    quantile: float = 0.9,
    multiplier: float = 3.0,
    floor_tolerances: dict[str, float] | None = None,
    ceiling_tolerances: dict[str, float] | None = None,
    apply_to: tuple[str, ...] = ("target", "preserve"),
    seed: int | None = None,
    include_targeted: bool = True,
    include_random: bool = True,
    require_hard_ok: bool = True,
) -> tuple[SAConfig, pd.DataFrame, pd.DataFrame]:
    """Return a copy of cfg with tolerances calibrated from local move noise.

    The recommended tolerance for each statistic is:

        multiplier * quantile(positive one-move absolute deltas)

    Existing tolerances are used as floors unless floor_tolerances overrides
    them. This avoids accidentally tightening a tolerance to zero for stats
    that many move kernels preserve exactly.
    """

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be > 0")
    valid_apply_to = {"target", "preserve"}
    unknown = set(apply_to) - valid_apply_to
    if unknown:
        raise ValueError(f"apply_to contains unknown entries: {tuple(sorted(unknown))}")

    samples_df = sample_local_move_stat_changes(
        state0,
        baseline_stats,
        cfg,
        n_samples_per_target=n_samples_per_target,
        seed=seed,
        include_targeted=include_targeted,
        include_random=include_random,
        require_hard_ok=require_hard_ok,
    )

    floor_tolerances = floor_tolerances or {}
    ceiling_tolerances = ceiling_tolerances or {}
    calibrated: dict[str, float] = {}
    diagnostics: list[dict[str, float | int | str]] = []

    for stat_name in STAT_NAMES:
        values = pd.to_numeric(
            samples_df.get(f"abs_delta_{stat_name}", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        positive = values.loc[values > 0.0]

        current_floor = max(
            float(cfg.target_tolerances.get(stat_name, 0.0)),
            float(cfg.preserve_tolerances.get(stat_name, 0.0)),
        )
        floor = float(floor_tolerances.get(stat_name, current_floor))

        if positive.empty:
            local_quantile = math.nan
            recommended = floor
        else:
            local_quantile = float(positive.quantile(quantile))
            recommended = max(floor, multiplier * local_quantile)

        if stat_name in ceiling_tolerances:
            recommended = min(recommended, float(ceiling_tolerances[stat_name]))

        calibrated[stat_name] = float(recommended)
        diagnostics.append(
            {
                "stat_name": stat_name,
                "n_samples": int(values.size),
                "n_positive": int(positive.size),
                "median_abs_delta_positive": (
                    math.nan if positive.empty else float(positive.median())
                ),
                f"q{quantile:g}_abs_delta_positive": local_quantile,
                "multiplier": float(multiplier),
                "floor_tolerance": floor,
                "recommended_tolerance": float(recommended),
                "current_target_tolerance": float(cfg.target_tolerances[stat_name]),
                "current_preserve_tolerance": float(cfg.preserve_tolerances[stat_name]),
            }
        )

    target_tolerances = dict(cfg.target_tolerances)
    preserve_tolerances = dict(cfg.preserve_tolerances)
    if "target" in apply_to:
        target_tolerances.update(calibrated)
    if "preserve" in apply_to:
        preserve_tolerances.update(calibrated)

    calibrated_cfg = replace(
        cfg,
        target_tolerances=target_tolerances,
        preserve_tolerances=preserve_tolerances,
    )
    return calibrated_cfg, pd.DataFrame(diagnostics), samples_df


def propose_move(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    rng: np.random.Generator,
    *,
    phase: str,
) -> Move | None:
    choose_targeted = rng.random() < cfg.targeted_move_prob
    strategies = [choose_targeted, not choose_targeted]
    candidates: list[tuple[float, Move]] = []
    seen: set[tuple[str, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = set()

    def consider_move(move: Move | None) -> None:
        if move is None:
            return
        signature = move.signature()
        if signature in seen or not move_is_legal(state, move):
            return
        seen.add(signature)

        apply_move(state, move)
        hard_penalty = (
            hard_violation_penalty(state, cfg)
            if move_may_break_connectivity(move, cfg)
            else 0.0
        )
        if phase == "compress":
            if hard_penalty > 0.0 or not is_feasible(
                state,
                baseline_stats,
                target_name,
                target_value,
                cfg,
                assume_hard_ok=True,
            ):
                score = math.inf
            else:
                score = compression_energy(
                    state, baseline_stats, target_name, target_value, cfg
                )
        else:
            if hard_penalty > 0.0:
                score = cfg.hard_constraint_penalty + hard_penalty
            else:
                score = attainment_energy(
                    state, baseline_stats, target_name, target_value, cfg
                )
        undo_move(state, move)

        if math.isfinite(score):
            candidates.append((score, move))

    if phase == "compress":
        for _ in range(max(cfg.best_of_k, 1)):
            consider_move(
                sample_compression_repair_move(state, target_name, rng, cfg)
            )

    for targeted in strategies:
        for _ in range(cfg.best_of_k):
            move = _sample_move_once(
                state,
                baseline_stats,
                target_name,
                target_value,
                cfg,
                rng,
                targeted=targeted,
            )
            consider_move(move)

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def estimate_initial_temperature(
    state0: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    rng: np.random.Generator,
    *,
    phase: str,
) -> float:
    state = state0.clone()
    if phase == "compress":
        base_energy = compression_energy(state, baseline_stats, target_name, target_value, cfg)
    else:
        base_energy = attainment_energy(state, baseline_stats, target_name, target_value, cfg)

    positive_deltas: list[float] = []
    for _ in range(cfg.pilot_temperature_samples):
        move = _sample_move_once(
            state,
            baseline_stats,
            target_name,
            target_value,
            cfg,
            rng,
            targeted=False,
        )
        if move is None or not move_is_legal(state, move):
            continue
        apply_move(state, move)
        hard_penalty = (
            hard_violation_penalty(state, cfg)
            if move_may_break_connectivity(move, cfg)
            else 0.0
        )
        if hard_penalty > 0.0:
            undo_move(state, move)
            continue
        if phase == "compress":
            if not is_feasible(
                state,
                baseline_stats,
                target_name,
                target_value,
                cfg,
                assume_hard_ok=True,
            ):
                undo_move(state, move)
                continue
            new_energy = compression_energy(
                state, baseline_stats, target_name, target_value, cfg
            )
        else:
            new_energy = attainment_energy(
                state, baseline_stats, target_name, target_value, cfg
            )
        delta = new_energy - base_energy
        if delta > 0.0:
            positive_deltas.append(float(delta))
        undo_move(state, move)

    if not positive_deltas:
        return 1.0
    mean_delta = float(np.mean(positive_deltas))
    acceptance = min(max(cfg.initial_acceptance_rate, 0.05), 0.95)
    return max(1e-6, -mean_delta / math.log(acceptance))


def _flatten_counter(counter: Counter, prefix: str) -> dict[str, int]:
    payload = {f"{prefix}_{kind}": int(counter.get(kind, 0)) for kind in MOVE_KINDS}
    payload[f"{prefix}_total"] = int(sum(counter.values()))
    return payload


def _trace_row(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    *,
    phase: str,
    accepted_step: int,
    proposal_step: int,
    temperature: float,
    energy: float,
    move_kind: str,
    best_feasible: GraphState | None = None,
    best_energy: float | None = None,
) -> dict[str, float | int | str | bool]:
    current = state.stats()
    preserve_total = sum(
        abs(current[stat_name] - baseline_stats[stat_name])
        for stat_name in STAT_NAMES
        if stat_name != target_name
    )
    row = {
        "phase": phase,
        "accepted_step": int(accepted_step),
        "proposal_step": int(proposal_step),
        "temperature": float(temperature),
        "energy": float(energy),
        "best_energy_so_far": float("nan") if best_energy is None else float(best_energy),
        "move_kind": move_kind,
        "target_name": target_name,
        "target_value": float(target_value),
        "density": float(current["density"]),
        "degree_gini": float(current["degree_gini"]),
        "clustering": float(current["clustering"]),
        "edit_distance": int(state.edit_distance()),
        "target_error": float(abs(current[target_name] - target_value)),
        "preserve_error_total": float(preserve_total),
        "feasible": bool(is_feasible(state, baseline_stats, target_name, target_value, cfg)),
    }
    if best_feasible is None:
        row.update(
            {
                "best_feasible_density": float("nan"),
                "best_feasible_degree_gini": float("nan"),
                "best_feasible_clustering": float("nan"),
                "best_feasible_edit_distance": float("nan"),
                "best_feasible_target_error": float("nan"),
                "best_feasible_preserve_error_total": float("nan"),
            }
        )
    else:
        best_stats = best_feasible.stats()
        best_preserve_total = sum(
            abs(best_stats[stat_name] - baseline_stats[stat_name])
            for stat_name in STAT_NAMES
            if stat_name != target_name
        )
        row.update(
            {
                "best_feasible_density": float(best_stats["density"]),
                "best_feasible_degree_gini": float(best_stats["degree_gini"]),
                "best_feasible_clustering": float(best_stats["clustering"]),
                "best_feasible_edit_distance": int(best_feasible.edit_distance()),
                "best_feasible_target_error": float(abs(best_stats[target_name] - target_value)),
                "best_feasible_preserve_error_total": float(best_preserve_total),
            }
        )
    return row


def _state_matrix_row(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    *,
    phase: str,
    accepted_step: int,
    proposal_step: int,
    energy: float,
    move_kind: str,
) -> dict[str, object]:
    current = state.stats()
    preserve_total = sum(
        abs(current[stat_name] - baseline_stats[stat_name])
        for stat_name in STAT_NAMES
        if stat_name != target_name
    )
    return {
        "phase": phase,
        "accepted_step": int(accepted_step),
        "proposal_step": int(proposal_step),
        "energy": float(energy),
        "move_kind": move_kind,
        "target_name": target_name,
        "target_value": float(target_value),
        "density": float(current["density"]),
        "degree_gini": float(current["degree_gini"]),
        "clustering": float(current["clustering"]),
        "edit_distance": int(state.edit_distance()),
        "target_error": float(abs(current[target_name] - target_value)),
        "preserve_error_total": float(preserve_total),
        "feasible": bool(is_feasible(state, baseline_stats, target_name, target_value, cfg)),
        "matrix_format": cfg.state_matrix_format,
        "adjacency_matrix": state.adjacency_matrix(
            matrix_format=cfg.state_matrix_format
        ),
    }


def _scalar_meta_items(meta: dict[str, object]) -> dict[str, object]:
    scalar_types = (bool, int, float, str, np.integer, np.floating)
    return {
        key: value
        for key, value in meta.items()
        if isinstance(value, scalar_types)
    }


def _anneal_impl(
    state0: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    *,
    seed: int,
    phase: str,
    total_steps: int,
) -> tuple[GraphState | None, dict[str, float]]:
    rng = np.random.default_rng(seed)
    state = state0.clone()
    initial_temperature = estimate_initial_temperature(
        state,
        baseline_stats,
        target_name,
        target_value,
        cfg,
        rng,
        phase=phase,
    )
    temperature = initial_temperature

    if phase == "compress":
        if not is_feasible(state, baseline_stats, target_name, target_value, cfg):
            return None, {"phase": phase, "seed": seed, "found_feasible": 0}
        current_energy = compression_energy(state, baseline_stats, target_name, target_value, cfg)
    else:
        current_energy = attainment_energy(state, baseline_stats, target_name, target_value, cfg)

    best_energy = current_energy
    best_overall = state.clone()
    best_feasible: GraphState | None = (
        state.clone() if is_feasible(state, baseline_stats, target_name, target_value, cfg) else None
    )
    best_feasible_key = (
        feasible_lexicographic_key(
            best_feasible,
            baseline_stats,
            target_name,
            target_value,
            cfg,
        )
        if best_feasible is not None
        else None
    )
    first_feasible_accepted_step = 0 if best_feasible is not None else None
    best_feasible_improved_at_accepted = 0 if best_feasible is not None else None
    stop_reason = "step_budget"

    proposal_counts: Counter = Counter()
    accepted_counts: Counter = Counter()
    accepted = 0
    proposed = 0
    blocks_without_improvement = 0
    trace_rows: list[dict[str, float | int | str | bool]] = []
    state_matrix_rows: list[dict[str, object]] = []
    no_feasible_proposal_limit = (
        cfg.no_feasible_proposal_limit(target_name) if phase == "attain" else 0
    )

    if cfg.record_acceptance_trace:
        trace_rows.append(
            _trace_row(
                state,
                baseline_stats,
                target_name,
                target_value,
                cfg,
                phase=phase,
                accepted_step=0,
                proposal_step=0,
                temperature=temperature,
                energy=current_energy,
                move_kind="initial",
                best_feasible=best_feasible,
                best_energy=best_energy,
            )
        )

    if cfg.record_state_matrices:
        state_matrix_rows.append(
            _state_matrix_row(
                state,
                baseline_stats,
                target_name,
                target_value,
                cfg,
                phase=phase,
                accepted_step=0,
                proposal_step=0,
                energy=current_energy,
                move_kind="initial",
            )
        )

    if (
        phase == "attain"
        and cfg.stop_attain_on_first_feasible
        and best_feasible is not None
    ):
        stop_reason = "initial_feasible"
        total_steps = 0

    started = time.perf_counter()
    n_blocks = max(1, math.ceil(total_steps / cfg.temperature_block_size))
    stop_requested = total_steps <= 0

    for block_index in range(n_blocks):
        if stop_requested:
            break
        improved = False
        steps_this_block = min(
            cfg.temperature_block_size,
            max(0, total_steps - block_index * cfg.temperature_block_size),
        )
        if steps_this_block <= 0:
            break

        for _ in range(steps_this_block):
            move = propose_move(
                state,
                baseline_stats,
                target_name,
                target_value,
                cfg,
                rng,
                phase=phase,
            )
            if move is None:
                continue

            proposed += 1
            proposal_counts[move.kind] += 1
            apply_move(state, move)
            hard_penalty = (
                hard_violation_penalty(state, cfg)
                if move_may_break_connectivity(move, cfg)
                else 0.0
            )
            if hard_penalty > 0.0:
                undo_move(state, move)
                continue

            if phase == "compress":
                if not is_feasible(
                    state,
                    baseline_stats,
                    target_name,
                    target_value,
                    cfg,
                    assume_hard_ok=True,
                ):
                    undo_move(state, move)
                    continue
                next_energy = compression_energy(
                    state, baseline_stats, target_name, target_value, cfg
                )
            else:
                next_energy = attainment_energy(
                    state, baseline_stats, target_name, target_value, cfg
                )

            delta = next_energy - current_energy
            accept = delta <= 0.0 or rng.random() < math.exp(
                -delta / max(temperature, 1e-9)
            )

            if accept:
                current_energy = next_energy
                accepted += 1
                accepted_counts[move.kind] += 1

                if next_energy < best_energy:
                    best_energy = next_energy
                    best_overall = state.clone()
                    improved = True

                state_feasible = is_feasible(
                    state,
                    baseline_stats,
                    target_name,
                    target_value,
                    cfg,
                    assume_hard_ok=True,
                )
                if state_feasible:
                    candidate_key = feasible_lexicographic_key(
                        state,
                        baseline_stats,
                        target_name,
                        target_value,
                        cfg,
                    )
                    if first_feasible_accepted_step is None:
                        first_feasible_accepted_step = accepted
                    if best_feasible is None or candidate_key < best_feasible_key:
                        best_feasible = state.clone()
                        best_feasible_key = candidate_key
                        best_feasible_improved_at_accepted = accepted
                        improved = True

                if cfg.record_acceptance_trace and accepted % cfg.trace_accept_stride == 0:
                    trace_rows.append(
                        _trace_row(
                            state,
                            baseline_stats,
                            target_name,
                            target_value,
                            cfg,
                            phase=phase,
                            accepted_step=accepted,
                            proposal_step=proposed,
                            temperature=temperature,
                            energy=current_energy,
                            move_kind=move.kind,
                            best_feasible=best_feasible,
                            best_energy=best_energy,
                        )
                    )
                if cfg.record_state_matrices and accepted % cfg.state_matrix_stride == 0:
                    state_matrix_rows.append(
                        _state_matrix_row(
                            state,
                            baseline_stats,
                            target_name,
                            target_value,
                            cfg,
                            phase=phase,
                            accepted_step=accepted,
                            proposal_step=proposed,
                            energy=current_energy,
                            move_kind=move.kind,
                        )
                    )
                if (
                    phase == "attain"
                    and cfg.stop_attain_on_first_feasible
                    and best_feasible is not None
                ):
                    stop_reason = "first_feasible"
                    stop_requested = True
                    break
                if (
                    phase == "attain"
                    and no_feasible_proposal_limit > 0
                    and best_feasible is None
                    and proposed >= no_feasible_proposal_limit
                ):
                    stop_reason = "attain_no_feasible_proposal_limit"
                    stop_requested = True
                    break
                if (
                    phase == "compress"
                    and cfg.compress_no_improvement_patience > 0
                    and best_feasible_improved_at_accepted is not None
                    and accepted - best_feasible_improved_at_accepted
                    >= cfg.compress_no_improvement_patience
                ):
                    stop_reason = "compress_no_improvement_patience"
                    stop_requested = True
                    break
            else:
                undo_move(state, move)
                if (
                    phase == "attain"
                    and no_feasible_proposal_limit > 0
                    and best_feasible is None
                    and proposed >= no_feasible_proposal_limit
                ):
                    stop_reason = "attain_no_feasible_proposal_limit"
                    stop_requested = True
                    break

        if stop_requested:
            break

        if improved:
            blocks_without_improvement = 0
        else:
            blocks_without_improvement += 1

        temperature *= cfg.cooling_alpha
        if blocks_without_improvement >= cfg.stall_blocks:
            stop_reason = "stall_blocks"
            break

    compression_start_edit_distance = math.nan
    compression_uncapped_best_feasible_edit_distance = math.nan
    compression_edit_cap_applied = False
    if phase == "compress":
        compression_start_edit_distance = int(state0.edit_distance())
        if best_feasible is not None:
            compression_uncapped_best_feasible_edit_distance = int(
                best_feasible.edit_distance()
            )
            # Compression should never make the returned graph farther away
            # than the feasible state it was asked to compress.
            if best_feasible.edit_distance() > compression_start_edit_distance:
                best_feasible = state0.clone()
                best_feasible_key = feasible_lexicographic_key(
                    best_feasible,
                    baseline_stats,
                    target_name,
                    target_value,
                    cfg,
                )
                best_feasible_improved_at_accepted = 0
                compression_edit_cap_applied = True

    runtime = time.perf_counter() - started
    best_overall_stats = best_overall.stats()
    meta = {
        "phase": phase,
        "seed": seed,
        "runtime_seconds": runtime,
        "initial_temperature": initial_temperature,
        "final_temperature": temperature,
        "best_energy": best_energy,
        "current_energy": current_energy,
        "n_proposals": proposed,
        "n_accepted": accepted,
        "acceptance_rate": safe_divide(accepted, proposed),
        "found_feasible": int(best_feasible is not None),
        "stop_reason": stop_reason,
        "first_feasible_accepted_step": (
            math.nan
            if first_feasible_accepted_step is None
            else int(first_feasible_accepted_step)
        ),
        "best_feasible_edit_distance": (
            math.nan if best_feasible is None else int(best_feasible.edit_distance())
        ),
        "best_overall_density": best_overall_stats["density"],
        "best_overall_degree_gini": best_overall_stats["degree_gini"],
        "best_overall_clustering": best_overall_stats["clustering"],
        "best_overall_edit_distance": best_overall.edit_distance(),
        "degree_gini_mode": state.degree_gini_mode,
        "connectivity_mode": cfg.connectivity_mode if cfg.connectivity_mode is not None else "none",
        **_flatten_counter(proposal_counts, "proposed"),
        **_flatten_counter(accepted_counts, "accepted"),
    }
    if phase == "compress":
        meta.update(
            {
                "start_edit_distance": compression_start_edit_distance,
                "uncapped_best_feasible_edit_distance": (
                    compression_uncapped_best_feasible_edit_distance
                ),
                "edit_cap_applied": int(compression_edit_cap_applied),
            }
        )
    if cfg.record_acceptance_trace:
        meta["acceptance_trace"] = pd.DataFrame(trace_rows)
    if cfg.record_state_matrices:
        meta["state_matrix_trace"] = pd.DataFrame(state_matrix_rows)

    return best_feasible, meta


def anneal_attain(
    state0: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    *,
    seed: int,
) -> tuple[GraphState | None, dict[str, float]]:
    return _anneal_impl(
        state0,
        baseline_stats,
        target_name,
        target_value,
        cfg,
        seed=seed,
        phase="attain",
        total_steps=cfg.attain_steps,
    )


def anneal_compress(
    state0: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
    *,
    seed: int,
) -> tuple[GraphState | None, dict[str, float]]:
    return _anneal_impl(
        state0,
        baseline_stats,
        target_name,
        target_value,
        cfg,
        seed=seed,
        phase="compress",
        total_steps=cfg.compress_steps,
    )


def estimate_reachable_grid(
    state0: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    cfg: SAConfig,
    *,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    if cfg.target_grid_mode == "explicit":
        baseline_value = baseline_stats[target_name]
        lower_raw, upper_raw = cfg.explicit_grid_bounds[target_name]
        lower = max(0.0, min(float(lower_raw), float(upper_raw)))
        upper = min(1.0, max(float(lower_raw), float(upper_raw)))
        if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
            grid = np.array([lower], dtype=float)
        else:
            grid = np.linspace(lower, upper, cfg.n_target_grid, dtype=float)
        meta = {
            "target_name": target_name,
            "baseline_value": baseline_value,
            "pilot_lower": lower,
            "pilot_upper": upper,
            "pilot_samples": 0,
            "grid_mode": "explicit",
        }
        return grid, meta

    if cfg.target_grid_mode == "centered":
        baseline_value = baseline_stats[target_name]
        half_width = float(cfg.centered_grid_half_widths[target_name])
        lower = max(0.0, baseline_value - half_width)
        upper = min(1.0, baseline_value + half_width)
        if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
            grid = np.array([baseline_value], dtype=float)
        else:
            grid = np.linspace(lower, upper, cfg.n_target_grid, dtype=float)
        meta = {
            "target_name": target_name,
            "baseline_value": baseline_value,
            "pilot_lower": lower,
            "pilot_upper": upper,
            "pilot_samples": 0,
            "grid_mode": "centered",
        }
        return grid, meta

    achieved: list[float] = [baseline_stats[target_name]]
    pilot_cfg = replace(cfg, attain_steps=cfg.pilot_steps, compress_steps=0)

    for direction, extreme_value in (("decrease", 0.0), ("increase", 1.0)):
        for run_idx in range(cfg.pilot_runs_per_direction):
            seed = cfg.random_seed + seed_offset + (10_000 * (direction == "increase")) + run_idx
            best_state, meta = anneal_attain(
                state0,
                baseline_stats,
                target_name,
                extreme_value,
                pilot_cfg,
                seed=seed,
            )
            if best_state is not None:
                achieved.append(best_state.stats()[target_name])
            else:
                achieved.append(float(meta[f"best_overall_{target_name}"]))

    lower = float(np.quantile(achieved, cfg.grid_quantiles[0]))
    upper = float(np.quantile(achieved, cfg.grid_quantiles[1]))
    lower = min(lower, baseline_stats[target_name])
    upper = max(upper, baseline_stats[target_name])
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-9):
        grid = np.array([baseline_stats[target_name]], dtype=float)
    else:
        grid = np.linspace(lower, upper, cfg.n_target_grid, dtype=float)

    meta = {
        "target_name": target_name,
        "baseline_value": baseline_stats[target_name],
        "pilot_lower": lower,
        "pilot_upper": upper,
        "pilot_samples": len(achieved),
        "grid_mode": "pilot",
    }
    return grid, meta


def estimate_all_target_grids(
    state0: GraphState,
    baseline_stats: dict[str, float],
    cfg: SAConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for target_index, target_name in enumerate(cfg.target_stats):
        grid, meta = estimate_reachable_grid(
            state0,
            baseline_stats,
            target_name,
            cfg,
            seed_offset=target_index * 10_000,
        )
        for grid_index, target_value in enumerate(grid):
            rows.append(
                {
                    "target_stat": target_name,
                    "grid_index": grid_index,
                    "target_value": float(target_value),
                    "degree_gini_mode": cfg.degree_gini_mode,
                    "connectivity_mode": cfg.connectivity_mode if cfg.connectivity_mode is not None else "none",
                    **meta,
                }
            )
    return pd.DataFrame(rows)


def _seed_for_run(
    cfg: SAConfig, graph_index: int, target_index: int, grid_index: int, seed_index: int
) -> int:
    return int(
        cfg.random_seed
        + graph_index * 10_000_000
        + target_index * 1_000_000
        + grid_index * 10_000
        + seed_index
    )


def save_graph_state(
    state: GraphState,
    path: str | Path,
    *,
    restore_labels: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = state.to_networkx(restore_labels=restore_labels)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)
    return path


def _state_matrix_suffix(matrix_format: str) -> str:
    if matrix_format == "csr":
        return ".npz"
    if matrix_format == "dense":
        return ".npy"
    raise ValueError(
        f"matrix_format must be one of {tuple(sorted(VALID_STATE_MATRIX_FORMATS))}"
    )


def save_state_matrix(
    state: GraphState,
    path: str | Path,
    *,
    matrix_format: str = "csr",
) -> Path:
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(_state_matrix_suffix(matrix_format))
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = state.adjacency_matrix(matrix_format=matrix_format)
    if matrix_format == "csr":
        sp.save_npz(path, matrix)
    elif matrix_format == "dense":
        np.save(path, matrix)
    else:
        raise ValueError(
            f"matrix_format must be one of {tuple(sorted(VALID_STATE_MATRIX_FORMATS))}"
        )
    return path


def load_state_matrix(path: str | Path):
    path = Path(path)
    if path.suffix == ".npz":
        return sp.load_npz(path)
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported matrix file suffix: {path.suffix}")


def matrix_feature_stack(matrices: Sequence[object]):
    matrices = list(matrices)
    if not matrices:
        return sp.csr_matrix((0, 0), dtype=np.uint8)
    if any(sp.issparse(matrix) for matrix in matrices):
        rows = [
            matrix.reshape(1, -1)
            if sp.issparse(matrix)
            else sp.csr_matrix(np.asarray(matrix).reshape(1, -1))
            for matrix in matrices
        ]
        return sp.vstack(rows, format="csr")
    return np.vstack([np.asarray(matrix).reshape(1, -1) for matrix in matrices])


def load_matrix_feature_stack(paths: Sequence[str | Path]):
    return matrix_feature_stack(load_state_matrix(path) for path in paths)


def trace_df_from_meta(meta: dict[str, object] | None) -> pd.DataFrame:
    if meta is None or "acceptance_trace" not in meta:
        return pd.DataFrame()
    trace_df = meta["acceptance_trace"]
    if isinstance(trace_df, pd.DataFrame):
        return trace_df.copy()
    return pd.DataFrame(trace_df)


def combine_phase_traces(*metas: dict[str, object] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    accepted_offset = 0
    proposal_offset = 0
    for meta in metas:
        frame = trace_df_from_meta(meta)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["accepted_step_global"] = frame["accepted_step"] + accepted_offset
        frame["proposal_step_global"] = frame["proposal_step"] + proposal_offset
        accepted_offset = int(frame["accepted_step_global"].max())
        proposal_offset = int(frame["proposal_step_global"].max())
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def state_matrix_trace_df_from_meta(meta: dict[str, object] | None) -> pd.DataFrame:
    if meta is None or "state_matrix_trace" not in meta:
        return pd.DataFrame()
    trace_df = meta["state_matrix_trace"]
    if isinstance(trace_df, pd.DataFrame):
        return trace_df.copy()
    return pd.DataFrame(trace_df)


def combine_phase_matrix_traces(*metas: dict[str, object] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    accepted_offset = 0
    proposal_offset = 0
    for meta in metas:
        frame = state_matrix_trace_df_from_meta(meta)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["accepted_step_global"] = frame["accepted_step"] + accepted_offset
        frame["proposal_step_global"] = frame["proposal_step"] + proposal_offset
        accepted_offset = int(frame["accepted_step_global"].max())
        proposal_offset = int(frame["proposal_step_global"].max())
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_trace_config(
    cfg: SAConfig,
    *,
    state_matrix_stride: int | None = None,
    state_matrix_format: str | None = None,
    record_state_matrices: bool = True,
) -> SAConfig:
    """Return a config that follows the same SA settings while recording traces."""
    return replace(
        cfg,
        record_acceptance_trace=True,
        trace_accept_stride=1,
        record_state_matrices=record_state_matrices,
        state_matrix_stride=(
            cfg.state_matrix_stride if state_matrix_stride is None else state_matrix_stride
        ),
        state_matrix_format=state_matrix_format or cfg.state_matrix_format,
        show_progress=False,
    )


def _bundle_state_and_stats(bundle: object) -> tuple[GraphState, dict[str, float]]:
    if isinstance(bundle, GraphState):
        return bundle, bundle.stats()
    if isinstance(bundle, Mapping):
        state = bundle["state"]
        if not isinstance(state, GraphState):
            raise TypeError("source_states bundles must contain a GraphState under 'state'")
        baseline_stats = bundle.get("baseline_stats")
        if baseline_stats is None:
            baseline_stats = state.stats()
        return state, dict(baseline_stats)
    raise TypeError("source_states values must be GraphState objects or mapping bundles")


def _source_id_for_result_row(
    row: pd.Series,
    source_states: Mapping[str, object],
) -> str:
    source_id_value = row.get("source_id", None)
    if source_id_value is None or pd.isna(source_id_value):
        if len(source_states) != 1:
            raise KeyError(
                "Result row has no source_id and source_states contains multiple sources"
            )
        return str(next(iter(source_states)))
    return str(source_id_value)


def replay_result_trace(
    row: pd.Series,
    source_states: Mapping[str, object],
    cfg: SAConfig,
    *,
    run_index: int = 0,
    state_matrix_stride: int | None = None,
    state_matrix_format: str | None = None,
    record_state_matrices: bool = True,
) -> dict[str, object]:
    """Replay one batch result row with trace recording enabled.

    The sampled run identity comes from the batch row: source, target statistic,
    requested target value, and seed. Trace recording does not change the move
    sequence.
    """
    trace_cfg = make_trace_config(
        cfg,
        state_matrix_stride=state_matrix_stride,
        state_matrix_format=state_matrix_format,
        record_state_matrices=record_state_matrices,
    )
    source_id = _source_id_for_result_row(row, source_states)
    baseline_state, baseline_stats = _bundle_state_and_stats(source_states[source_id])
    target_name = str(row["target_stat"])
    target_value = float(row["target_value_requested"])
    seed = int(row["seed"])
    result_index = row.name

    attained_state, attain_meta = anneal_attain(
        baseline_state,
        baseline_stats,
        target_name,
        target_value,
        trace_cfg,
        seed=seed,
    )
    if attained_state is not None:
        final_state, compress_meta = anneal_compress(
            attained_state,
            baseline_stats,
            target_name,
            target_value,
            trace_cfg,
            seed=seed + 10_000_000,
        )
    else:
        final_state, compress_meta = None, None

    trace_df = combine_phase_traces(attain_meta, compress_meta)
    matrix_trace_df = combine_phase_matrix_traces(attain_meta, compress_meta)
    chosen_state = final_state or attained_state

    for frame in (trace_df, matrix_trace_df):
        if not frame.empty:
            frame["inspection_run_index"] = run_index
            frame["result_index"] = result_index
            frame["source_id"] = source_id
            frame["replay_seed"] = seed

    replay_stats = (
        {name: math.nan for name in STAT_NAMES}
        if chosen_state is None
        else chosen_state.stats()
    )
    replay_feasible = False if chosen_state is None else is_feasible(
        chosen_state,
        baseline_stats,
        target_name,
        target_value,
        trace_cfg,
    )

    summary_row = {
        "inspection_run_index": run_index,
        "result_index": result_index,
        "source_id": source_id,
        "target_stat": target_name,
        "target_value_requested": target_value,
        "seed": seed,
        "batch_feasible": bool(row.get("feasible", False)),
        "replay_feasible": replay_feasible,
        "batch_edit_distance": row.get("edit_distance", math.nan),
        "replay_edit_distance": math.nan if chosen_state is None else chosen_state.edit_distance(),
        "batch_achieved_density": row.get("achieved_density", math.nan),
        "replay_achieved_density": replay_stats["density"],
        "batch_achieved_degree_gini": row.get("achieved_degree_gini", math.nan),
        "replay_achieved_degree_gini": replay_stats["degree_gini"],
        "batch_achieved_clustering": row.get("achieved_clustering", math.nan),
        "replay_achieved_clustering": replay_stats["clustering"],
        "trace_points": int(len(trace_df)),
        "matrix_trace_points": int(len(matrix_trace_df)),
        "attain_stop_reason": attain_meta.get("stop_reason"),
        "attain_first_feasible_step": attain_meta.get("first_feasible_accepted_step"),
        "attain_best_feasible_edit_distance": attain_meta.get("best_feasible_edit_distance"),
        "compress_stop_reason": None if compress_meta is None else compress_meta.get("stop_reason"),
        "compress_best_feasible_edit_distance": (
            None if compress_meta is None else compress_meta.get("best_feasible_edit_distance")
        ),
        "compress_start_edit_distance": (
            None if compress_meta is None else compress_meta.get("start_edit_distance")
        ),
        "compress_uncapped_best_feasible_edit_distance": (
            None
            if compress_meta is None
            else compress_meta.get("uncapped_best_feasible_edit_distance")
        ),
        "compress_edit_cap_applied": (
            None if compress_meta is None else compress_meta.get("edit_cap_applied")
        ),
        "attain_acceptance_rate": attain_meta.get("acceptance_rate"),
        "compress_acceptance_rate": (
            None if compress_meta is None else compress_meta.get("acceptance_rate")
        ),
    }

    return {
        "inspection_run_index": run_index,
        "result_index": result_index,
        "source_id": source_id,
        "target_stat": target_name,
        "target_value": target_value,
        "seed": seed,
        "baseline_stats": baseline_stats,
        "attained_state": attained_state,
        "final_state": final_state,
        "chosen_state": chosen_state,
        "attain_meta": attain_meta,
        "compress_meta": compress_meta,
        "trace_df": trace_df,
        "matrix_trace_df": matrix_trace_df,
        "row": summary_row,
    }


def replay_random_result_traces(
    results_df: pd.DataFrame,
    source_states: Mapping[str, object],
    cfg: SAConfig,
    *,
    n_runs: int = 3,
    random_state: int | None = None,
    feasible_only: bool = False,
    state_matrix_stride: int | None = None,
    state_matrix_format: str | None = None,
    record_state_matrices: bool = True,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    """Sample real batch rows and replay them with trace recording."""
    candidates = results_df.copy()
    if feasible_only:
        candidates = candidates.loc[candidates["feasible"].fillna(False)].copy()
    if candidates.empty:
        raise ValueError("No result rows are available for trace replay")

    selected_rows = candidates.sample(
        n=min(int(n_runs), len(candidates)),
        random_state=random_state,
    ).copy()
    runs = [
        replay_result_trace(
            row,
            source_states,
            cfg,
            run_index=run_index,
            state_matrix_stride=state_matrix_stride,
            state_matrix_format=state_matrix_format,
            record_state_matrices=record_state_matrices,
        )
        for run_index, (_, row) in enumerate(selected_rows.iterrows())
    ]
    return runs, selected_rows, pd.DataFrame([run["row"] for run in runs])


def trace_y_ranges_for_runs(
    grid_df: pd.DataFrame,
    runs: Sequence[Mapping[str, object]],
    cfg: SAConfig,
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for stat_name in cfg.target_stats:
        values: list[float] = []
        for run in runs:
            baseline_stats = run.get("baseline_stats", {})
            if stat_name in baseline_stats:
                values.append(float(baseline_stats[stat_name]))

        target_rows = grid_df.loc[grid_df["target_stat"] == stat_name]
        for column in ("pilot_lower", "pilot_upper", "target_value", "baseline_value"):
            if column in target_rows.columns:
                values.extend(target_rows[column].dropna().astype(float).tolist())

        for run in runs:
            trace_df = run["trace_df"]
            for column in (stat_name, f"best_feasible_{stat_name}"):
                if column in trace_df.columns:
                    values.extend(trace_df[column].dropna().astype(float).tolist())

            if run["target_stat"] == stat_name:
                tolerance = float(cfg.target_tolerances.get(stat_name, 0.0))
                target_value = float(run["target_value"])
                values.extend([target_value - tolerance, target_value + tolerance])
            else:
                baseline_stats = run.get("baseline_stats", {})
                if stat_name in baseline_stats:
                    tolerance = float(cfg.preserve_tolerances.get(stat_name, 0.0))
                    baseline_value = float(baseline_stats[stat_name])
                    values.extend([baseline_value - tolerance, baseline_value + tolerance])

        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) == 0:
            continue

        lower = max(0.0, float(finite_values.min()))
        upper = min(1.0, float(finite_values.max()))
        if not math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
            padding = 0.02 * (upper - lower)
            lower = max(0.0, lower - padding)
            upper = min(1.0, upper + padding)
        ranges[stat_name] = (lower, upper)
    return ranges


def plot_optimization_traces(
    trace_df: pd.DataFrame,
    *,
    columns: Sequence[str] = (
        "density",
        "degree_gini",
        "clustering",
        "edit_distance",
        "energy",
    ),
    x: str = "accepted_step_global",
    title: str | None = None,
    show_best_feasible: bool = True,
    final_values: Mapping[str, float] | None = None,
    final_label: str = "returned solution",
    target_tolerances: Mapping[str, float] | None = None,
    preserve_tolerances: Mapping[str, float] | None = None,
    y_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    if trace_df.empty:
        raise ValueError("trace_df is empty")

    plot_df = trace_df.copy()
    if x not in plot_df.columns:
        plot_df[x] = np.arange(len(plot_df))

    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2.4 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]

    phases = list(dict.fromkeys(plot_df["phase"])) if "phase" in plot_df.columns else [None]
    colors = {"attain": "#1f77b4", "compress": "#ff7f0e"}
    best_color = "#2ca02c"
    final_color = "#111111"
    final_x = float(plot_df[x].max())
    baseline_reference = {
        column: float(plot_df[column].iloc[0])
        for column in ("density", "degree_gini", "clustering")
        if column in plot_df.columns and not plot_df[column].empty
    }
    target_name = None
    target_value = None
    if "target_name" in plot_df.columns and not plot_df["target_name"].dropna().empty:
        target_name = str(plot_df["target_name"].dropna().iloc[0])
    if "target_value" in plot_df.columns and not plot_df["target_value"].dropna().empty:
        target_value = float(plot_df["target_value"].dropna().iloc[0])
    phase_boundaries: list[float] = []
    if "phase" in plot_df.columns:
        for phase in phases[1:]:
            phase_df = plot_df.loc[plot_df["phase"] == phase]
            if not phase_df.empty:
                phase_boundaries.append(float(phase_df[x].iloc[0]))

    for ax, column in zip(axes, columns):
        for phase in phases:
            phase_df = plot_df if phase is None else plot_df.loc[plot_df["phase"] == phase]
            if phase_df.empty or column not in phase_df.columns:
                continue
            ax.plot(
                phase_df[x],
                phase_df[column],
                linewidth=1.6,
                alpha=0.9,
                label=phase if phase is not None else column,
                color=colors.get(phase, None),
            )
        if show_best_feasible:
            best_column = f"best_feasible_{column}"
            if column == "energy":
                best_column = "best_energy_so_far"
            if best_column in plot_df.columns and plot_df[best_column].notna().any():
                ax.plot(
                    plot_df[x],
                    plot_df[best_column],
                    color=best_color,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                    label="best feasible" if column != "energy" else "best energy",
                )
        if final_values is not None and column in final_values:
            final_value = float(final_values[column])
            if math.isfinite(final_value):
                ax.axhline(
                    final_value,
                    color=final_color,
                    linestyle=(0, (1, 2)),
                    linewidth=1.0,
                    alpha=0.75,
                )
                ax.scatter(
                    [final_x],
                    [final_value],
                    color=final_color,
                    marker="D",
                    s=34,
                    zorder=6,
                )
        if column in baseline_reference:
            ax.axhline(
                baseline_reference[column],
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
        if target_name == column and target_value is not None:
            ax.axhline(
                target_value,
                color="#d62728",
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
            )
        if column in STAT_NAMES:
            if target_name == column and target_value is not None and target_tolerances:
                tolerance = target_tolerances.get(column)
                if tolerance is not None and math.isfinite(float(tolerance)):
                    for value in (target_value - float(tolerance), target_value + float(tolerance)):
                        ax.axhline(
                            value,
                            color="#d62728",
                            linestyle="-.",
                            linewidth=0.9,
                            alpha=0.65,
                        )
            elif column in baseline_reference and preserve_tolerances:
                tolerance = preserve_tolerances.get(column)
                if tolerance is not None and math.isfinite(float(tolerance)):
                    baseline_value = baseline_reference[column]
                    for value in (
                        baseline_value - float(tolerance),
                        baseline_value + float(tolerance),
                    ):
                        ax.axhline(
                            value,
                            color="#444444",
                            linestyle="-.",
                            linewidth=0.9,
                            alpha=0.55,
                        )
        for boundary in phase_boundaries:
            ax.axvline(boundary, color="#777777", linestyle=":", linewidth=0.9, alpha=0.6)
        if y_ranges and column in y_ranges:
            lower, upper = y_ranges[column]
            lower = float(lower)
            upper = float(upper)
            if math.isfinite(lower) and math.isfinite(upper):
                if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
                    padding = max(abs(lower) * 0.05, 1e-6)
                    lower -= padding
                    upper += padding
                ax.set_ylim(lower, upper)
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(x)
    if title:
        axes[0].set_title(title)
    legend_handles: list[Line2D] = []
    for phase in phases:
        if phase is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colors.get(phase, "#1f77b4"),
                    linewidth=1.6,
                    label=phase,
                )
            )
    if show_best_feasible:
        has_best = any(
            (
                ("best_energy_so_far" if column == "energy" else f"best_feasible_{column}")
                in plot_df.columns
                and plot_df[
                    "best_energy_so_far" if column == "energy" else f"best_feasible_{column}"
                ].notna().any()
            )
            for column in columns
        )
        if has_best:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=best_color,
                    linestyle="--",
                    linewidth=1.4,
                    label="best feasible",
                )
            )
    if final_values is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=final_color,
                marker="D",
                linestyle=(0, (1, 2)),
                linewidth=1.0,
                markersize=5,
                label=final_label,
            )
        )
    if any(column in baseline_reference for column in columns):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                label="baseline",
            )
        )
    if target_name in columns and target_value is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#d62728",
                linestyle=":",
                linewidth=1.2,
                label="target",
            )
        )
    if (
        target_name in columns
        and target_value is not None
        and target_tolerances
        and target_tolerances.get(target_name) is not None
    ):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#d62728",
                linestyle="-.",
                linewidth=0.9,
                label="target tolerance",
            )
        )
    if preserve_tolerances and any(
        column in STAT_NAMES and column != target_name for column in columns
    ):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#444444",
                linestyle="-.",
                linewidth=0.9,
                label="preserve tolerance",
            )
        )
    if phase_boundaries:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#777777",
                linestyle=":",
                linewidth=0.9,
                label="phase split",
            )
        )
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    return fig


def plot_replayed_result_traces(
    runs: Sequence[Mapping[str, object]],
    grid_df: pd.DataFrame,
    cfg: SAConfig,
    *,
    columns: Sequence[str] = (
        "density",
        "degree_gini",
        "clustering",
        "edit_distance",
        "energy",
    ),
    x: str = "accepted_step_global",
) -> list[plt.Figure]:
    y_ranges = trace_y_ranges_for_runs(grid_df, runs, cfg)
    figures: list[plt.Figure] = []
    for run in runs:
        chosen_state = run.get("chosen_state")
        if isinstance(chosen_state, GraphState):
            final_values = {
                **chosen_state.stats(),
                "edit_distance": float(chosen_state.edit_distance()),
            }
        else:
            final_values = None
        summary = run.get("row", {})
        baseline_stats = run.get("baseline_stats", {})
        target_stat = str(run["target_stat"])
        target_value = float(run["target_value"])
        target_delta = target_value - float(baseline_stats.get(target_stat, math.nan))
        target_tolerance = cfg.target_tolerances.get(target_stat, math.nan)
        cap_applied = summary.get("compress_edit_cap_applied")
        title = (
            f"{run['source_id']}: row {run['result_index']} | "
            f"{target_stat} target {target_value:.4g} "
            f"(delta {target_delta:+.3g}), seed {run['seed']}\n"
            f"returned edit {summary.get('replay_edit_distance')}, "
            f"target tol {float(target_tolerance):.3g}, "
            f"attain {summary.get('attain_stop_reason')}, "
            f"compress {summary.get('compress_stop_reason')}, "
            f"edit cap {cap_applied}"
        )
        figures.append(
            plot_optimization_traces(
                run["trace_df"],
                columns=columns,
                x=x,
                title=title,
                target_tolerances=cfg.target_tolerances,
                preserve_tolerances=cfg.preserve_tolerances,
                y_ranges=y_ranges,
                final_values=final_values,
            )
        )
    return figures


def plot_trace_matrix(
    trace_map: dict[str, pd.DataFrame],
    *,
    target_order: Sequence[str] = STAT_NAMES,
    columns: Sequence[str] = STAT_NAMES,
    x: str = "accepted_step_global",
    title: str | None = None,
    show_best_feasible: bool = True,
) -> plt.Figure:
    row_targets = [target_name for target_name in target_order if target_name in trace_map]
    if not row_targets:
        raise ValueError("trace_map does not contain any requested target traces")

    fig, axes = plt.subplots(
        len(row_targets),
        len(columns),
        figsize=(4.2 * len(columns), 2.9 * len(row_targets)),
        sharex="col",
        squeeze=False,
    )
    colors = {"attain": "#1f77b4", "compress": "#ff7f0e"}
    best_color = "#2ca02c"

    for row_index, target_name in enumerate(row_targets):
        plot_df = trace_map[target_name].copy()
        if plot_df.empty:
            for col_index, column in enumerate(columns):
                ax = axes[row_index][col_index]
                ax.text(0.5, 0.5, "no trace", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            continue

        if x not in plot_df.columns:
            plot_df[x] = np.arange(len(plot_df))

        phases = list(dict.fromkeys(plot_df["phase"])) if "phase" in plot_df.columns else [None]
        baseline_reference = {
            column: float(plot_df[column].iloc[0])
            for column in columns
            if column in plot_df.columns and not plot_df[column].empty
        }
        row_target_name = target_name
        if "target_name" in plot_df.columns and not plot_df["target_name"].dropna().empty:
            row_target_name = str(plot_df["target_name"].dropna().iloc[0])
        row_target_value = None
        if "target_value" in plot_df.columns and not plot_df["target_value"].dropna().empty:
            row_target_value = float(plot_df["target_value"].dropna().iloc[0])
        phase_boundaries: list[float] = []
        if "phase" in plot_df.columns:
            for phase in phases[1:]:
                phase_df = plot_df.loc[plot_df["phase"] == phase]
                if not phase_df.empty:
                    phase_boundaries.append(float(phase_df[x].iloc[0]))

        for col_index, column in enumerate(columns):
            ax = axes[row_index][col_index]
            for phase in phases:
                phase_df = plot_df if phase is None else plot_df.loc[plot_df["phase"] == phase]
                if phase_df.empty or column not in phase_df.columns:
                    continue
                ax.plot(
                    phase_df[x],
                    phase_df[column],
                    linewidth=1.6,
                    alpha=0.9,
                    color=colors.get(phase, None),
                )
            best_column = f"best_feasible_{column}"
            if (
                show_best_feasible
                and best_column in plot_df.columns
                and plot_df[best_column].notna().any()
            ):
                ax.plot(
                    plot_df[x],
                    plot_df[best_column],
                    color=best_color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                )
            if column in baseline_reference:
                ax.axhline(
                    baseline_reference[column],
                    color="#444444",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                )
            if row_target_name == column and row_target_value is not None:
                ax.axhline(
                    row_target_value,
                    color="#d62728",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.9,
                )
            for boundary in phase_boundaries:
                ax.axvline(boundary, color="#777777", linestyle=":", linewidth=0.9, alpha=0.6)

            if row_index == 0:
                ax.set_title(column)
            if col_index == 0:
                ax.set_ylabel(target_name)
            if row_index == len(row_targets) - 1:
                ax.set_xlabel(x)
            ax.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], color=colors["attain"], linewidth=1.6, label="attain"),
        Line2D([0], [0], color=colors["compress"], linewidth=1.6, label="compress"),
        Line2D([0], [0], color=best_color, linestyle="--", linewidth=1.2, label="best feasible"),
        Line2D([0], [0], color="#444444", linestyle="--", linewidth=1.0, label="baseline"),
        Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.2, label="target"),
        Line2D([0], [0], color="#777777", linestyle=":", linewidth=0.9, label="phase split"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _record_run(
    *,
    cfg: SAConfig,
    source_id: str,
    source_path: Path,
    target_name: str,
    target_value: float,
    seed: int,
    baseline_stats: dict[str, float],
    baseline_state: GraphState,
    attained_state: GraphState | None,
    final_state: GraphState | None,
    attain_meta: dict[str, float],
    compress_meta: dict[str, float] | None,
) -> dict[str, float | int | str | bool]:
    state = final_state or attained_state
    if state is None:
        achieved_stats = {name: math.nan for name in STAT_NAMES}
        deltas = {f"delta_{name}": math.nan for name in STAT_NAMES}
        edit_distance = math.nan
        feasible = False
        target_error = math.nan
        preserve_errors = {
            f"preserve_error_{name}": math.nan
            for name in STAT_NAMES
            if name != target_name
        }
        preserve_error_scaled = {
            f"preserve_error_scaled_{name}": math.nan
            for name in STAT_NAMES
            if name != target_name
        }
        target_error_scaled = math.nan
        preserve_center = math.nan
        max_preserve_error_scaled = math.nan
    else:
        achieved_stats = state.stats()
        deltas = {
            f"delta_{name}": achieved_stats[name] - baseline_stats[name]
            for name in STAT_NAMES
        }
        edit_distance = state.edit_distance()
        feasible = is_feasible(state, baseline_stats, target_name, target_value, cfg)
        target_error = abs(achieved_stats[target_name] - target_value)
        preserve_errors = {
            f"preserve_error_{name}": abs(achieved_stats[name] - baseline_stats[name])
            for name in STAT_NAMES
            if name != target_name
        }
        preserve_error_scaled = {
            f"preserve_error_scaled_{name}": _scaled_error(
                preserve_errors[f"preserve_error_{name}"],
                cfg.preserve_tolerances[name],
            )
            for name in STAT_NAMES
            if name != target_name
        }
        target_error_scaled = _scaled_error(
            target_error,
            cfg.target_tolerances[target_name],
        )
        preserve_center = sum(
            _squared(value) for value in preserve_error_scaled.values()
        )
        max_preserve_error_scaled = max(
            preserve_error_scaled.values(),
            default=math.nan,
        )

    row: dict[str, float | int | str | bool] = {
        "source_id": source_id,
        "source_path": str(source_path),
        "n": int(baseline_state.n),
        "m0": int(len(baseline_state.original_edge_set)),
        "target_stat": target_name,
        "target_value_requested": float(target_value),
        "seed": int(seed),
        "degree_gini_mode": cfg.degree_gini_mode,
        "connectivity_mode": cfg.connectivity_mode if cfg.connectivity_mode is not None else "none",
        "in_band_loss_mode": cfg.in_band_loss_mode,
        "target_in_band_loss_mode": cfg.band_loss_mode(target=True),
        "preserve_in_band_loss_mode": cfg.band_loss_mode(target=False),
        "feasible_preserve_weight": float(cfg.feasible_preserve_weight),
        "feasible_selection_mode": cfg.feasible_selection_mode,
        "baseline_density": baseline_stats["density"],
        "baseline_degree_gini": baseline_stats["degree_gini"],
        "baseline_clustering": baseline_stats["clustering"],
        "baseline_triplets": int(baseline_state.triplets),
        "baseline_closed_triplets": int(baseline_state.closed_triplets),
        "target_value_achieved": achieved_stats.get(target_name, math.nan),
        "achieved_density": achieved_stats["density"],
        "achieved_degree_gini": achieved_stats["degree_gini"],
        "achieved_clustering": achieved_stats["clustering"],
        "achieved_triplets": math.nan if state is None else int(state.triplets),
        "achieved_closed_triplets": math.nan if state is None else int(state.closed_triplets),
        "target_error": target_error,
        "target_error_scaled": target_error_scaled,
        "preserve_center": preserve_center,
        "max_preserve_error_scaled": max_preserve_error_scaled,
        "edit_distance": edit_distance,
        "feasible": feasible,
        **deltas,
        **preserve_errors,
        **preserve_error_scaled,
    }

    row.update({f"attain_{key}": value for key, value in _scalar_meta_items(attain_meta).items()})
    if compress_meta is not None:
        row.update(
            {
                f"compress_{key}": value
                for key, value in _scalar_meta_items(compress_meta).items()
            }
        )
    attain_edit = row.get("attain_best_feasible_edit_distance", math.nan)
    compress_edit = row.get("compress_best_feasible_edit_distance", math.nan)
    if pd.notna(attain_edit) and pd.notna(compress_edit):
        improvement = float(attain_edit) - float(compress_edit)
    else:
        improvement = math.nan
    row["compression_edit_improvement"] = improvement
    row["compression_improved_edit_distance"] = (
        bool(improvement > 0.0) if pd.notna(improvement) else False
    )
    return row


def _effective_n_jobs(n_jobs: int) -> int:
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero; use 1 for serial or -1 for all cores")
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return int(n_jobs)


def _parallel_mp_context(cfg: SAConfig):
    if cfg.parallel_start_method is not None:
        return mp.get_context(cfg.parallel_start_method)
    methods = mp.get_all_start_methods()
    if os.name == "posix" and "fork" in methods:
        return mp.get_context("fork")
    return None


def _run_batch_sweep_task(
    task: dict[str, object],
) -> tuple[int, dict[str, object], GraphState | None, Path | None, tuple[str, str]]:
    task_index = int(task["task_index"])
    source_path = Path(task["source_path"])
    source_id = str(task["source_id"])
    target_name = str(task["target_name"])
    target_value = float(task["target_value"])
    grid_index = int(task["grid_index"])
    seed_index = int(task["seed_index"])
    seed = int(task["seed"])
    cfg = task["cfg"]
    baseline_state = task["baseline_state"]
    baseline_stats = task["baseline_stats"]
    if not isinstance(cfg, SAConfig):
        raise TypeError("task cfg must be an SAConfig")
    if not isinstance(baseline_state, GraphState):
        raise TypeError("task baseline_state must be a GraphState")
    if not isinstance(baseline_stats, dict):
        raise TypeError("task baseline_stats must be a dict")

    attained_state, attain_meta = anneal_attain(
        baseline_state,
        baseline_stats,
        target_name,
        target_value,
        cfg,
        seed=seed,
    )

    if attained_state is not None:
        final_state, compress_meta = anneal_compress(
            attained_state,
            baseline_stats,
            target_name,
            target_value,
            cfg,
            seed=seed + 10_000_000,
        )
    else:
        final_state, compress_meta = None, None

    row = _record_run(
        cfg=cfg,
        source_id=source_id,
        source_path=source_path,
        target_name=target_name,
        target_value=target_value,
        seed=seed,
        baseline_stats=baseline_stats,
        baseline_state=baseline_state,
        attained_state=attained_state,
        final_state=final_state,
        attain_meta=attain_meta,
        compress_meta=compress_meta,
    )
    row["grid_index"] = grid_index
    row["seed_index"] = seed_index
    row["saved_graph_path"] = ""
    row["matrix_path"] = ""

    chosen_state = final_state or attained_state
    if cfg.save_run_matrices and chosen_state is not None:
        matrix_path = (
            cfg.output_dir
            / "matrices"
            / source_id
            / target_name
            / (
                f"grid_{grid_index:03d}_seed_{seed}"
                f"{_state_matrix_suffix(cfg.state_matrix_format)}"
            )
        )
        save_state_matrix(
            chosen_state,
            matrix_path,
            matrix_format=cfg.state_matrix_format,
        )
        row["matrix_path"] = str(matrix_path)

    selected_state = (
        chosen_state
        if cfg.save_selected_graphs and row["feasible"] and chosen_state is not None
        else None
    )
    selected_path = (
        cfg.output_dir
        / "saved_graphs"
        / source_id
        / target_name
        / f"grid_{grid_index:03d}_seed_{seed}.pkl"
    )
    return task_index, row, selected_state, selected_path, (source_id, target_name)


def run_batch_sweep(
    sources: Sequence[str | Path],
    cfg: SAConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_paths = resolve_source_paths(sources, suffixes=DEFAULT_SUFFIXES)
    results: list[dict[str, float | int | str | bool]] = []
    grid_rows: list[dict[str, float | int | str]] = []
    saved_counts: Counter = Counter()
    prepared_batches: list[
        tuple[
            int,
            Path,
            GraphState,
            dict[str, float],
            list[tuple[int, str, np.ndarray]],
        ]
    ] = []
    total_runs = 0

    for graph_index, source_path in enumerate(source_paths):
        graph = load_source_graph(source_path, cfg)
        baseline_state = GraphState.from_networkx(
            graph,
            degree_gini_mode=cfg.degree_gini_mode,
        )
        baseline_stats = baseline_state.stats()
        target_batches: list[tuple[int, str, np.ndarray]] = []

        for target_index, target_name in enumerate(cfg.target_stats):
            target_grid, grid_meta = estimate_reachable_grid(
                baseline_state,
                baseline_stats,
                target_name,
                cfg,
                seed_offset=graph_index * 100_000 + target_index * 10_000,
            )
            target_batches.append((target_index, target_name, target_grid))
            total_runs += len(target_grid) * cfg.n_seeds

            for grid_index, target_value in enumerate(target_grid):
                grid_rows.append(
                    {
                        "source_id": source_path.stem,
                        "source_path": str(source_path),
                        "target_stat": target_name,
                        "grid_index": grid_index,
                        "target_value": float(target_value),
                        "degree_gini_mode": cfg.degree_gini_mode,
                        "connectivity_mode": cfg.connectivity_mode if cfg.connectivity_mode is not None else "none",
                        **grid_meta,
                    }
                )

        prepared_batches.append(
            (graph_index, source_path, baseline_state, baseline_stats, target_batches)
        )

    tasks: list[dict[str, object]] = []
    task_index = 0
    for graph_index, source_path, baseline_state, baseline_stats, target_batches in prepared_batches:
        for target_index, target_name, target_grid in target_batches:
            for grid_index, target_value in enumerate(target_grid):
                for seed_index in range(cfg.n_seeds):
                    seed = _seed_for_run(
                        cfg, graph_index, target_index, grid_index, seed_index
                    )
                    tasks.append(
                        {
                            "task_index": task_index,
                            "graph_index": graph_index,
                            "source_path": source_path,
                            "source_id": source_path.stem,
                            "baseline_state": baseline_state,
                            "baseline_stats": baseline_stats,
                            "target_index": target_index,
                            "target_name": target_name,
                            "target_value": float(target_value),
                            "grid_index": grid_index,
                            "seed_index": seed_index,
                            "seed": seed,
                            "cfg": cfg,
                        }
                    )
                    task_index += 1

    total_runs = len(tasks)
    progress = (
        tqdm(total=total_runs, desc=cfg.progress_desc)
        if cfg.show_progress and tqdm is not None
        else None
    )
    task_results: list[
        tuple[int, dict[str, object], GraphState | None, Path | None, tuple[str, str]]
    ] = []
    n_jobs = _effective_n_jobs(cfg.n_jobs)

    def update_progress(row: dict[str, object]) -> None:
        if progress is None:
            return
        progress.update(1)
        progress.set_postfix_str(
            f"{row.get('source_id', '')} | {row.get('target_stat', '')} | "
            f"grid {int(row.get('grid_index', 0)) + 1}"
        )

    try:
        if n_jobs == 1 or total_runs <= 1:
            for task in tasks:
                result = _run_batch_sweep_task(task)
                task_results.append(result)
                update_progress(result[1])
        else:
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=_parallel_mp_context(cfg),
            ) as executor:
                future_map = {
                    executor.submit(_run_batch_sweep_task, task): int(task["task_index"])
                    for task in tasks
                }
                for future in as_completed(future_map):
                    result = future.result()
                    task_results.append(result)
                    update_progress(result[1])
    finally:
        if progress is not None:
            progress.close()

    for _, row, selected_state, selected_path, save_key in sorted(
        task_results, key=lambda item: item[0]
    ):
        if (
            selected_state is not None
            and selected_path is not None
            and saved_counts[save_key] < cfg.max_saved_graphs_per_group
        ):
            save_graph_state(selected_state, selected_path)
            row["saved_graph_path"] = str(selected_path)
            saved_counts[save_key] += 1
        results.append(row)

    return pd.DataFrame(results), pd.DataFrame(grid_rows)


def summarize_compression_effectiveness(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    if "target_stat" not in results_df.columns:
        raise ValueError("results_df is missing target_stat")

    df = results_df.copy()
    for column in (
        "attain_best_feasible_edit_distance",
        "compress_best_feasible_edit_distance",
    ):
        if column not in df.columns:
            df[column] = math.nan
    df["attain_best_feasible_edit_distance"] = pd.to_numeric(
        df["attain_best_feasible_edit_distance"], errors="coerce"
    )
    df["compress_best_feasible_edit_distance"] = pd.to_numeric(
        df["compress_best_feasible_edit_distance"], errors="coerce"
    )
    df["compression_edit_improvement"] = (
        df["attain_best_feasible_edit_distance"]
        - df["compress_best_feasible_edit_distance"]
    )
    df["compression_improved_edit_distance"] = df["compression_edit_improvement"] > 0
    df["compression_ran"] = df["compress_best_feasible_edit_distance"].notna()
    df["attainment_found"] = df["attain_best_feasible_edit_distance"].notna()

    grouped = df.groupby("target_stat", dropna=False)
    summary = grouped.agg(
        runs=("target_stat", "size"),
        attainment_found=("attainment_found", "sum"),
        compression_runs=("compression_ran", "sum"),
        improved_runs=("compression_improved_edit_distance", "sum"),
        median_attain_edit_distance=("attain_best_feasible_edit_distance", "median"),
        median_compress_edit_distance=("compress_best_feasible_edit_distance", "median"),
        median_edit_improvement=("compression_edit_improvement", "median"),
        mean_edit_improvement=("compression_edit_improvement", "mean"),
        max_edit_improvement=("compression_edit_improvement", "max"),
    ).reset_index()
    summary["improvement_rate"] = summary["improved_runs"] / summary[
        "compression_runs"
    ].replace(0, np.nan)
    return summary


def summarize_preservation_drift(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize off-target movement in units of each preservation tolerance."""

    if results_df.empty:
        return pd.DataFrame()
    if "target_stat" not in results_df.columns:
        raise ValueError("results_df is missing target_stat")

    usable = results_df.loc[results_df["feasible"].fillna(False)].copy()
    rows: list[dict[str, float | int | str]] = []
    group_cols = [column for column in ("source_id", "target_stat") if column in usable]
    if not group_cols:
        group_cols = ["target_stat"]

    for group_key, group in usable.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_cols, group_key))
        target_stat = str(group_meta["target_stat"])
        for stat_name in STAT_NAMES:
            if stat_name == target_stat:
                continue
            scaled_col = f"preserve_error_scaled_{stat_name}"
            raw_col = f"preserve_error_{stat_name}"
            if scaled_col not in group.columns:
                continue
            scaled = pd.to_numeric(group[scaled_col], errors="coerce").dropna()
            if scaled.empty:
                continue
            raw = (
                pd.to_numeric(group[raw_col], errors="coerce").dropna()
                if raw_col in group.columns
                else pd.Series(dtype=float)
            )
            row: dict[str, float | int | str] = {
                **group_meta,
                "preserved_stat": stat_name,
                "n_runs": int(len(scaled)),
                "median_scaled_abs_drift": float(scaled.median()),
                "p95_scaled_abs_drift": float(scaled.quantile(0.95)),
                "max_scaled_abs_drift": float(scaled.max()),
                "within_tolerance_rate": float((scaled <= 1.0).mean()),
            }
            if not raw.empty:
                row.update(
                    {
                        "median_raw_abs_drift": float(raw.median()),
                        "p95_raw_abs_drift": float(raw.quantile(0.95)),
                        "max_raw_abs_drift": float(raw.max()),
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def compute_group_correlation_tables(
    results_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = results_df.loc[results_df["feasible"].fillna(False)].copy()
    delta_cols = ["delta_density", "delta_degree_gini", "delta_clustering"]
    pairs = [
        ("delta_density", "delta_degree_gini"),
        ("delta_density", "delta_clustering"),
        ("delta_degree_gini", "delta_clustering"),
    ]
    corr_rows: list[dict[str, float | str | int]] = []
    partial_rows: list[dict[str, float | str | int]] = []
    regression_rows: list[dict[str, float | str | int]] = []

    for (source_id, target_stat), group in usable.groupby(["source_id", "target_stat"]):
        clean = group[delta_cols + ["edit_distance"]].dropna()
        if len(clean) < 3:
            continue

        for left, right in pairs:
            corr_rows.append(
                {
                    "source_id": source_id,
                    "target_stat": target_stat,
                    "left": left,
                    "right": right,
                    "spearman_rho": safe_spearman(clean[left], clean[right]),
                    "n_obs": int(len(clean)),
                }
            )
            partial_rows.append(
                {
                    "source_id": source_id,
                    "target_stat": target_stat,
                    "left": left,
                    "right": right,
                    "partial_spearman_rho": partial_spearman(
                        clean[left], clean[right], clean["edit_distance"]
                    ),
                    "n_obs": int(len(clean)),
                }
            )

        target_delta = f"delta_{target_stat}"
        for response in delta_cols:
            if response == target_delta:
                continue
            regression_input = group[[target_delta, response, "edit_distance"]].dropna()
            regression_rows.append(
                {
                    "source_id": source_id,
                    "target_stat": target_stat,
                    "response": response,
                    **regression_with_edit_distance(
                        regression_input,
                        target_col=target_delta,
                        response_col=response,
                    ),
                }
            )

    return (
        pd.DataFrame(corr_rows),
        pd.DataFrame(partial_rows),
        pd.DataFrame(regression_rows),
    )


def aggregate_correlations(corr_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    if corr_df.empty:
        return pd.DataFrame(rows)

    for (target_stat, left, right), group in corr_df.groupby(["target_stat", "left", "right"]):
        values = group[value_col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            median = math.nan
            fisher = math.nan
            low, high = (math.nan, math.nan)
        else:
            median = float(np.median(finite))
            fisher = fisher_z_average(finite)
            low, high = bootstrap_ci(finite)
        rows.append(
            {
                "target_stat": target_stat,
                "left": left,
                "right": right,
                "n_groups": int(finite.size),
                "median_correlation": median,
                "fisher_z_average": fisher,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )

    return pd.DataFrame(rows)


def summarize_group_matrix(
    results_df: pd.DataFrame, *, source_id: str, target_stat: str
) -> pd.DataFrame:
    subset = results_df.loc[
        results_df["feasible"].fillna(False)
        & (results_df["source_id"] == source_id)
        & (results_df["target_stat"] == target_stat)
    ]
    delta_cols = ["delta_density", "delta_degree_gini", "delta_clustering"]
    if subset.empty:
        return pd.DataFrame(np.nan, index=delta_cols, columns=delta_cols)
    return safe_spearman_matrix(subset, delta_cols)
