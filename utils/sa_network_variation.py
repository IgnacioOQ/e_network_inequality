"""Simulated annealing workflow for constrained network variation.

Version 1 targets simple, undirected, unweighted graphs and the three
statistics defined in the notebook spec:

- density
- degree-gini
- global transitivity ("clustering")

The core design choices are:

1. target-specific move kernels
2. two-phase attainment/compression annealing
3. edit distance to the original graph via symmetric-difference size
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import math
import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

try:
    import dill
except ImportError:  # pragma: no cover - dill is optional at runtime
    dill = None


STAT_NAMES = ("density", "degree_gini", "clustering")
MOVE_KINDS = (
    "add_edge",
    "delete_edge",
    "endpoint_rewire",
    "double_edge_swap",
    "density_repair",
)
VALID_TARGET_GRID_MODES = {"pilot", "centered", "explicit"}
DEFAULT_SUFFIXES = (
    ".pkl",
    ".pickle",
    ".dill",
    ".edgelist",
    ".txt",
    ".csv",
    ".tsv",
    ".gml",
    ".graphml",
)


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    """Return the canonical undirected edge representation."""
    if u == v:
        raise ValueError("self-loops are not allowed in a simple graph")
    return (u, v) if u < v else (v, u)


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def degree_gini_from_degrees(degrees: np.ndarray) -> float:
    """Compute the fixed degree-gini definition from the degree vector.

    This uses the standard sorted-vector formulation, which is equivalent to
    the pairwise absolute-difference definition for non-negative values.
    """

    deg = np.asarray(degrees, dtype=float)
    total = float(deg.sum())
    if total <= 0.0:
        return 0.0
    sorted_deg = np.sort(deg)
    n = float(sorted_deg.size)
    weighted = np.dot(np.arange(1, sorted_deg.size + 1, dtype=float), sorted_deg)
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def triplets_from_degrees(degrees: np.ndarray) -> int:
    deg = np.asarray(degrees, dtype=np.int64)
    return int(np.sum((deg * (deg - 1)) // 2))


def transitivity_from_counts(triangles: int, triplets: int) -> float:
    if triplets <= 0:
        return 0.0
    return float((3.0 * triangles) / triplets)


def resolve_source_paths(
    sources: Sequence[str | Path], suffixes: Sequence[str] = DEFAULT_SUFFIXES
) -> list[Path]:
    """Expand files, directories, and glob-like patterns into concrete paths."""

    resolved: list[Path] = []
    seen: set[Path] = set()

    def register(path: Path) -> None:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            resolved.append(path)

    for item in sources:
        raw = Path(item)
        if any(token in str(item) for token in ("*", "?", "[")):
            for match in sorted(Path().glob(str(item))):
                if match.is_file():
                    register(match)
            continue

        if raw.is_dir():
            for child in sorted(raw.iterdir()):
                if child.is_file() and child.suffix.lower() in suffixes:
                    register(child)
            continue

        if raw.is_file():
            register(raw)

    return resolved


def _read_edgelist_dataframe(path: Path) -> nx.Graph:
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] < 2:
        raise ValueError(f"{path} does not contain two edge-list columns")
    source_col, target_col = df.columns[:2]
    return nx.from_pandas_edgelist(df, source=source_col, target=target_col)


def load_graph_object(path: str | Path):
    """Load a graph-like object from disk."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if suffix == ".dill":
        if dill is None:
            raise ImportError("dill is required to load .dill files")
        with path.open("rb") as handle:
            return dill.load(handle)
    if suffix in {".edgelist", ".txt"}:
        return nx.read_edgelist(path)
    if suffix in {".csv", ".tsv"}:
        return _read_edgelist_dataframe(path)
    if suffix == ".gml":
        return nx.read_gml(path)
    if suffix == ".graphml":
        return nx.read_graphml(path)
    raise ValueError(f"Unsupported graph format: {path}")


def normalize_to_simple_undirected(
    graph_obj,
    *,
    keep_largest_component: bool = True,
    drop_isolates: bool = False,
) -> nx.Graph:
    """Normalize a graph-like input to a simple undirected NetworkX graph."""

    if isinstance(graph_obj, pd.DataFrame):
        if graph_obj.shape[1] < 2:
            raise ValueError("graph DataFrame must contain at least two columns")
        source_col, target_col = graph_obj.columns[:2]
        graph = nx.from_pandas_edgelist(graph_obj, source_col, target_col)
    elif isinstance(
        graph_obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    ):
        graph = nx.Graph(graph_obj)
    else:
        raise TypeError(f"Unsupported graph object type: {type(graph_obj)!r}")

    graph.remove_edges_from(nx.selfloop_edges(graph))
    if drop_isolates:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    if graph.number_of_nodes() == 0:
        raise ValueError("normalized graph has zero nodes")

    if keep_largest_component and not nx.is_connected(graph):
        largest = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()

    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute="original_label"
    )
    return graph


def load_source_graph(path: str | Path, cfg: "SAConfig") -> nx.Graph:
    graph_obj = load_graph_object(path)
    graph = normalize_to_simple_undirected(
        graph_obj,
        keep_largest_component=cfg.keep_largest_component,
        drop_isolates=cfg.drop_isolates,
    )
    if cfg.require_connected and not nx.is_connected(graph):
        raise ValueError(f"{path} is not connected after normalization")
    return graph


@dataclass(slots=True)
class Move:
    kind: str
    remove: tuple[tuple[int, int], ...] = ()
    add: tuple[tuple[int, int], ...] = ()

    def signature(self) -> tuple[str, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        return (
            self.kind,
            tuple(sorted(self.remove)),
            tuple(sorted(self.add)),
        )


@dataclass
class SAConfig:
    target_stats: tuple[str, ...] = STAT_NAMES
    require_connected: bool = True
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
    compress_preserve_weight: float = 0.2
    hard_constraint_penalty: float = 1e6
    output_dir: Path = Path("NetworkInequality/sa_variation_outputs")
    save_selected_graphs: bool = False
    max_saved_graphs_per_group: int = 2
    show_progress: bool = True
    progress_desc: str = "SA batch runs"

    def __post_init__(self) -> None:
        if self.target_grid_mode not in VALID_TARGET_GRID_MODES:
            raise ValueError(
                f"target_grid_mode must be one of {tuple(sorted(VALID_TARGET_GRID_MODES))}"
            )

    def tolerance_scale(self, stat_name: str, *, target: bool) -> float:
        tolerance_map = self.target_tolerances if target else self.preserve_tolerances
        return max(float(tolerance_map[stat_name]), 1e-9)


@dataclass
class GraphState:
    n: int
    adj: list[set[int]]
    edges: list[tuple[int, int]]
    edge_positions: dict[tuple[int, int], int]
    degrees: np.ndarray
    m: int
    triangles: int
    triplets: int
    original_edge_set: frozenset[tuple[int, int]]
    diff_edges: set[tuple[int, int]]
    node_labels: tuple[object, ...]
    _degree_gini_cache: float | None = None

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        *,
        original_edge_set: frozenset[tuple[int, int]] | None = None,
    ) -> "GraphState":
        n = graph.number_of_nodes()
        adj = [set() for _ in range(n)]
        edges: list[tuple[int, int]] = []
        edge_positions: dict[tuple[int, int], int] = {}
        degrees = np.zeros(n, dtype=np.int64)

        for u, v in graph.edges():
            edge = canonical_edge(int(u), int(v))
            edge_positions[edge] = len(edges)
            edges.append(edge)
            adj[edge[0]].add(edge[1])
            adj[edge[1]].add(edge[0])
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1

        triangles = int(sum(nx.triangles(graph).values()) // 3)
        triplets = triplets_from_degrees(degrees)
        original = frozenset(edges) if original_edge_set is None else original_edge_set
        diff_edges = set(set(edges).symmetric_difference(original))
        node_labels = tuple(graph.nodes[i].get("original_label", i) for i in range(n))

        return cls(
            n=n,
            adj=adj,
            edges=edges,
            edge_positions=edge_positions,
            degrees=degrees,
            m=len(edges),
            triangles=triangles,
            triplets=triplets,
            original_edge_set=original,
            diff_edges=diff_edges,
            node_labels=node_labels,
            _degree_gini_cache=degree_gini_from_degrees(degrees),
        )

    def clone(self) -> "GraphState":
        return GraphState(
            n=self.n,
            adj=[set(neighbors) for neighbors in self.adj],
            edges=list(self.edges),
            edge_positions=dict(self.edge_positions),
            degrees=self.degrees.copy(),
            m=self.m,
            triangles=self.triangles,
            triplets=self.triplets,
            original_edge_set=self.original_edge_set,
            diff_edges=set(self.diff_edges),
            node_labels=self.node_labels,
            _degree_gini_cache=self._degree_gini_cache,
        )

    def has_edge(self, u: int, v: int) -> bool:
        return canonical_edge(u, v) in self.edge_positions

    def common_neighbor_count(self, u: int, v: int) -> int:
        left = self.adj[u]
        right = self.adj[v]
        if len(left) > len(right):
            left, right = right, left
        return sum(1 for node in left if node in right)

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
        edge = canonical_edge(u, v)
        if edge in self.edge_positions:
            raise ValueError(f"edge already exists: {edge}")

        common = self.common_neighbor_count(edge[0], edge[1])
        deg_u = int(self.degrees[edge[0]])
        deg_v = int(self.degrees[edge[1]])

        self.triangles += common
        self.triplets += deg_u + deg_v
        self.adj[edge[0]].add(edge[1])
        self.adj[edge[1]].add(edge[0])
        self.degrees[edge[0]] += 1
        self.degrees[edge[1]] += 1
        self._add_edge_index(edge)
        self.m += 1
        self._toggle_diff_edge(edge)
        self._degree_gini_cache = None

    def delete_edge(self, u: int, v: int) -> None:
        edge = canonical_edge(u, v)
        if edge not in self.edge_positions:
            raise ValueError(f"edge does not exist: {edge}")

        common = self.common_neighbor_count(edge[0], edge[1])
        deg_u = int(self.degrees[edge[0]])
        deg_v = int(self.degrees[edge[1]])

        self.triangles -= common
        self.triplets -= (deg_u - 1) + (deg_v - 1)
        self.adj[edge[0]].remove(edge[1])
        self.adj[edge[1]].remove(edge[0])
        self.degrees[edge[0]] -= 1
        self.degrees[edge[1]] -= 1
        self._remove_edge_index(edge)
        self.m -= 1
        self._toggle_diff_edge(edge)
        self._degree_gini_cache = None

    def degree_gini(self) -> float:
        if self._degree_gini_cache is None:
            self._degree_gini_cache = degree_gini_from_degrees(self.degrees)
        return float(self._degree_gini_cache)

    def density(self) -> float:
        denom = self.n * (self.n - 1)
        return safe_divide(2.0 * self.m, denom)

    def clustering(self) -> float:
        return transitivity_from_counts(self.triangles, self.triplets)

    def edit_distance(self) -> int:
        return len(self.diff_edges)

    def stats(self) -> dict[str, float]:
        return {
            "density": self.density(),
            "degree_gini": self.degree_gini(),
            "clustering": self.clustering(),
        }

    def is_connected(self) -> bool:
        if self.n <= 1:
            return True
        start = 0
        visited = [False] * self.n
        visited[start] = True
        seen = 1
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in self.adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    seen += 1
                    if seen == self.n:
                        return True
                    stack.append(neighbor)
        return False

    def to_networkx(self, *, restore_labels: bool = True) -> nx.Graph:
        graph = nx.Graph()
        for node in range(self.n):
            label = self.node_labels[node] if restore_labels else node
            graph.add_node(label, original_label=self.node_labels[node])
        for u, v in self.edges:
            left = self.node_labels[u] if restore_labels else u
            right = self.node_labels[v] if restore_labels else v
            graph.add_edge(left, right)
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
    if cfg.require_connected and not state.is_connected():
        return cfg.hard_constraint_penalty
    return 0.0


def stat_errors(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
) -> tuple[dict[str, float], float, float]:
    current = state.stats()
    target_error = abs(current[target_name] - target_value)
    preserve_errors = {
        stat_name: abs(current[stat_name] - baseline_stats[stat_name])
        for stat_name in STAT_NAMES
        if stat_name != target_name
    }
    preserve_total = float(sum(preserve_errors.values()))
    return current, target_error, preserve_total


def preserve_excess(
    current_stats: dict[str, float],
    baseline_stats: dict[str, float],
    target_name: str,
    cfg: SAConfig,
) -> tuple[dict[str, float], float]:
    per_stat: dict[str, float] = {}
    total = 0.0
    for stat_name in STAT_NAMES:
        if stat_name == target_name:
            continue
        error = abs(current_stats[stat_name] - baseline_stats[stat_name])
        tol = cfg.preserve_tolerances[stat_name]
        excess = max(0.0, error - tol)
        per_stat[stat_name] = excess
        total += excess
    return per_stat, total


def is_feasible(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
) -> bool:
    current = state.stats()
    if abs(current[target_name] - target_value) > cfg.target_tolerances[target_name]:
        return False
    for stat_name in STAT_NAMES:
        if stat_name == target_name:
            continue
        if abs(current[stat_name] - baseline_stats[stat_name]) > cfg.preserve_tolerances[stat_name]:
            return False
    return hard_violation_penalty(state, cfg) == 0.0


def attainment_energy(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
) -> float:
    current = state.stats()
    target_loss = abs(current[target_name] - target_value) / cfg.tolerance_scale(
        target_name, target=True
    )
    preserve_loss = 0.0
    for stat_name in STAT_NAMES:
        if stat_name == target_name:
            continue
        excess = max(
            0.0,
            abs(current[stat_name] - baseline_stats[stat_name])
            - cfg.preserve_tolerances[stat_name],
        )
        preserve_loss += excess / cfg.tolerance_scale(stat_name, target=False)

    distance_loss = safe_divide(state.edit_distance(), max(1, len(state.original_edge_set)))
    hard_loss = hard_violation_penalty(state, cfg)

    return (
        cfg.attain_target_weight * target_loss
        + cfg.attain_preserve_weight * preserve_loss
        + cfg.attain_distance_weight * distance_loss
        + hard_loss
    )


def compression_energy(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
    cfg: SAConfig,
) -> float:
    current = state.stats()
    target_loss = abs(current[target_name] - target_value) / cfg.tolerance_scale(
        target_name, target=True
    )
    preserve_loss = 0.0
    for stat_name in STAT_NAMES:
        if stat_name == target_name:
            continue
        preserve_loss += abs(current[stat_name] - baseline_stats[stat_name]) / cfg.tolerance_scale(
            stat_name, target=False
        )

    distance_loss = safe_divide(state.edit_distance(), max(1, len(state.original_edge_set)))
    return (
        cfg.compress_distance_weight * distance_loss
        + cfg.compress_target_weight * target_loss
        + cfg.compress_preserve_weight * preserve_loss
    )


def feasible_lexicographic_key(
    state: GraphState,
    baseline_stats: dict[str, float],
    target_name: str,
    target_value: float,
) -> tuple[float, float, float]:
    current = state.stats()
    target_error = abs(current[target_name] - target_value)
    preserve_total = sum(
        abs(current[stat_name] - baseline_stats[stat_name])
        for stat_name in STAT_NAMES
        if stat_name != target_name
    )
    return (float(state.edit_distance()), float(target_error), float(preserve_total))


def _sample_edge(state: GraphState, rng: np.random.Generator) -> tuple[int, int] | None:
    if not state.edges:
        return None
    idx = int(rng.integers(len(state.edges)))
    return state.edges[idx]


def _sample_nonedge(
    state: GraphState,
    rng: np.random.Generator,
    max_trials: int,
    forbidden_nodes: set[int] | None = None,
) -> tuple[int, int] | None:
    forbidden = forbidden_nodes or set()
    for _ in range(max_trials):
        u = int(rng.integers(state.n))
        v = int(rng.integers(state.n))
        if u == v or u in forbidden or v in forbidden:
            continue
        edge = canonical_edge(u, v)
        if edge not in state.edge_positions:
            return edge
    return None


def _sample_extreme_node(
    state: GraphState,
    rng: np.random.Generator,
    *,
    high: bool,
    must_have_edge: bool,
    exclude: set[int] | None = None,
    batch_size: int,
) -> int | None:
    exclude = exclude or set()
    candidates: list[int] = []
    for _ in range(batch_size * 3):
        node = int(rng.integers(state.n))
        if node in exclude:
            continue
        if must_have_edge and state.degrees[node] <= 0:
            continue
        candidates.append(node)
        if len(candidates) >= batch_size:
            break
    if not candidates:
        return None
    key = lambda node: int(state.degrees[node])
    return max(candidates, key=key) if high else min(candidates, key=key)


def _sample_triangle_edge(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> tuple[int, int] | None:
    best_edge = None
    best_score = -1
    for _ in range(cfg.node_selection_batch):
        edge = _sample_edge(state, rng)
        if edge is None:
            break
        score = state.common_neighbor_count(*edge)
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

        pairings = (
            (canonical_edge(a, c), canonical_edge(b, d)),
            (canonical_edge(a, d), canonical_edge(b, c)),
        )
        order = [0, 1]
        rng.shuffle(order)
        for idx in order:
            move = Move(
                kind="double_edge_swap",
                remove=(first, second),
                add=pairings[idx],
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
    current = state.clustering()
    increase = target_value >= current

    if increase:
        for _ in range(cfg.candidate_trials):
            w = int(rng.integers(state.n))
            neighbors = tuple(state.adj[w])
            if len(neighbors) < 2:
                continue
            idx = rng.choice(len(neighbors), size=2, replace=False)
            u, v = neighbors[int(idx[0])], neighbors[int(idx[1])]
            if state.has_edge(u, v):
                continue

            u_neighbors = tuple(x for x in state.adj[u] if x not in {w, v})
            v_neighbors = tuple(x for x in state.adj[v] if x not in {w, u})
            if not u_neighbors or not v_neighbors:
                continue
            for _inner in range(8):
                x = u_neighbors[int(rng.integers(len(u_neighbors)))]
                y = v_neighbors[int(rng.integers(len(v_neighbors)))]
                if len({u, v, w, x, y}) < 5:
                    continue
                move = Move(
                    kind="double_edge_swap",
                    remove=(canonical_edge(u, x), canonical_edge(v, y)),
                    add=(canonical_edge(u, v), canonical_edge(x, y)),
                )
                if move_is_legal(state, move):
                    return move
    else:
        triangle_edge = _sample_triangle_edge(state, rng, cfg)
        if triangle_edge is None or state.common_neighbor_count(*triangle_edge) <= 0:
            return sample_random_swap(state, rng, cfg)
        u, v = triangle_edge
        for _ in range(cfg.candidate_trials):
            other = _sample_edge(state, rng)
            if other is None or other == triangle_edge:
                continue
            x, y = other
            if len({u, v, x, y}) < 4:
                continue
            options = (
                (canonical_edge(u, x), canonical_edge(v, y)),
                (canonical_edge(u, y), canonical_edge(v, x)),
            )
            order = [0, 1]
            rng.shuffle(order)
            for idx in order:
                move = Move(
                    kind="double_edge_swap",
                    remove=(triangle_edge, other),
                    add=options[idx],
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
        if rng.random() < 0.5:
            moving, fixed = u, v
        else:
            moving, fixed = v, u

        for _inner in range(12):
            w = int(rng.integers(state.n))
            if w in {moving, fixed}:
                continue
            new_edge = canonical_edge(w, fixed)
            move = Move(
                kind="endpoint_rewire",
                remove=(edge,),
                add=(new_edge,),
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

    for _ in range(cfg.candidate_trials):
        if increase:
            moving = _sample_extreme_node(
                state,
                rng,
                high=False,
                must_have_edge=True,
                batch_size=cfg.node_selection_batch,
            )
            new_node = _sample_extreme_node(
                state,
                rng,
                high=True,
                must_have_edge=False,
                exclude={moving} if moving is not None else None,
                batch_size=cfg.node_selection_batch,
            )
        else:
            moving = _sample_extreme_node(
                state,
                rng,
                high=True,
                must_have_edge=True,
                batch_size=cfg.node_selection_batch,
            )
            new_node = _sample_extreme_node(
                state,
                rng,
                high=False,
                must_have_edge=False,
                exclude={moving} if moving is not None else None,
                batch_size=cfg.node_selection_batch,
            )

        if moving is None or new_node is None or moving == new_node:
            continue

        neighbors = tuple(state.adj[moving])
        if not neighbors:
            continue
        fixed = neighbors[int(rng.integers(len(neighbors)))]
        if new_node in {moving, fixed}:
            continue

        move = Move(
            kind="endpoint_rewire",
            remove=(canonical_edge(moving, fixed),),
            add=(canonical_edge(new_node, fixed),),
        )
        if move_is_legal(state, move):
            return move

    return sample_random_rewire(state, rng, cfg)


def sample_density_repair_move(
    state: GraphState, rng: np.random.Generator, cfg: SAConfig
) -> Move | None:
    added_edges = [
        edge for edge in state.diff_edges if edge not in state.original_edge_set and edge in state.edge_positions
    ]
    missing_original = [
        edge for edge in state.diff_edges if edge in state.original_edge_set and edge not in state.edge_positions
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


def sample_random_density_move(
    state: GraphState,
    target_value: float,
    rng: np.random.Generator,
    cfg: SAConfig,
) -> Move | None:
    current = state.density()
    prefer_add = target_value >= current
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
        repair_move = sample_density_repair_move(state, rng, cfg)
        if repair_move is not None:
            return repair_move
    return sample_random_density_move(state, target_value, rng, cfg)


def _sampler_for_target(
    target_name: str,
    *,
    targeted: bool,
):
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
            if move is None:
                continue
            signature = move.signature()
            if signature in seen or not move_is_legal(state, move):
                continue
            seen.add(signature)

            apply_move(state, move)
            hard_penalty = hard_violation_penalty(state, cfg)
            if phase == "compress":
                if hard_penalty > 0.0 or not is_feasible(
                    state, baseline_stats, target_name, target_value, cfg
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

        if candidates:
            break

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
        if phase == "compress":
            if hard_violation_penalty(state, cfg) > 0.0 or not is_feasible(
                state, baseline_stats, target_name, target_value, cfg
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
        feasible_lexicographic_key(best_feasible, baseline_stats, target_name, target_value)
        if best_feasible is not None
        else None
    )

    proposal_counts: Counter = Counter()
    accepted_counts: Counter = Counter()
    accepted = 0
    proposed = 0
    blocks_without_improvement = 0

    started = time.perf_counter()
    n_blocks = max(1, math.ceil(total_steps / cfg.temperature_block_size))

    for block_index in range(n_blocks):
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

            if phase == "compress":
                if hard_violation_penalty(state, cfg) > 0.0 or not is_feasible(
                    state, baseline_stats, target_name, target_value, cfg
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

                if is_feasible(state, baseline_stats, target_name, target_value, cfg):
                    candidate_key = feasible_lexicographic_key(
                        state, baseline_stats, target_name, target_value
                    )
                    if best_feasible is None or candidate_key < best_feasible_key:
                        best_feasible = state.clone()
                        best_feasible_key = candidate_key
                        improved = True
            else:
                undo_move(state, move)

        if improved:
            blocks_without_improvement = 0
        else:
            blocks_without_improvement += 1

        temperature *= cfg.cooling_alpha
        if blocks_without_improvement >= cfg.stall_blocks:
            break

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
        "best_overall_density": best_overall_stats["density"],
        "best_overall_degree_gini": best_overall_stats["degree_gini"],
        "best_overall_clustering": best_overall_stats["clustering"],
        "best_overall_edit_distance": best_overall.edit_distance(),
        **_flatten_counter(proposal_counts, "proposed"),
        **_flatten_counter(accepted_counts, "accepted"),
    }

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
    pilot_cfg = SAConfig(**{**cfg.__dict__, "attain_steps": cfg.pilot_steps, "compress_steps": 0})

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
        preserve_errors = {f"preserve_error_{name}": math.nan for name in STAT_NAMES if name != target_name}
    else:
        achieved_stats = state.stats()
        deltas = {
            f"delta_{name}": achieved_stats[name] - baseline_stats[name] for name in STAT_NAMES
        }
        edit_distance = state.edit_distance()
        feasible = is_feasible(state, baseline_stats, target_name, target_value, cfg)
        target_error = abs(achieved_stats[target_name] - target_value)
        preserve_errors = {
            f"preserve_error_{name}": abs(achieved_stats[name] - baseline_stats[name])
            for name in STAT_NAMES
            if name != target_name
        }

    row: dict[str, float | int | str | bool] = {
        "source_id": source_id,
        "source_path": str(source_path),
        "n": int(baseline_state.n),
        "m0": int(len(baseline_state.original_edge_set)),
        "target_stat": target_name,
        "target_value_requested": float(target_value),
        "seed": int(seed),
        "baseline_density": baseline_stats["density"],
        "baseline_degree_gini": baseline_stats["degree_gini"],
        "baseline_clustering": baseline_stats["clustering"],
        "target_value_achieved": achieved_stats.get(target_name, math.nan),
        "achieved_density": achieved_stats["density"],
        "achieved_degree_gini": achieved_stats["degree_gini"],
        "achieved_clustering": achieved_stats["clustering"],
        "target_error": target_error,
        "edit_distance": edit_distance,
        "feasible": feasible,
        **deltas,
        **preserve_errors,
    }

    row.update({f"attain_{key}": value for key, value in attain_meta.items()})
    if compress_meta is not None:
        row.update({f"compress_{key}": value for key, value in compress_meta.items()})
    return row


def run_batch_sweep(
    sources: Sequence[str | Path],
    cfg: SAConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_paths = resolve_source_paths(sources)
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
        baseline_state = GraphState.from_networkx(graph)
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
                        **grid_meta,
                    }
                )

        prepared_batches.append(
            (graph_index, source_path, baseline_state, baseline_stats, target_batches)
        )

    progress = (
        tqdm(total=total_runs, desc=cfg.progress_desc)
        if cfg.show_progress and tqdm is not None
        else None
    )

    try:
        for graph_index, source_path, baseline_state, baseline_stats, target_batches in prepared_batches:
            for target_index, target_name, target_grid in target_batches:
                for grid_index, target_value in enumerate(target_grid):
                    for seed_index in range(cfg.n_seeds):
                        seed = _seed_for_run(
                            cfg, graph_index, target_index, grid_index, seed_index
                        )
                        attained_state, attain_meta = anneal_attain(
                            baseline_state,
                            baseline_stats,
                            target_name,
                            float(target_value),
                            cfg,
                            seed=seed,
                        )

                        if attained_state is not None:
                            final_state, compress_meta = anneal_compress(
                                attained_state,
                                baseline_stats,
                                target_name,
                                float(target_value),
                                cfg,
                                seed=seed + 10_000_000,
                            )
                        else:
                            final_state, compress_meta = None, None

                        row = _record_run(
                            cfg=cfg,
                            source_id=source_path.stem,
                            source_path=source_path,
                            target_name=target_name,
                            target_value=float(target_value),
                            seed=seed,
                            baseline_stats=baseline_stats,
                            baseline_state=baseline_state,
                            attained_state=attained_state,
                            final_state=final_state,
                            attain_meta=attain_meta,
                            compress_meta=compress_meta,
                        )
                        row["saved_graph_path"] = ""

                        chosen_state = final_state or attained_state
                        save_key = (source_path.stem, target_name)
                        should_save = (
                            cfg.save_selected_graphs
                            and row["feasible"]
                            and chosen_state is not None
                            and saved_counts[save_key] < cfg.max_saved_graphs_per_group
                        )
                        if should_save:
                            save_path = (
                                cfg.output_dir
                                / "saved_graphs"
                                / source_path.stem
                                / target_name
                                / f"grid_{grid_index:03d}_seed_{seed}.pkl"
                            )
                            save_graph_state(chosen_state, save_path)
                            row["saved_graph_path"] = str(save_path)
                            saved_counts[save_key] += 1

                        results.append(row)
                        if progress is not None:
                            progress.update(1)
                            progress.set_postfix_str(
                                f"{source_path.stem} | {target_name} | {grid_index + 1}/{len(target_grid)}"
                            )
    finally:
        if progress is not None:
            progress.close()

    results_df = pd.DataFrame(results)
    grid_df = pd.DataFrame(grid_rows)
    return results_df, grid_df


def save_results_table(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    stem: str = "sa_network_variation_results",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{stem}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = output_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path


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


def _ranked(values: Sequence[float]) -> np.ndarray:
    return stats.rankdata(values, method="average")


def _finite_arrays(*values: Sequence[float]) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays:
        return []
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return [array[mask] for array in arrays]


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr, y_arr = _finite_arrays(x, y)
    if x_arr.size < 2:
        return 0.0
    if np.std(x_arr) <= 0.0 or np.std(y_arr) <= 0.0:
        return 0.0
    corr = stats.spearmanr(x_arr, y_arr).statistic
    return 0.0 if not np.isfinite(corr) else float(corr)


def partial_spearman(x: Sequence[float], y: Sequence[float], control: Sequence[float]) -> float:
    x_arr, y_arr, z_arr = _finite_arrays(x, y, control)
    if x_arr.size < 3:
        return 0.0
    if np.std(x_arr) <= 0.0 or np.std(y_arr) <= 0.0:
        return 0.0
    x_rank = _ranked(x_arr)
    y_rank = _ranked(y_arr)
    z_rank = _ranked(z_arr)
    design = np.column_stack([np.ones(len(z_rank)), z_rank])
    beta_x, *_ = np.linalg.lstsq(design, x_rank, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y_rank, rcond=None)
    resid_x = x_rank - design @ beta_x
    resid_y = y_rank - design @ beta_y
    if np.std(resid_x) <= 0.0 or np.std(resid_y) <= 0.0:
        return 0.0
    corr = float(np.corrcoef(resid_x, resid_y)[0, 1])
    return 0.0 if not np.isfinite(corr) else corr


def safe_spearman_matrix(
    df: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for left in columns:
        for right in columns:
            if left == right:
                matrix.loc[left, right] = 1.0
            else:
                matrix.loc[left, right] = safe_spearman(df[left], df[right])
    return matrix


def _correlation_columns(variable_set: str) -> tuple[str, ...]:
    if variable_set == "achieved":
        return (
            "achieved_density",
            "achieved_degree_gini",
            "achieved_clustering",
        )
    if variable_set == "delta":
        return (
            "delta_density",
            "delta_degree_gini",
            "delta_clustering",
        )
    raise ValueError("variable_set must be 'achieved' or 'delta'")


def _clean_correlation_frame(
    results_df: pd.DataFrame,
    *,
    variable_set: str,
    feasible_only: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    columns = _correlation_columns(variable_set)
    subset = results_df.copy()
    if feasible_only and "feasible" in subset.columns:
        subset = subset.loc[subset["feasible"].fillna(False)]
    missing = [column for column in columns if column not in subset.columns]
    if missing:
        raise ValueError("results_df is missing columns: " + ", ".join(missing))
    for column in columns:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    keep_columns = list(columns)
    if "edit_distance" in subset.columns:
        subset["edit_distance"] = pd.to_numeric(subset["edit_distance"], errors="coerce")
        keep_columns.append("edit_distance")
    for column in ("source_id", "target_stat"):
        if column in subset.columns:
            keep_columns.append(column)
    return subset[keep_columns].dropna(subset=list(columns)), columns


def compute_overall_correlation_matrix(
    results_df: pd.DataFrame,
    *,
    variable_set: str = "achieved",
    feasible_only: bool = True,
) -> pd.DataFrame:
    clean, columns = _clean_correlation_frame(
        results_df,
        variable_set=variable_set,
        feasible_only=feasible_only,
    )
    if clean.empty:
        return pd.DataFrame(np.nan, index=columns, columns=columns)
    return safe_spearman_matrix(clean, columns)


def compute_overall_correlation_table(
    results_df: pd.DataFrame,
    *,
    variable_set: str = "achieved",
    feasible_only: bool = True,
    partial_control: str | None = "edit_distance",
) -> pd.DataFrame:
    clean, columns = _clean_correlation_frame(
        results_df,
        variable_set=variable_set,
        feasible_only=feasible_only,
    )
    pairs = [
        (columns[0], columns[1]),
        (columns[0], columns[2]),
        (columns[1], columns[2]),
    ]
    rows = []
    for left, right in pairs:
        pair_df = clean[[left, right]].dropna()
        row = {
            "variable_set": variable_set,
            "left": left,
            "right": right,
            "spearman_rho": safe_spearman(pair_df[left], pair_df[right]),
            "n_obs": int(len(pair_df)),
        }
        if partial_control is not None and partial_control in clean.columns:
            partial_df = clean[[left, right, partial_control]].dropna()
            row[f"partial_spearman_rho_control_{partial_control}"] = partial_spearman(
                partial_df[left],
                partial_df[right],
                partial_df[partial_control],
            )
            row["partial_n_obs"] = int(len(partial_df))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_residualized_correlation_matrix(
    results_df: pd.DataFrame,
    *,
    variable_set: str = "achieved",
    feasible_only: bool = True,
    group_cols: Sequence[str] = ("source_id", "target_stat"),
) -> pd.DataFrame:
    clean, columns = _clean_correlation_frame(
        results_df,
        variable_set=variable_set,
        feasible_only=feasible_only,
    )
    usable_group_cols = [column for column in group_cols if column in clean.columns]
    if clean.empty:
        return pd.DataFrame(np.nan, index=columns, columns=columns)
    if not usable_group_cols:
        return safe_spearman_matrix(clean, columns)

    residual_df = pd.DataFrame(index=clean.index)
    grouped = clean.groupby(usable_group_cols, dropna=False)
    for column in columns:
        residual_df[column] = clean[column] - grouped[column].transform("mean")
    return safe_spearman_matrix(residual_df, columns)


def compute_residualized_correlation_table(
    results_df: pd.DataFrame,
    *,
    variable_set: str = "achieved",
    feasible_only: bool = True,
    group_cols: Sequence[str] = ("source_id", "target_stat"),
    partial_control: str | None = "edit_distance",
) -> pd.DataFrame:
    clean, columns = _clean_correlation_frame(
        results_df,
        variable_set=variable_set,
        feasible_only=feasible_only,
    )
    usable_group_cols = [column for column in group_cols if column in clean.columns]
    if usable_group_cols:
        grouped = clean.groupby(usable_group_cols, dropna=False)
        residual_df = pd.DataFrame(index=clean.index)
        for column in columns:
            residual_df[column] = clean[column] - grouped[column].transform("mean")
        if partial_control is not None and partial_control in clean.columns:
            residual_df[partial_control] = clean[partial_control] - grouped[
                partial_control
            ].transform("mean")
    else:
        residual_df = clean.copy()

    pairs = [
        (columns[0], columns[1]),
        (columns[0], columns[2]),
        (columns[1], columns[2]),
    ]
    rows = []
    for left, right in pairs:
        pair_df = residual_df[[left, right]].dropna()
        row = {
            "variable_set": variable_set,
            "group_cols": ",".join(usable_group_cols) if usable_group_cols else "",
            "left": left,
            "right": right,
            "spearman_rho": safe_spearman(pair_df[left], pair_df[right]),
            "n_obs": int(len(pair_df)),
        }
        if partial_control is not None and partial_control in residual_df.columns:
            partial_df = residual_df[[left, right, partial_control]].dropna()
            row[f"partial_spearman_rho_control_{partial_control}"] = partial_spearman(
                partial_df[left],
                partial_df[right],
                partial_df[partial_control],
            )
            row["partial_n_obs"] = int(len(partial_df))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_target_intervention_diagnostics(
    results_df: pd.DataFrame,
    *,
    feasible_only: bool = True,
    source_id: str | None = None,
) -> pd.DataFrame:
    """Summarize whether each target intervention drags preserved stats along.

    The key question is target-specific: within density-target runs, does
    delta_density correlate with delta_degree_gini or delta_clustering?
    Pooled correlations across all target types answer a different question.
    """
    subset = results_df.copy()
    if feasible_only and "feasible" in subset.columns:
        subset = subset.loc[subset["feasible"].fillna(False)]
    if source_id is not None and "source_id" in subset.columns:
        subset = subset.loc[subset["source_id"] == source_id]

    rows: list[dict[str, float | str | int | None]] = []
    for target_stat in STAT_NAMES:
        target_col = f"delta_{target_stat}"
        if target_col not in subset.columns:
            continue
        target_subset = subset.loc[subset["target_stat"] == target_stat].copy()
        target_subset[target_col] = pd.to_numeric(
            target_subset[target_col], errors="coerce"
        )
        if "edit_distance" in target_subset.columns:
            target_subset["edit_distance"] = pd.to_numeric(
                target_subset["edit_distance"], errors="coerce"
            )

        for response_stat in STAT_NAMES:
            if response_stat == target_stat:
                continue
            response_col = f"delta_{response_stat}"
            if response_col not in target_subset.columns:
                continue
            response_values = pd.to_numeric(
                target_subset[response_col], errors="coerce"
            )
            clean = pd.DataFrame(
                {
                    target_col: target_subset[target_col],
                    response_col: response_values,
                }
            ).dropna()
            abs_response = clean[response_col].abs()

            row: dict[str, float | str | int | None] = {
                "source_id": source_id,
                "target_stat": target_stat,
                "response_stat": response_stat,
                "target_delta": target_col,
                "response_delta": response_col,
                "n_obs": int(len(clean)),
                "target_delta_min": float(clean[target_col].min())
                if len(clean)
                else math.nan,
                "target_delta_max": float(clean[target_col].max())
                if len(clean)
                else math.nan,
                "target_delta_sd": float(clean[target_col].std(ddof=0))
                if len(clean)
                else math.nan,
                "response_delta_mean": float(clean[response_col].mean())
                if len(clean)
                else math.nan,
                "response_delta_sd": float(clean[response_col].std(ddof=0))
                if len(clean)
                else math.nan,
                "response_abs_delta_median": float(abs_response.median())
                if len(clean)
                else math.nan,
                "response_abs_delta_p95": float(abs_response.quantile(0.95))
                if len(clean)
                else math.nan,
                "response_abs_delta_max": float(abs_response.max())
                if len(clean)
                else math.nan,
                "spearman_rho_target_vs_response": safe_spearman(
                    clean[target_col], clean[response_col]
                ),
            }

            preserve_error_col = f"preserve_error_{response_stat}"
            if preserve_error_col in target_subset.columns:
                preserve_values = pd.to_numeric(
                    target_subset[preserve_error_col], errors="coerce"
                ).dropna()
                row["preserve_error_median"] = (
                    float(preserve_values.median())
                    if len(preserve_values)
                    else math.nan
                )
                row["preserve_error_p95"] = (
                    float(preserve_values.quantile(0.95))
                    if len(preserve_values)
                    else math.nan
                )

            if "edit_distance" in target_subset.columns:
                partial_clean = pd.DataFrame(
                    {
                        target_col: target_subset[target_col],
                        response_col: response_values,
                        "edit_distance": target_subset["edit_distance"],
                    }
                ).dropna()
                row["partial_spearman_rho_control_edit_distance"] = partial_spearman(
                    partial_clean[target_col],
                    partial_clean[response_col],
                    partial_clean["edit_distance"],
                )
                row["partial_n_obs"] = int(len(partial_clean))

            rows.append(row)

    return pd.DataFrame(rows)


def regression_with_edit_distance(
    df: pd.DataFrame, *, target_col: str, response_col: str
) -> dict[str, float]:
    subset = df[[target_col, response_col, "edit_distance"]].dropna()
    if len(subset) < 3:
        return {
            "beta_intercept": math.nan,
            "beta_target": math.nan,
            "beta_edit_distance": math.nan,
            "r_squared": math.nan,
            "n_obs": len(subset),
        }

    X = np.column_stack(
        [
            np.ones(len(subset)),
            subset[target_col].to_numpy(dtype=float),
            subset["edit_distance"].to_numpy(dtype=float),
        ]
    )
    y = subset[response_col].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "beta_intercept": float(beta[0]),
        "beta_target": float(beta[1]),
        "beta_edit_distance": float(beta[2]),
        "r_squared": 1.0 - safe_divide(ss_res, ss_tot),
        "n_obs": int(len(subset)),
    }


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


def fisher_z_average(correlations: Sequence[float]) -> float:
    values = np.asarray(correlations, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    clipped = np.clip(values, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (math.nan, math.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    stats_boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        stats_boot[idx] = np.median(sample)
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.quantile(stats_boot, alpha)),
        float(np.quantile(stats_boot, 1.0 - alpha)),
    )


def aggregate_correlations(
    corr_df: pd.DataFrame, value_col: str
) -> pd.DataFrame:
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


def plot_requested_vs_achieved(results_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(STAT_NAMES), figsize=(5 * len(STAT_NAMES), 4))
    if len(STAT_NAMES) == 1:
        axes = [axes]

    for ax, stat_name in zip(axes, STAT_NAMES):
        subset = results_df.loc[results_df["target_stat"] == stat_name]
        ax.scatter(
            subset["target_value_requested"],
            subset["target_value_achieved"],
            s=18,
            alpha=0.4,
        )
        bounds = [
            min(subset["target_value_requested"].min(), subset["target_value_achieved"].min()),
            max(subset["target_value_requested"].max(), subset["target_value_achieved"].max()),
        ]
        ax.plot(bounds, bounds, linestyle="--", color="black", linewidth=1)
        ax.set_title(stat_name)
        ax.set_xlabel("Requested")
        ax.set_ylabel("Achieved")

    fig.tight_layout()
    return fig


def plot_edit_distance_vs_target_delta(results_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(STAT_NAMES), figsize=(5 * len(STAT_NAMES), 4))
    if len(STAT_NAMES) == 1:
        axes = [axes]

    for ax, stat_name in zip(axes, STAT_NAMES):
        subset = results_df.loc[results_df["target_stat"] == stat_name].copy()
        subset["target_delta"] = subset[f"delta_{stat_name}"]
        ax.scatter(subset["edit_distance"], subset["target_delta"], s=18, alpha=0.4)
        ax.set_title(stat_name)
        ax.set_xlabel("Edit distance")
        ax.set_ylabel(f"Delta {stat_name}")

    fig.tight_layout()
    return fig


def plot_target_delta_vs_edit_distance(
    results_df: pd.DataFrame,
    *,
    source_id: str | None = None,
    feasible_only: bool = False,
    hue: str | None = "target_stat",
    facet_by_target: bool = True,
) -> plt.Figure:
    subset = results_df.copy()
    if source_id is not None:
        subset = subset.loc[subset["source_id"] == source_id]
    if feasible_only and "feasible" in subset.columns:
        subset = subset.loc[subset["feasible"].fillna(False)]

    frames = []
    for stat_name in STAT_NAMES:
        stat_subset = subset.loc[subset["target_stat"] == stat_name].copy()
        delta_column = f"delta_{stat_name}"
        if delta_column not in stat_subset.columns:
            continue
        stat_subset["target_delta"] = pd.to_numeric(
            stat_subset[delta_column], errors="coerce"
        )
        stat_subset["edit_distance"] = pd.to_numeric(
            stat_subset["edit_distance"], errors="coerce"
        )
        frames.append(stat_subset)

    if not frames:
        raise ValueError("No target-delta columns are available to plot")

    plot_df = pd.concat(frames, ignore_index=True)
    required_columns = ["target_delta", "edit_distance"]
    if hue is not None and hue in plot_df.columns:
        required_columns.append(hue)
    plot_df = plot_df.dropna(subset=required_columns)
    if plot_df.empty:
        raise ValueError("No rows available for target-change versus edit-distance plot")

    if facet_by_target:
        targets = [
            stat_name
            for stat_name in STAT_NAMES
            if stat_name in set(plot_df["target_stat"])
        ]
        fig, axes = plt.subplots(
            1,
            len(targets),
            figsize=(5.2 * len(targets), 4.6),
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()
        for ax, stat_name in zip(axes, targets):
            stat_df = plot_df.loc[plot_df["target_stat"] == stat_name]
            if hue is not None and hue != "target_stat" and hue in stat_df.columns:
                for label, group in stat_df.groupby(hue, sort=False):
                    ax.scatter(
                        group["target_delta"],
                        group["edit_distance"],
                        s=22,
                        alpha=0.45,
                        label=str(label),
                    )
                ax.legend(title=hue)
            else:
                ax.scatter(
                    stat_df["target_delta"],
                    stat_df["edit_distance"],
                    s=22,
                    alpha=0.45,
                )
            ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(stat_name)
            ax.set_xlabel(f"Achieved delta {stat_name}")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Edit distance from original graph")
        title = "Target change vs edit distance"
        if source_id is not None:
            title = f"{source_id}: {title}"
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(7, 5))
    if hue is not None and hue in plot_df.columns:
        for label, group in plot_df.groupby(hue, sort=False):
            ax.scatter(
                group["target_delta"],
                group["edit_distance"],
                s=22,
                alpha=0.45,
                label=str(label),
            )
        ax.legend(title=hue)
    else:
        ax.scatter(plot_df["target_delta"], plot_df["edit_distance"], s=22, alpha=0.45)

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Achieved target-stat change from baseline (+/-)")
    ax.set_ylabel("Edit distance from original graph")
    ax.set_title("Target change vs edit distance")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_delta_scatter(
    results_df: pd.DataFrame, *, source_id: str | None = None, target_stat: str | None = None
) -> plt.Figure:
    subset = results_df.copy()
    if source_id is not None:
        subset = subset.loc[subset["source_id"] == source_id]
    if target_stat is not None:
        subset = subset.loc[subset["target_stat"] == target_stat]

    pairs = [
        ("delta_density", "delta_degree_gini"),
        ("delta_density", "delta_clustering"),
        ("delta_degree_gini", "delta_clustering"),
    ]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    for ax, (left, right) in zip(axes, pairs):
        ax.scatter(subset[left], subset[right], s=18, alpha=0.35)
        ax.set_xlabel(left)
        ax.set_ylabel(right)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.axvline(0.0, color="black", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_target_intervention_scatter(
    results_df: pd.DataFrame,
    *,
    source_id: str | None = None,
    feasible_only: bool = True,
) -> plt.Figure:
    """Plot target-delta against each off-target delta for each intervention."""
    subset = results_df.copy()
    if feasible_only and "feasible" in subset.columns:
        subset = subset.loc[subset["feasible"].fillna(False)]
    if source_id is not None and "source_id" in subset.columns:
        subset = subset.loc[subset["source_id"] == source_id]

    fig, axes = plt.subplots(
        len(STAT_NAMES),
        len(STAT_NAMES) - 1,
        figsize=(11, 3.6 * len(STAT_NAMES)),
        squeeze=False,
    )

    for row_idx, target_stat in enumerate(STAT_NAMES):
        target_col = f"delta_{target_stat}"
        responses = [name for name in STAT_NAMES if name != target_stat]
        target_subset = subset.loc[subset["target_stat"] == target_stat].copy()

        for col_idx, response_stat in enumerate(responses):
            ax = axes[row_idx, col_idx]
            response_col = f"delta_{response_stat}"
            if target_col not in target_subset.columns or response_col not in target_subset.columns:
                ax.text(0.5, 0.5, "missing columns", ha="center", va="center")
                ax.set_axis_off()
                continue

            plot_df = pd.DataFrame(
                {
                    target_col: pd.to_numeric(
                        target_subset[target_col], errors="coerce"
                    ),
                    response_col: pd.to_numeric(
                        target_subset[response_col], errors="coerce"
                    ),
                }
            ).dropna()

            if plot_df.empty:
                ax.text(0.5, 0.5, "no feasible rows", ha="center", va="center")
                ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
                ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--")
            else:
                ax.scatter(
                    plot_df[target_col],
                    plot_df[response_col],
                    s=22,
                    alpha=0.45,
                )
                rho = safe_spearman(plot_df[target_col], plot_df[response_col])
                median_abs = float(plot_df[response_col].abs().median())
                ax.text(
                    0.03,
                    0.97,
                    f"rho={rho:.2g}\nmedian |drift|={median_abs:.2g}\nn={len(plot_df)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                )
                ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
                ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--")

            ax.set_title(f"{target_stat} target: {response_stat} drift")
            ax.set_xlabel(f"delta_{target_stat}")
            ax.set_ylabel(f"delta_{response_stat}")
            ax.grid(True, alpha=0.2)

    title = "Target-specific off-target drift"
    if source_id is not None:
        title = f"{source_id}: {title}"
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def plot_target_intervention_diagnostic_heatmap(
    results_df: pd.DataFrame,
    *,
    metric: str = "spearman_rho_target_vs_response",
    source_id: str | None = None,
    feasible_only: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """Heatmap of target-specific coupling or drift metrics.

    Rows are the statistic being targeted. Columns are the statistic that should
    be preserved. The diagonal is intentionally blank.
    """
    diagnostics = compute_target_intervention_diagnostics(
        results_df,
        feasible_only=feasible_only,
        source_id=source_id,
    )
    matrix = pd.DataFrame(np.nan, index=STAT_NAMES, columns=STAT_NAMES, dtype=float)
    if not diagnostics.empty and metric in diagnostics.columns:
        for _, row in diagnostics.iterrows():
            matrix.loc[row["target_stat"], row["response_stat"]] = row[metric]

    is_correlation = "rho" in metric or "correlation" in metric
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2g",
        cmap="coolwarm" if is_correlation else "viridis",
        center=0.0 if is_correlation else None,
        vmin=-1.0 if is_correlation else None,
        vmax=1.0 if is_correlation else None,
        mask=matrix.isna(),
        ax=ax,
    )
    ax.set_xlabel("Preserved/off-target statistic")
    ax.set_ylabel("Intervention target")
    if title is None:
        title = metric.replace("_", " ")
        if source_id is not None:
            title = f"{source_id}: {title}"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_final_pairplot(
    results_df: pd.DataFrame,
    *,
    source_id: str | None = None,
    target_stat: str | None = None,
    feasible_only: bool = True,
    hue: str | None = "target_stat",
    variables: Sequence[str] = (
        "achieved_density",
        "achieved_degree_gini",
        "achieved_clustering",
    ),
):
    subset = results_df.copy()
    if feasible_only and "feasible" in subset.columns:
        subset = subset.loc[subset["feasible"].fillna(False)]
    if source_id is not None:
        subset = subset.loc[subset["source_id"] == source_id]
    if target_stat is not None:
        subset = subset.loc[subset["target_stat"] == target_stat]

    plot_df = pd.DataFrame(index=subset.index)
    fallback_map = {
        "achieved_density": [
            "achieved_density",
            "compress_best_overall_density",
            "attain_best_overall_density",
        ],
        "achieved_degree_gini": [
            "achieved_degree_gini",
            "compress_best_overall_degree_gini",
            "attain_best_overall_degree_gini",
        ],
        "achieved_clustering": [
            "achieved_clustering",
            "compress_best_overall_clustering",
            "attain_best_overall_clustering",
        ],
    }

    for variable in variables:
        if variable in fallback_map:
            filled = pd.Series(np.nan, index=subset.index, dtype=float)
            for column in fallback_map[variable]:
                if column in subset.columns:
                    filled = filled.fillna(pd.to_numeric(subset[column], errors="coerce"))
            plot_df[variable] = filled
        else:
            plot_df[variable] = pd.to_numeric(subset[variable], errors="coerce")

    plot_columns = list(plot_df.columns)
    if hue is not None and hue in subset.columns and hue not in plot_columns:
        plot_df[hue] = subset[hue]
        plot_columns.append(hue)

    plot_df = plot_df.dropna()
    if plot_df.empty:
        raise ValueError("No rows available for the final-variable pairplot")

    grid = sns.pairplot(
        plot_df,
        vars=list(variables),
        hue=hue if hue is not None and hue in plot_df.columns else None,
        corner=True,
        diag_kind="hist",
        plot_kws={"s": 18, "alpha": 0.4},
        diag_kws={"bins": 20},
    )
    return grid


def plot_correlation_heatmap(
    matrix: pd.DataFrame, *, title: str | None = None, annot: bool = True
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=annot, cmap="coolwarm", center=0.0, ax=ax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


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
