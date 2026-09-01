"""Structural checks on the repository itself, not on the model's logic.

These four checks were the substance of the old HOUSEKEEPING.md workflow, which
was a document asking to be run by hand. They cover failure modes the rest of
the suite cannot see -- a moved module, an unloadable network pickle, a notebook
corrupted by a bad merge -- and every one of them is a plausible outcome of an
ordinary file reorganisation, which is exactly when nobody remembers to run a
checklist. They are tests so that they run whether or not anyone remembers.
"""

import importlib
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CITATION_DATA = REPO_ROOT / "networks" / "citation_data"

# Node and edge counts for the three episodes the paper reports, as stated in
# the README table. Hard-coded on purpose: a silent change here means the
# published networks are not the ones the results were computed from.
PUBLISHED_NETWORKS = [
    ("pud_network.pkl", 90, 160),
    ("tobacco_network.pkl", 289, 1229),
    ("ego_network.pkl", 503, 2933),
]


# Every module a reader reaches through the notebooks or the test suite.
# `utils.network_plot_utils` is deliberately absent: it imports
# `NetworkInequality.edgebundling`, which is on no package index -- the known
# gap recorded in PUBLICATION_CHECKLIST.md.
IMPORTABLE_MODULES = [
    "model.agents",
    "model.bandit",
    "model.equality_study",
    "model.model",
    "model.simulation_functions",
    "model.vectorized_model",
    "model.vectorized_simulation_functions",
    "networks.variation_methods",
    "utils.imports",
    "utils.network_utils",
]


@pytest.mark.parametrize("module_name", IMPORTABLE_MODULES)
def test_module_imports_from_the_project_root(module_name):
    """The absolute-import convention holds across every package."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ["utils.imports", "utils.network_utils"])
def test_star_import_hub_exports_something(module_name):
    """The modules the notebooks consume with `from X import *`.

    Importing them is not enough: a hub that imports cleanly but exports nothing
    breaks every notebook downstream of it, silently and at the first use.
    Executed via `exec` because `import *` is a syntax error inside a function.
    """
    namespace = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102
    public = [k for k in namespace if not k.startswith("__")]
    assert public, module_name


@pytest.mark.parametrize("filename,n_nodes,n_edges", PUBLISHED_NETWORKS)
def test_published_network_loads_at_its_published_size(filename, n_nodes, n_edges):
    with open(CITATION_DATA / filename, "rb") as f:
        graph = pickle.load(f)
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == n_nodes
    assert graph.number_of_edges() == n_edges


def test_every_citation_network_unpickles():
    """All shipped networks must unpickle cleanly."""
    paths = sorted(CITATION_DATA.glob("*_network.pkl"))
    assert paths, f"no *_network.pkl found in {CITATION_DATA}"
    for path in paths:
        with open(path, "rb") as f:
            assert isinstance(pickle.load(f), nx.Graph), path.name


def test_notebooks_are_valid_json_with_cells():
    """Catches a notebook truncated or mangled by a merge or a failed save."""
    notebooks = sorted(REPO_ROOT.glob("*.ipynb"))
    assert len(notebooks) == 8, [nb.name for nb in notebooks]
    for notebook in notebooks:
        with open(notebook, encoding="utf-8") as f:
            content = json.load(f)
        assert content.get("cells"), notebook.name


@pytest.mark.slow
def test_snapshot_recording_survives_a_full_run():
    """End-to-end smoke test over the PUD network (~6s).

    Guards `VectorizedModel` and its snapshot bookkeeping together: with
    tolerance stopping off, 5,000 steps at an interval of 1,000 must record
    exactly five snapshots holding in-range values.
    """
    from model.vectorized_model import VectorizedModel

    with open(CITATION_DATA / "pud_network.pkl", "rb") as f:
        graph = pickle.load(f)
    graph = nx.relabel_nodes(graph, {node: i for i, node in enumerate(graph.nodes())})

    model = VectorizedModel(
        graph,
        n_experiments=5,
        uncertainty=0.001,
        agent_type="beta",
        snapshot_interval=1000,
        tolerance_stopping=False,
    )
    model.run_simulation(number_of_steps=5000, show_bar=False)

    snapshots = model.snapshots
    assert len(snapshots["step"]) == 5
    assert all(0 <= value <= 1 for value in snapshots["truth_share"])
    assert all(value >= 0 for value in snapshots["max_belief_change"])
