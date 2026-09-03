"""Phase-A variant cache: the two-phase run must not change the study.

`build_variants` builds every variant of an arm in parallel and dumps it; the
replicate loop in `run_arm` then loads instead of building. These tests pin the
properties that makes safe:

- one cache file per requested variant, and the resume set finds them;
- a cached variant is bit-identical to what an inline `build_setting` produces
  for the same index;
- the cache is resumable and respects `variant_slice`;
- a shard produced via the cache equals the shard produced without it.
"""

import pickle
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.equality_study import (  # noqa: E402
    build_setting,
    build_variants,
    completed_variant_caches,
    read_variant_cache,
    run_arm,
    variant_cache_dir,
    variant_cache_path,
    variant_seeds,
    variant_sequences,
)

CITATION_DATA = Path(__file__).resolve().parents[1] / "networks" / "citation_data"

MASTER_SEED = 20260723
UNCERTAINTY = 0.001
N_EXPERIMENTS = 1000
PROPORTION_EDGES_MAX = 0.1
BUILD_KWARGS = dict(proportion_edges_max=PROPORTION_EDGES_MAX)


@pytest.fixture(scope="module")
def pud():
    with open(CITATION_DATA / "pud_network.pkl", "rb") as f:
        G = pickle.load(f)
    return nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes())})


def _study_kwargs(**over):
    kw = dict(
        methods=("equalize",),
        master_seed=MASTER_SEED,
        n_variants=3,
        n_runs=4,
        uncertainty=UNCERTAINTY,
        n_experiments=N_EXPERIMENTS,
        num_cores=2,
        build_kwargs=BUILD_KWARGS,
        progress=False,
    )
    kw.update(over)
    return kw


def _fresh_setting(G, method, index, n_variants, n_runs):
    """What an inline build produces for one variant index."""
    sequences = variant_sequences(MASTER_SEED, "pud", method, n_variants)
    variation_seed, _ = variant_seeds(sequences[index], n_runs)
    return build_setting(
        G,
        method,
        variation_seed,
        uncertainty=UNCERTAINTY,
        n_experiments=N_EXPERIMENTS,
        **BUILD_KWARGS,
    )


def _assert_settings_equal(a, b):
    ga, gb = a["network"], b["network"]
    assert ga.is_directed() == gb.is_directed()
    assert set(ga.nodes()) == set(gb.nodes())
    assert set(ga.edges()) == set(gb.edges())
    for k in a:
        if k == "network":
            continue
        assert b[k] == pytest.approx(a[k]), k
    assert set(a) == set(b)


def test_build_variants_writes_one_cache_per_variant(pud, tmp_path):
    cost = build_variants([("pud", pud)], tmp_path, **_study_kwargs())

    directory = variant_cache_dir(tmp_path, "pud", "equalize")
    assert completed_variant_caches(directory) == {0, 1, 2}
    assert sorted(cost["variant_index"]) == [0, 1, 2]
    assert (cost["network"] == "pud").all()
    assert (cost["build_sec"] >= 0).all()


def test_cached_variant_matches_fresh_build(pud, tmp_path):
    build_variants([("pud", pud)], tmp_path, **_study_kwargs())

    for i in range(3):
        cached = read_variant_cache(
            variant_cache_path(tmp_path, "pud", "equalize", i)
        )
        _assert_settings_equal(_fresh_setting(pud, "equalize", i, 3, 4), cached)


def test_build_variants_is_resumable(pud, tmp_path):
    build_variants([("pud", pud)], tmp_path, **_study_kwargs())
    again = build_variants([("pud", pud)], tmp_path, **_study_kwargs())

    assert again.empty  # nothing left to build
    assert completed_variant_caches(
        variant_cache_dir(tmp_path, "pud", "equalize")
    ) == {0, 1, 2}


def test_variant_slice_restricts_the_build(pud, tmp_path):
    build_variants(
        [("pud", pud)], tmp_path, **_study_kwargs(n_variants=5, variant_slice=(1, 3))
    )

    assert completed_variant_caches(
        variant_cache_dir(tmp_path, "pud", "equalize")
    ) == {1, 2}


@pytest.mark.slow
def test_cache_does_not_change_the_shard(pud, tmp_path):
    """run_arm with the cache produces the same replicate rows as without it."""
    sim_kwargs = dict(
        tolerance_stopping=False,
        choice_stability_stopping=True,
        choice_stability_window=20,
        choice_stability_min_steps=0,
        record_choice_flips=False,
        number_of_steps=2000,
        show_bar=False,
        agent_type="beta",
    )
    common = dict(
        master_seed=MASTER_SEED,
        n_variants=2,
        n_runs=3,
        uncertainty=UNCERTAINTY,
        n_experiments=50,
        sim_kwargs=sim_kwargs,
        num_cores=2,
        build_kwargs=BUILD_KWARGS,
        progress=False,
    )

    no_cache = tmp_path / "no_cache"
    run_arm(pud, "pud", "equalize", no_cache, **common)

    cache_root = tmp_path / "cache"
    with_cache = tmp_path / "with_cache"
    build_variants(
        [("pud", pud)],
        cache_root,
        methods=("equalize",),
        master_seed=MASTER_SEED,
        n_variants=2,
        n_runs=3,
        uncertainty=UNCERTAINTY,
        n_experiments=50,
        num_cores=2,
        build_kwargs=BUILD_KWARGS,
        progress=False,
    )
    run_arm(
        pud, "pud", "equalize", with_cache, variant_cache_root=cache_root, **common
    )

    def _load(root):
        frames = [
            pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            for p in sorted((root / "pud_equalize").glob("variant_*"))
        ]
        return pd.concat(frames, ignore_index=True).sort_values(
            ["variant_index", "seed"]
        ).reset_index(drop=True)

    pd.testing.assert_frame_equal(_load(no_cache), _load(with_cache))
