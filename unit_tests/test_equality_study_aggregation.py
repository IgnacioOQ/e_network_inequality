"""Aggregation over a PARTIAL study.

The equality study is a multi-day job, so the normal state of its directories is
"some of it is done". These tests pin the two properties aggregation needs in
that state: a subset summary is identical to the same rows of a whole-arm
summary, and the columns downstream notebooks open by name do not move.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.equality_study import (  # noqa: E402
    OUTCOMES,
    VARIANT_COVARIATES,
    completed_variants,
    load_arm,
    shard_path,
    summarise_arm,
    write_shard,
)

# The header of results/option1_literature_summary.csv — what 3. Results Data
# Analysis.ipynb opens by name. Written out rather than derived from the
# constants above, so that a change to those constants FAILS here instead of
# agreeing with itself and breaking the downstream notebook silently.
NOTEBOOK3_COLUMNS = [
    "network", "method", "variant_index",
    "proportion_edges", "uncertainty", "n_experiments", "n_agents", "n_edges",
    "average_degree", "degree_gini_coefficient",
    "approx_average_clustering_coefficient", "variation_seed",
    "mean_share_of_correct_agents_at_convergence",
    "std_share_of_correct_agents_at_convergence",
    "mean_convergence_step", "std_convergence_step",
    "mean_proportion_reached_by_truth", "std_proportion_reached_by_truth",
    "n_runs",
]


@pytest.fixture
def arm(tmp_path):
    """An arm holding variants 0..4, three replicates each."""
    directory = tmp_path / "pud_equalize"
    directory.mkdir()
    rng = np.random.default_rng(20260826)
    for index in range(5):
        n = 3
        df = pd.DataFrame({c: np.repeat(rng.random(), n) for c in VARIANT_COVARIATES})
        for c in OUTCOMES:
            df[c] = rng.random(n)
        df.insert(0, "method", "equalize")
        df.insert(0, "network", "pud")
        df.insert(0, "variant_index", index)
        write_shard(df, shard_path(directory, index))
    return directory


def test_summary_columns_match_the_downstream_contract(arm):
    assert list(summarise_arm(arm).columns) == NOTEBOOK3_COLUMNS


def test_subset_summary_equals_the_same_rows_of_the_whole(arm):
    """The property that makes multi-session aggregation sound.

    Every aggregate is computed within a variant, so summarising {3, 4} alone
    must give exactly the rows summarising 0..4 gives for 3 and 4. If an
    across-variant aggregate were ever added, this fails — which is the point.
    """
    whole = summarise_arm(arm)
    part = summarise_arm(arm, variants=[3, 4])

    expected = whole[whole["variant_index"].isin([3, 4])].reset_index(drop=True)
    pd.testing.assert_frame_equal(part.reset_index(drop=True), expected)


def test_resuming_in_two_passes_reproduces_a_single_pass(arm):
    """Aggregate 0..2, bank two more variants, aggregate only those, concatenate."""
    first = summarise_arm(arm, variants=[0, 1, 2])
    second = summarise_arm(arm, variants=[3, 4])
    resumed = (
        pd.concat([first, second], ignore_index=True)
        .sort_values("variant_index")
        .reset_index(drop=True)
    )

    single = summarise_arm(arm).sort_values("variant_index").reset_index(drop=True)
    pd.testing.assert_frame_equal(resumed, single)


def test_variants_none_reads_the_whole_arm(arm):
    assert len(load_arm(arm)) == 15
    assert len(load_arm(arm, variants=None)) == 15


def test_unknown_variant_indices_are_ignored_not_fatal(arm):
    """A cache naming variants that are no longer on disk must not raise."""
    assert len(load_arm(arm, variants=[4, 99])) == 3
    assert summarise_arm(arm, variants=[99]).empty


def test_a_partial_arm_summarises_to_what_it_has(arm):
    """No target is consulted: an arm reports the variants it banked, not a goal."""
    for index in (3, 4):
        shard_path(arm, index).unlink()

    summary = summarise_arm(arm)
    assert completed_variants(arm) == {0, 1, 2}
    assert sorted(summary["variant_index"]) == [0, 1, 2]
    assert list(summary.columns) == NOTEBOOK3_COLUMNS


def test_cache_and_shard_helpers_ignore_non_shard_files(arm):
    """The aggregation cache lives in the arm directory and must stay invisible."""
    (arm / "arm_summary_cache.csv").write_text("network,method\\n")
    (arm / "cost.csv").write_text("network\\n")
    (arm / "variant_00009.parquet.tmp").write_text("")

    assert completed_variants(arm) == {0, 1, 2, 3, 4}
    assert len(load_arm(arm)) == 15
