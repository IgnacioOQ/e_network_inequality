"""Runner for the equality / clustering parameter-variant studies.

The three option notebooks (``2a`` / ``2b`` / ``2c``) are deliberately thin: each
sets a configuration and calls into here. All of the machinery — deterministic
variant generation, the nested variant x replicate design, shard checkpointing,
resume, and the seeding-validity check — lives in this module so the three
notebooks cannot drift apart.

Design
------
The study is a fully-crossed grid over

    network  x  variation arm  x  network variant  x  replicate run

with ``n_variants`` distinct network variants per (network, arm) cell and
``n_runs`` replicate simulations per variant. Replicates share a parameter
setting and differ only in their simulation seed — which is exactly what makes
:func:`check_variance` a real test of the seeding rather than a formality.

Every arm is density-preserving, so density is constant across the whole study
and the manipulated dimensions are degree equality and clustering:

===============  ==========================================  ====================
Arm              Mechanism                                   Invariant
===============  ==========================================  ====================
randomization    rewire k random edges (remove one/add one)  ``|E|``
generate_equalize_variant rewire k triangle edges toward equality     ``|E|``
===============  ==========================================  ====================

Reproducibility
---------------
Everything is a pure function of ``master_seed``. Variant seeds are spawned from
a :class:`~numpy.random.SeedSequence` keyed on (network, arm, variant index), so
a session resuming at variant 457 rebuilds precisely the variant an earlier
session would have built. This is a hard requirement of the resume path, not a
nicety: the original notebook drew its variation seeds from ``os.urandom`` and
therefore could not re-derive a variant at all.

Both RNGs are seeded per variant. ``numpy.random`` drives the parameter draws;
the stdlib ``random`` drives every variation helper in
``networks.variation_methods`` and ``utils.network_utils``. Seeding only numpy
would leave the variant itself irreproducible.
"""

import hashlib
import json
import os
import random
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.random as rd
import pandas as pd
from numpy.random import SeedSequence
from tqdm.auto import tqdm

from model.vectorized_simulation_functions import run_vectorized_simulation_with_params
from networks.variation_methods import generate_network_variant, randomize_network, generate_equalize_variant
from utils.network_utils import network_statistics

# The four density-preserving arms. Density arms ('densify', 'densify_fixed')
# belong to "2. GColab Simulations.ipynb" and are deliberately absent here.
METHODS = ("randomization", "equalize")

# Shards are Parquet where pyarrow is available (Colab) and CSV otherwise, so a
# laptop smoke run works without adding a dependency. Resume tolerates both.
try:  # pragma: no cover - depends on the runtime, not on logic
    import pyarrow  # noqa: F401

    SHARD_EXT = ".parquet"
except ImportError:  # pragma: no cover
    SHARD_EXT = ".csv"

_SHARD_EXTS = (".parquet", ".csv")


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────


def _stable_key(text):
    """A process-stable 32-bit int for `text`.

    Python's built-in ``hash()`` is salted per interpreter, so it cannot key a
    seed sequence that must reproduce across sessions and machines.
    """
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "little")


def variant_sequences(master_seed, network_label, method, n_variants):
    """One :class:`SeedSequence` per variant, keyed on (network, arm, index).

    Keying on the labels rather than on call order means the PUD/randomization
    cell yields the same variants whether or not the Tobacco cell ran first, and
    an arm can be re-run in isolation without disturbing any other.
    """
    root = SeedSequence(
        master_seed, spawn_key=(_stable_key(network_label), _stable_key(method))
    )
    return root.spawn(n_variants)


def variant_seeds(variant_sequence, n_runs):
    """Split one variant's sequence into (variation seed, replicate seeds).

    Two independent children rather than one stream, so changing ``n_runs``
    cannot perturb the variation seed and thus the network variant itself.
    """
    var_ss, run_ss = variant_sequence.spawn(2)
    variation_seed = int(var_ss.generate_state(1)[0])
    replicate_seeds = [int(s.generate_state(1)[0]) for s in run_ss.spawn(n_runs)]
    return variation_seed, replicate_seeds


# ─────────────────────────────────────────────────────────────────────────────
# Variant construction
# ─────────────────────────────────────────────────────────────────────────────


def build_setting(
    G,
    method,
    variation_seed,
    *,
    uncertainty,
    n_experiments,
    proportion_edges_max=0.1,
    rewiring_tolerance=1e-3,
    max_post_rewire_factor=10,
):
    """Draw one parameter setting and build its network variant.

    Deterministic given `variation_seed`. `uncertainty` is either a float (fixed
    — options 1 and 2) or a ``(lo, hi)`` pair drawn uniformly per setting
    (option 3, the phase-transition sweep).

    `proportion_edges` is the intensity knob shared by all arms: the
    fraction of edges rewired for randomization/equalize. The
    1/3 cap exists because `equalize` samples that many triangles and raises
    "Sample larger than population" beyond it.

    Returns the parameter dict consumed by
    :func:`run_vectorized_simulation_with_params`, carrying the variant under
    ``network`` plus the scalar covariates that the simulation copies into every
    result row.
    """
    random.seed(variation_seed)
    rd.seed(variation_seed)

    proportion_edges = float(rd.rand() * proportion_edges_max)
    if isinstance(uncertainty, (tuple, list)):
        lo, hi = uncertainty
        unc = float(rd.uniform(lo, hi))
    else:
        unc = float(uncertainty)

    n_edges = G.number_of_edges()
    if method == "randomization":
        variant = randomize_network(G, n_edges=int(n_edges * proportion_edges))
    elif method == "equalize":
        variant = generate_equalize_variant(G, n_edges=int(n_edges * proportion_edges))[0]
    else:
        raise ValueError(f"unknown variation method: {method!r} (expected one of {METHODS})")

    setting = {
        "network": variant,
        "n_experiments": int(n_experiments),
        "uncertainty": unc,
        "proportion_edges": proportion_edges,
        "variation_seed": int(variation_seed),
        "n_agents": int(variant.number_of_nodes()),
        "n_edges": int(variant.number_of_edges()),
    }
    # Cast explicitly: numpy scalars other than float64 do not pass the
    # isinstance((int, float, ...)) filter that decides which parameters make it
    # into a result row, so an un-cast np.int64 would silently vanish.
    setting.update({k: float(v) for k, v in network_statistics(variant).items()})
    return setting


# ─────────────────────────────────────────────────────────────────────────────
# Replicate execution
# ─────────────────────────────────────────────────────────────────────────────

_WORKER = {}


def _init_worker(setting, sim_kwargs):
    _WORKER["setting"] = setting
    _WORKER["sim_kwargs"] = sim_kwargs


def _run_one(seed):
    params = dict(_WORKER["setting"])
    params["seed"] = int(seed)
    return run_vectorized_simulation_with_params(params, **_WORKER["sim_kwargs"])


def run_setting(setting, replicate_seeds, sim_kwargs, num_cores, progress=False):
    """Run every replicate of one setting; one row per run.

    The variant is handed to the pool **once**, through the initializer, rather
    than riding inside every job's parameter dict. At 1,000 replicates that is
    the difference between pickling the graph once per worker and once per run —
    on the Ego network, across the full study, hundreds of gigabytes of pure
    serialisation.
    """
    with Pool(num_cores, initializer=_init_worker, initargs=(setting, sim_kwargs)) as pool:
        it = pool.imap_unordered(_run_one, [int(s) for s in replicate_seeds])
        if progress:
            it = tqdm(it, total=len(replicate_seeds), leave=False, desc="    runs")
        rows = list(it)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Shard I/O and resume
# ─────────────────────────────────────────────────────────────────────────────


def arm_dir(results_dir, network_label, method):
    return Path(results_dir) / f"{network_label}_{method}"


def shard_path(directory, variant_index, ext=SHARD_EXT):
    return Path(directory) / f"variant_{variant_index:05d}{ext}"


def write_shard(df, path):
    """Write one variant's replicates atomically.

    A Colab disconnect mid-write must not leave a truncated shard that resume
    then trusts, so the shard is written to a temporary name and renamed into
    place — ``os.replace`` is atomic within a filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_shard(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def completed_variants(directory):
    """Variant indices already on disk — the whole of the resume state.

    Resume costs a directory listing, never a read of everything computed so
    far. That matters: a finished arm holds a million rows.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return set()
    done = set()
    for p in directory.iterdir():
        if p.name.startswith("variant_") and p.suffix in _SHARD_EXTS:
            try:
                done.add(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return done


def load_arm(directory):
    """Concatenate every shard in an arm directory."""
    directory = Path(directory)
    paths = sorted(
        p for p in directory.iterdir() if p.name.startswith("variant_") and p.suffix in _SHARD_EXTS
    )
    if not paths:
        return pd.DataFrame()
    return pd.concat([read_shard(p) for p in paths], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config fingerprint
# ─────────────────────────────────────────────────────────────────────────────


def _jsonable(value):
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def check_fingerprint(results_dir, config, filename="equality_study_config.json"):
    """Stamp the configuration on first use; refuse to resume when it changed.

    Two runs whose fingerprints differ must never share a checkpoint. Without
    this a smoke run and a full run silently merge: the smoke variants already
    exist, every variant is skipped, zero simulations execute, and the summaries
    are then built from the other run's data under this run's label.
    """
    path = Path(results_dir) / filename
    current = {"schema": 1, **{k: _jsonable(v) for k, v in sorted(config.items())}}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != current:
            raise RuntimeError(
                "Checkpoint config mismatch — refusing to merge runs.\n"
                f"  dir:     {results_dir}\n"
                f"  saved:   {saved}\n"
                f"  current: {current}\n"
                "Point RESULTS_DIR at a fresh directory, or move the existing "
                "checkpoint aside before re-running."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2))
    return current


# ─────────────────────────────────────────────────────────────────────────────
# The arm runner
# ─────────────────────────────────────────────────────────────────────────────

COST_COLS = [
    "network",
    "method",
    "variant_index",
    "n_runs",
    "build_sec",
    "run_sec",
    "sec_per_run",
    "total_steps",
    "num_cores",
]


def run_arm(
    G,
    network_label,
    method,
    results_dir,
    *,
    master_seed,
    n_variants,
    n_runs,
    uncertainty,
    n_experiments,
    sim_kwargs,
    num_cores,
    variant_slice=None,
    build_kwargs=None,
    progress=True,
):
    """Run one (network, arm) cell, checkpointing one shard per variant.

    `variant_slice` is a ``(start, stop)`` pair restricting this session to a
    bounded index range, so several machines can divide an arm between them
    without coordinating. Leave it ``None`` to take the whole arm.

    Returns the per-variant cost table. The replicate rows themselves stay on
    disk — a finished arm is a million rows and does not belong in a notebook
    variable.
    """
    build_kwargs = build_kwargs or {}
    directory = arm_dir(results_dir, network_label, method)
    directory.mkdir(parents=True, exist_ok=True)

    sequences = variant_sequences(master_seed, network_label, method, n_variants)
    wanted = range(n_variants) if variant_slice is None else range(*variant_slice)
    done = completed_variants(directory)
    todo = [i for i in wanted if i not in done]

    cost_path = directory / "cost.csv"
    cost_rows = pd.read_csv(cost_path).to_dict("records") if cost_path.exists() else []

    print(
        f"  [{network_label}/{method}] {len(wanted)} variants requested, "
        f"{len(done & set(wanted))} already on disk, {len(todo)} to run "
        f"({len(todo) * n_runs:,} simulations)"
    )
    if not todo:
        return pd.DataFrame(cost_rows, columns=COST_COLS)

    bar = tqdm(todo, desc=f"[{network_label}/{method}]", unit="variant", disable=not progress)
    for i in bar:
        variation_seed, replicate_seeds = variant_seeds(sequences[i], n_runs)

        t0 = time.time()
        setting = build_setting(
            G,
            method,
            variation_seed,
            uncertainty=uncertainty,
            n_experiments=n_experiments,
            **build_kwargs,
        )
        build_sec = time.time() - t0

        t1 = time.time()
        df = run_setting(setting, replicate_seeds, sim_kwargs, num_cores, progress=progress)
        run_sec = time.time() - t1

        df.insert(0, "method", method)
        df.insert(0, "network", network_label)
        df.insert(0, "variant_index", i)
        write_shard(df, shard_path(directory, i))

        total_steps = int(df["convergence_step"].sum()) if "convergence_step" in df else 0
        cost_rows.append(
            {
                "network": network_label,
                "method": method,
                "variant_index": i,
                "n_runs": n_runs,
                "build_sec": build_sec,
                "run_sec": run_sec,
                "sec_per_run": run_sec / max(n_runs, 1),
                "total_steps": total_steps,
                "num_cores": num_cores,
            }
        )
        pd.DataFrame(cost_rows, columns=COST_COLS).to_csv(cost_path, index=False)
        bar.set_postfix(build=f"{build_sec:.1f}s", run=f"{run_sec:.1f}s")

    return pd.DataFrame(cost_rows, columns=COST_COLS)


def run_study(
    networks,
    results_dir,
    *,
    methods=METHODS,
    skip=(),
    continue_on_error=True,
    **kwargs,
):
    """Run every (network, arm) cell, isolating per-arm failures.

    `networks` is a list of ``(label, graph)`` pairs. `skip` holds
    ``(network_label, method)`` pairs to leave out.

    With ``continue_on_error`` (the default) a raising arm does not abort the
    study: the exception is caught, recorded to ``failed_arms.json`` in
    ``results_dir`` with its traceback, and the remaining arms still run. Arms
    finished before the failure are already checkpointed per variant, so nothing
    done is lost and a rerun resumes past them. This matters because some arms
    are known-fragile — Ego/equalize is flagged unreliable by TODO_WORKFLOW —
    and one bad arm should not forfeit a multi-day run of the others. Pass
    ``continue_on_error=False`` to let the first failure propagate instead.

    ``failed_arms.json`` is written on every run (an empty list means no arm
    failed), so its contents are never stale.
    """
    skip = {tuple(s) for s in skip}
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    costs, failures = [], []
    for label, G in networks:
        for method in methods:
            if (label, method) in skip:
                print(f"  [{label}/{method}] SKIPPED by configuration")
                continue
            try:
                costs.append(run_arm(G, label, method, results_dir, **kwargs))
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append({
                    "network": label,
                    "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                print(f"\n  !!! [{label}/{method}] FAILED — isolated, continuing "
                      "with the remaining arms.")
                print(f"      {type(exc).__name__}: {exc}")
                print("      Arms finished before this point are checkpointed; "
                      "a rerun resumes past them.")

    (results_dir / "failed_arms.json").write_text(json.dumps(failures, indent=2))
    if failures:
        print(f"\n  ⚠ {len(failures)} arm(s) FAILED this run and were skipped: "
              f"{[(f['network'], f['method']) for f in failures]}")
        print(f"    Details + tracebacks: {results_dir / 'failed_arms.json'}")

    return pd.concat(costs, ignore_index=True) if costs else pd.DataFrame(columns=COST_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# Validation and summarisation
# ─────────────────────────────────────────────────────────────────────────────

OUTCOMES = (
    "share_of_correct_agents_at_convergence",
    "convergence_step",
    "proportion_reached_by_truth",
)


def check_variance(directory, outcomes=OUTCOMES, verbose=True):
    """The seeding-validity check: does each variant's replicate set actually vary?

    For every variant this asserts two things:

    1. its replicate seeds are all distinct, and
    2. at least one outcome varies across those replicates.

    A variant whose replicates are identical is the signature of the
    fork-inherited-RNG bug — every worker drawing the same stream — which
    presents as unanimous agreement and zero variance rather than as a crash,
    and is therefore easy to publish by accident. Reported loudly, never
    silently.

    Returns a per-variant DataFrame; the caller decides whether to raise.
    """
    df = load_arm(directory)
    if df.empty:
        print(f"  [{Path(directory).name}] no shards — nothing to check")
        return pd.DataFrame()

    present = [c for c in outcomes if c in df.columns]
    rows = []
    for idx, group in df.groupby("variant_index"):
        n = len(group)
        n_unique_seeds = group["seed"].nunique() if "seed" in group else np.nan
        varies = {c: float(group[c].std(ddof=0)) for c in present}
        rows.append(
            {
                "variant_index": idx,
                "n_runs": n,
                "n_unique_seeds": n_unique_seeds,
                "seeds_distinct": bool(n_unique_seeds == n),
                **{f"std_{c}": v for c, v in varies.items()},
                "any_variance": any(v > 0 for v in varies.values()),
            }
        )
    report = pd.DataFrame(rows)

    bad_seeds = report[~report["seeds_distinct"]]
    bad_var = report[~report["any_variance"]]
    if verbose:
        name = Path(directory).name
        print(f"  [{name}] {len(report)} variants x {int(report['n_runs'].median())} replicates")
        if len(bad_seeds):
            print(
                f"  [{name}] FAIL — {len(bad_seeds)} variant(s) have duplicate replicate seeds: "
                f"{bad_seeds['variant_index'].tolist()[:10]}"
            )
        if len(bad_var):
            print(
                f"  [{name}] FAIL — {len(bad_var)} variant(s) show ZERO variance across "
                f"replicates: {bad_var['variant_index'].tolist()[:10]}"
            )
        if not len(bad_seeds) and not len(bad_var):
            print(f"  [{name}] OK — seeds distinct and every variant varies across replicates")
    return report


VARIANT_COVARIATES = (
    "proportion_edges",
    "uncertainty",
    "n_experiments",
    "n_agents",
    "n_edges",
    "average_degree",
    "degree_gini_coefficient",
    "approx_average_clustering_coefficient",
    "variation_seed",
)


def summarise_arm(directory, outcomes=OUTCOMES):
    """One row per variant: its covariates plus aggregates over its replicates.

    This is the analysis-facing table. A finished option is twelve million
    replicate rows but only twelve thousand variant rows, and the variant is the
    unit at which the equality and clustering covariates actually vary.
    """
    df = load_arm(directory)
    if df.empty:
        return pd.DataFrame()

    keys = ["network", "method", "variant_index"]
    covariates = [c for c in VARIANT_COVARIATES if c in df.columns]
    present = [c for c in outcomes if c in df.columns]

    # Named aggregation keeps the result flat. A dict-of-lists agg would return
    # a MultiIndex that then needs flattening by hand, which is where the
    # covariate names pick up spurious 'first_' prefixes.
    named = {c: pd.NamedAgg(column=c, aggfunc="first") for c in covariates}
    for c in present:
        named[f"mean_{c}"] = pd.NamedAgg(column=c, aggfunc="mean")
        named[f"std_{c}"] = pd.NamedAgg(column=c, aggfunc="std")
    named["n_runs"] = pd.NamedAgg(column="variant_index", aggfunc="size")

    return df.groupby(keys, as_index=False).agg(**named)


def plot_variant_summary(
    summary,
    target="mean_share_of_correct_agents_at_convergence",
    predictors=(
        "degree_gini_coefficient",
        "approx_average_clustering_coefficient",
        "proportion_edges",
        "average_degree",
    ),
    title=None,
    save_path=None,
):
    """Scatter the variant-level target against each structural predictor.

    Deliberately not ``utils.network_utils.scatter_plot``: that one plots every
    numeric column, which here would include seeds and indices. The unit is the
    variant, so one point is a variant's mean over its replicates.
    """
    predictors = [p for p in predictors if p in summary.columns]
    if not predictors or target not in summary.columns:
        print("  nothing to plot — missing target or predictors")
        return

    ncols = 2
    nrows = (len(predictors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, predictor in zip(axes, predictors):
        ax.scatter(summary[predictor], summary[target], alpha=0.45, s=14)
        ax.set_xlabel(predictor)
        ax.set_ylabel(target)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(predictors) :]:
        fig.delaxes(ax)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def project_runtime(cost, n_variants, n_runs):
    """Extrapolate an arm's remaining wall-clock from the variants already done."""
    if cost.empty:
        return pd.DataFrame()
    per = cost.groupby(["network", "method"]).agg(
        variants_done=("variant_index", "count"),
        mean_build_sec=("build_sec", "mean"),
        mean_run_sec=("run_sec", "mean"),
        sec_per_run=("sec_per_run", "mean"),
    )
    per["variants_left"] = n_variants - per["variants_done"]
    per["projected_hours_left"] = (
        per["variants_left"] * (per["mean_build_sec"] + per["mean_run_sec"]) / 3600.0
    )
    per["projected_total_hours"] = (
        n_variants * (per["mean_build_sec"] + per["mean_run_sec"]) / 3600.0
    )
    return per.reset_index()
