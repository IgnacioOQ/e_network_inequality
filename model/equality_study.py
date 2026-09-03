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

Both arms are density-preserving, so density is constant across the whole study
and the manipulated dimension is degree equality:

===============  ==========================================  ====================
Arm              Mechanism                                   Invariant
===============  ==========================================  ====================
randomization    rewire k random edges (remove one/add one)  ``|E|``
equalize         rewire k triangle edges toward equality     ``|E|``
===============  ==========================================  ====================

The notebooks set ``INCLUDE_RANDOMIZATION = False``, so ``equalize`` is the only
arm that runs by default. Clustering arms once lived here and were removed; no
arm manipulates clustering any more.

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
import shutil
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import dill
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

# The two density-preserving arms. Density arms ('densify', 'densify_fixed')
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


def draw_setting_params(variation_seed, *, uncertainty, proportion_edges_max=0.1):
    """The parameter draw for one variant, without building its network.

    Split out of :func:`build_setting`, which draws exactly these values and only
    then spends seconds constructing the variant graph. Keeping the draw callable
    on its own is what makes :func:`parameter_coverage` possible: the parameters
    of a *pending* variant can be reported instantly, so the coverage table is a
    plan for the whole grid rather than a report on the part already finished.

    Seeds both RNGs, exactly as ``build_setting`` did, because the variation
    helpers it calls next consume the stdlib ``random`` stream. Callers that go
    on to build the variant therefore see an unchanged RNG state.
    """
    random.seed(variation_seed)
    rd.seed(variation_seed)

    proportion_edges = float(rd.rand() * proportion_edges_max)
    if isinstance(uncertainty, (tuple, list)):
        lo, hi = uncertainty
        unc = float(rd.uniform(lo, hi))
    else:
        unc = float(uncertainty)

    return {"uncertainty": unc, "proportion_edges": proportion_edges}


def build_setting(
    G,
    method,
    variation_seed,
    *,
    uncertainty,
    n_experiments,
    proportion_edges_max=0.1,
):
    """Draw one parameter setting and build its network variant.

    Deterministic given `variation_seed`. `uncertainty` is either a float (fixed
    — options 1 and 2) or a ``(lo, hi)`` pair drawn uniformly per setting
    (option 3, the phase-transition sweep).

    `proportion_edges` is the intensity knob shared by both arms: the fraction
    of edges rewired. The `proportion_edges_max` cap (0.1 in the notebooks)
    exists because `equalize` samples that many triangles and raises "Sample
    larger than population" beyond it.

    Returns the parameter dict consumed by
    :func:`run_vectorized_simulation_with_params`, carrying the variant under
    ``network`` plus the scalar covariates that the simulation copies into every
    result row.
    """
    drawn = draw_setting_params(
        variation_seed,
        uncertainty=uncertainty,
        proportion_edges_max=proportion_edges_max,
    )
    proportion_edges = drawn["proportion_edges"]
    unc = drawn["uncertainty"]

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
# Variant cache — phase A of the two-phase run
# ─────────────────────────────────────────────────────────────────────────────
#
# Variant construction is single-threaded, so in the one-phase run it leaves
# num_cores - 1 cores idle while it works. The two-phase run builds every variant
# of an arm up front, in parallel, and dumps each to a cache file; the replicate
# loop then loads the variant instead of building it. The cache is a pure
# function of the variation seeds (see docs/PARALLELIZATION_PLAN.md), so it is
# disposable: delete it and the next run rebuilds it. Keep it on fast local
# storage, never a Drive mount.

VARIANT_CACHE_EXT = ".pkl"


def variant_cache_dir(cache_root, network_label, method):
    """The cache directory for one arm — mirrors :func:`arm_dir`."""
    return Path(cache_root) / f"{network_label}_{method}"


def variant_cache_path(cache_root, network_label, method, variant_index):
    return (
        variant_cache_dir(cache_root, network_label, method)
        / f"variant_{variant_index:05d}{VARIANT_CACHE_EXT}"
    )


def completed_variant_caches(directory):
    """Variant indices whose cache file is on disk — the phase-A resume state.

    The mirror of :func:`completed_variants` (which is the phase-B / results
    resume state). A directory listing, never a read of the cached objects.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return set()
    done = set()
    for p in directory.iterdir():
        if p.name.startswith("variant_") and p.suffix == VARIANT_CACHE_EXT:
            try:
                done.add(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return done


def write_variant_cache(setting, path):
    """Dump one built ``setting`` dict atomically.

    Same tmp-then-``os.replace`` guard as :func:`write_shard`: a kill mid-write
    must not leave a truncated file that a later phase-A resume then trusts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        dill.dump(setting, f)
    os.replace(tmp, path)
    return path


def read_variant_cache(path):
    with open(path, "rb") as f:
        return dill.load(f)


_BUILDER = {}


def _init_builder(G, method, sequences, n_runs, uncertainty, n_experiments, build_kwargs, directory):
    _BUILDER.update(
        G=G,
        method=method,
        sequences=sequences,
        n_runs=n_runs,
        uncertainty=uncertainty,
        n_experiments=n_experiments,
        build_kwargs=build_kwargs,
        directory=Path(directory),
    )


def _build_one(i):
    # variant_seeds' variation seed is an independent child of the variant
    # sequence, so n_runs cannot perturb it; passed through only to match how
    # run_arm derives the same value.
    variation_seed, _ = variant_seeds(_BUILDER["sequences"][i], _BUILDER["n_runs"])
    t0 = time.time()
    setting = build_setting(
        _BUILDER["G"],
        _BUILDER["method"],
        variation_seed,
        uncertainty=_BUILDER["uncertainty"],
        n_experiments=_BUILDER["n_experiments"],
        **_BUILDER["build_kwargs"],
    )
    build_sec = time.time() - t0
    write_variant_cache(
        setting, _BUILDER["directory"] / f"variant_{i:05d}{VARIANT_CACHE_EXT}"
    )
    return i, build_sec


BUILD_COST_COLS = ["network", "method", "variant_index", "build_sec", "num_cores"]


def build_variants(
    networks,
    cache_root,
    *,
    methods=METHODS,
    skip=(),
    master_seed,
    n_variants,
    n_runs,
    uncertainty,
    n_experiments,
    num_cores,
    variant_slice=None,
    build_kwargs=None,
    progress=True,
    sim_kwargs=None,  # accepted and ignored, so a notebook can hand the same
    #                   kwargs dict to build_variants and run_study
):
    """Phase A: build every requested variant in parallel and cache it to disk.

    Enumerates ``(network, arm, variant)`` exactly as :func:`run_study` does, so
    the cache it writes lines up one-to-one with what :func:`run_arm` asks for.
    Resumable: a variant whose cache file already exists is left alone, so a
    rerun after a disconnect only builds what is missing. ``variant_slice``
    restricts the session to ``range(start, stop)`` just as in :func:`run_arm`.

    Returns a per-variant build-cost table. Passing the resulting ``cache_root``
    to :func:`run_study` as ``variant_cache_root`` makes phase B load instead of
    build; without that argument the study still runs, building inline as before.
    """
    build_kwargs = build_kwargs or {}
    skip = {tuple(s) for s in skip}
    cache_root = Path(cache_root)
    rows = []
    for label, G in networks:
        for method in methods:
            if (label, method) in skip:
                print(f"  [{label}/{method}] SKIPPED by configuration")
                continue
            directory = variant_cache_dir(cache_root, label, method)
            directory.mkdir(parents=True, exist_ok=True)

            sequences = variant_sequences(master_seed, label, method, n_variants)
            wanted = range(n_variants) if variant_slice is None else range(*variant_slice)
            done = completed_variant_caches(directory)
            todo = [i for i in wanted if i not in done]

            print(
                f"  [{label}/{method}] {len(wanted)} variants requested, "
                f"{len(done & set(wanted))} already cached, {len(todo)} to build"
            )
            if not todo:
                continue

            with Pool(
                num_cores,
                initializer=_init_builder,
                initargs=(G, method, sequences, n_runs, uncertainty, n_experiments, build_kwargs, directory),
            ) as pool:
                it = pool.imap_unordered(_build_one, todo)
                if progress:
                    it = tqdm(it, total=len(todo), desc=f"[{label}/{method}] build", unit="variant")
                for i, build_sec in it:
                    rows.append(
                        {
                            "network": label,
                            "method": method,
                            "variant_index": i,
                            "build_sec": build_sec,
                            "num_cores": num_cores,
                        }
                    )

    return pd.DataFrame(rows, columns=BUILD_COST_COLS)


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


def _variant_index(path):
    """The variant index encoded in a shard filename, or None if it is not one."""
    if not path.name.startswith("variant_") or path.suffix not in _SHARD_EXTS:
        return None
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return None


def load_arm(directory, variants=None):
    """Concatenate the shards in an arm directory.

    ``variants`` restricts the read to those variant indices. That is what makes
    aggregation resumable: a caller holding a summary of the variants it read
    last session can ask for only the ones banked since, instead of paying to
    re-read a million rows across a Drive mount to learn what it already knows.
    ``None`` (the default) reads the whole arm.
    """
    directory = Path(directory)
    indexed = [(i, p) for p in directory.iterdir() if (i := _variant_index(p)) is not None]
    if variants is not None:
        wanted = {int(v) for v in variants}
        indexed = [(i, p) for i, p in indexed if i in wanted]
    if not indexed:
        return pd.DataFrame()
    return pd.concat([read_shard(p) for _, p in sorted(indexed)], ignore_index=True)


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


# Which configuration keys mean "this is a different study" versus "this is the
# same study, carried further".
#
# The distinction is what makes a multi-session run possible at all. Banking 500
# variants today and 500 more tomorrow raises `n_variants`; if that counted as a
# different study, the first 500 would be thrown away. Because variant seeds are
# keyed on (network, arm, index) and SeedSequence.spawn is incremental,
# `variant_sequences(..., 1000)[:500]` is bit-identical to
# `variant_sequences(..., 500)` — so a grown grid genuinely extends the old one
# rather than reinterpreting it.
IDENTITY_KEYS = (
    "option",
    "master_seed",
    "smoke",
    "uncertainty",
    "n_experiments",
    "max_steps",
    "window",
    "min_steps",
    "proportion_edges_max",
    # n_runs is identity, not extent: raising it would leave already-banked
    # shards holding fewer replicates than new ones, making the study ragged and
    # every cross-variant mean silently weight-inconsistent. Topping shards up
    # is a separate feature, not a side effect of editing a parameter cell.
    "n_runs",
)

EXTENT_KEYS = ("n_variants", "methods", "skip_arms")


def _banked_units(results_dir):
    """How many variant shards actually exist under `results_dir`, across all arms.

    Counted by directory listing, so it stays cheap on a finished study.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return 0
    return sum(len(completed_variants(d)) for d in results_dir.iterdir() if d.is_dir())


def check_fingerprint(
    results_dir,
    config,
    *,
    accumulate=True,
    filename="equality_study_config.json",
):
    """Reconcile this session's configuration with the study already on disk.

    ``accumulate=True`` (the default) carries an existing study further. Every
    key in :data:`IDENTITY_KEYS` must match what was stamped; the keys in
    :data:`EXTENT_KEYS` may *grow*. Raising ``n_variants`` from 500 to 1000 adds
    variants 500..999 to the 500 already banked, and the earlier ones are left
    exactly as the first session built them.

    ``accumulate=False`` starts clean: the run-tag subtree is deleted and
    re-stamped.

    A conflict on an identity key **raises and deletes nothing**. The previous
    behaviour was to ``rmtree`` the whole study on any difference whatsoever,
    which turned a one-character edit in a parameter cell into the loss of a
    multi-day run.
    """
    results_dir = Path(results_dir)
    path = results_dir / filename
    current = {"schema": 1, **{k: _jsonable(v) for k, v in sorted(config.items())}}

    # ── ACCUMULATE=False — deliberate clean slate ────────────────────────────
    if not accumulate:
        if results_dir.exists():
            # Scoped hard: only ever a run-tag subtree, never its parent. A
            # mis-set RESULTS_DIR must not be able to take the study with it.
            if results_dir.name not in ("smoke", "full"):
                raise ValueError(
                    f"refusing to wipe {results_dir} — expected a 'smoke' or 'full' "
                    "directory. Check RESULTS_DIR before setting ACCUMULATE=False."
                )
            doomed = [p for p in results_dir.rglob("*") if p.is_file()]
            print(f"  ACCUMULATE=False — deleting {len(doomed)} file(s) under {results_dir}")
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2))
        print("  Fresh study stamped; this run starts from variant 0.")
        return current

    # ── First session for this configuration ─────────────────────────────────
    if not path.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2))
        print("  New study — configuration stamped.")
        return current

    saved = json.loads(path.read_text())

    # ── Identity keys must match, or we refuse (without touching anything) ───
    differing = [k for k in ("schema",) + IDENTITY_KEYS
                 if k in current and saved.get(k) != current.get(k)]
    if differing:
        lines = "\n".join(
            f"    {k}:  on disk {saved.get(k)!r}   ->   this session {current.get(k)!r}"
            for k in differing
        )
        # An earlier revision auto-re-stamped when no shards were found, on the
        # reasoning that a stamp with no work behind it protects nothing. That
        # is true, but it cannot be acted on: this count comes from a directory
        # listing, and a cloud-mounted filesystem (Drive under Colab) can report
        # an empty or partial listing for a directory holding thousands of
        # files. Acting on a false zero would silently adopt a conflicting
        # config over a completed study — the precise failure this guard exists
        # to prevent. So report the observation and let a human judge it.
        hint = ""
        if not _banked_units(results_dir):
            hint = (
                "\n  NOTE: no completed variants are visible here, so this stamp may be left over "
                "from a\n  session that died before finishing its first variant. Confirm with a "
                "directory listing\n  before trusting that — a cloud-mounted filesystem can report "
                "an empty or partial\n  listing for a directory that is not actually empty."
            )
        raise ValueError(
            "Configuration conflict — this session describes a different study "
            f"from the one already in\n  {results_dir}\n{lines}\n"
            "  NOTHING HAS BEEN DELETED. Either restore the values shown on disk, "
            "or set ACCUMULATE = False\n"
            "  to start this configuration from scratch (which discards that "
            f"directory).{hint}"
        )

    # ── Extent keys may grow ─────────────────────────────────────────────────
    merged, notes = dict(saved), []

    saved_nv, current_nv = saved.get("n_variants"), current.get("n_variants")
    if isinstance(saved_nv, int) and isinstance(current_nv, int):
        if current_nv > saved_nv:
            notes.append(
                f"n_variants {saved_nv} -> {current_nv}: "
                f"variants {saved_nv}..{current_nv - 1} are new this session"
            )
        elif current_nv < saved_nv:
            notes.append(
                f"n_variants {saved_nv} -> {current_nv} (LOWER). Shards for variants "
                f"{current_nv}..{saved_nv - 1} stay on disk and are STILL counted by "
                "summarise_arm/check_variance, which read every shard present."
            )
        merged["n_variants"] = max(saved_nv, current_nv)

    saved_methods = list(saved.get("methods") or [])
    current_methods = list(current.get("methods") or [])
    added = [m for m in current_methods if m not in saved_methods]
    dropped = [m for m in saved_methods if m not in current_methods]
    if added:
        notes.append(f"methods: added {added}")
    if dropped:
        notes.append(f"methods: {dropped} not run this session (their shards remain)")
    merged["methods"] = saved_methods + added
    merged["skip_arms"] = current.get("skip_arms", saved.get("skip_arms"))
    merged["schema"] = current["schema"]

    path.write_text(json.dumps(merged, indent=2))
    if notes:
        print("  ACCUMULATE=True — continuing the existing study:")
        for note in notes:
            print(f"    {note}")
    else:
        print("  ACCUMULATE=True — continuing the existing study (config unchanged).")
    return merged


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
    variant_cache_root=None,
    build_kwargs=None,
    progress=True,
):
    """Run one (network, arm) cell, checkpointing one shard per variant.

    `variant_slice` is a ``(start, stop)`` pair restricting this session to a
    bounded index range, so several machines can divide an arm between them
    without coordinating. Leave it ``None`` to take the whole arm.

    `variant_cache_root` points at a phase-A cache written by
    :func:`build_variants`. When set, a variant whose cache file exists is loaded
    rather than rebuilt (``build_sec`` then measures the load); a missing entry
    falls back to an inline build, so the argument is safe to pass before phase A
    has been run for every variant.

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
        setting = None
        if variant_cache_root is not None:
            cpath = variant_cache_path(variant_cache_root, network_label, method, i)
            if cpath.exists():
                setting = read_variant_cache(cpath)
        if setting is None:
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


def summarise_arm(directory, outcomes=OUTCOMES, variants=None):
    """One row per variant: its covariates plus aggregates over its replicates.

    This is the analysis-facing table. A finished option is twelve million
    replicate rows but only twelve thousand variant rows, and the variant is the
    unit at which the equality and clustering covariates actually vary.

    ``variants`` is passed through to :func:`load_arm`, summarising only those
    variant indices. Because every aggregate here is computed *within* a variant
    — the groupby key is ``(network, method, variant_index)`` — a subset
    produces exactly the rows it would have produced as part of the whole, so
    summaries built over several sessions concatenate without qualification.
    That would not hold for an aggregate taken across variants, and nothing here
    may become one.
    """
    df = load_arm(directory, variants=variants)
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


def study_report(results_dir, networks, methods, n_variants, n_runs, verbose=True):
    """Progress against the goal: simulations banked, time left, and done-or-not.

    The multi-session dashboard. Answers the question you actually have when
    reopening the notebook on day three — how far did earlier sessions get, how
    much is left, and is this finished.

    Reads only directory listings and each arm's small ``cost.csv``, never the
    shards, so it stays instant on a finished study holding millions of rows —
    which matters most on Colab, where every shard read crosses a Drive mount.
    Safe to run at any point: an arm with nothing on disk simply reports zero.

    ``sims_done`` is exact rather than estimated: :func:`run_arm` writes one
    shard per variant holding exactly ``n_runs`` rows, so banked variants times
    ``n_runs`` is the true simulation count.
    """
    results_dir = Path(results_dir)
    rows = []
    for label, _ in networks:
        for method in methods:
            directory = arm_dir(results_dir, label, method)
            done = completed_variants(directory)
            within = {i for i in done if i < n_variants}
            beyond = done - within
            left = n_variants - len(within)
            row = {
                "network": label,
                "method": method,
                "variants_done": len(within),
                "variants_target": n_variants,
                "sims_done": len(within) * n_runs,
                "sims_target": n_variants * n_runs,
                "pct_complete": round(100.0 * len(within) / n_variants, 1) if n_variants else 0.0,
                "variants_left": left,
                "sims_left": left * n_runs,
                # Shards above the current target, left by a session that ran a
                # larger grid. summarise_arm still counts them.
                "beyond_target": len(beyond),
                "sec_per_variant": np.nan,
                "hours_left": np.nan,
                "status": "complete" if left <= 0 else ("not started" if not within else "in progress"),
            }
            cost_path = directory / "cost.csv"
            if cost_path.exists():
                cost = pd.read_csv(cost_path)
                if len(cost):
                    per_variant = float((cost["build_sec"] + cost["run_sec"]).mean())
                    row["sec_per_variant"] = round(per_variant, 1)
                    row["hours_left"] = round(max(left, 0) * per_variant / 3600.0, 2)
            rows.append(row)

    report = pd.DataFrame(rows)
    if verbose and len(report):
        sims_done = int(report["sims_done"].sum())
        sims_target = int(report["sims_target"].sum())
        pct = 100.0 * sims_done / sims_target if sims_target else 0.0
        n_complete = int((report["status"] == "complete").sum())

        print(f"Progress report — {results_dir}")
        print(f"  {sims_done:,} / {sims_target:,} simulations  ({pct:.1f}% of goal)")
        print(f"  {int(report['variants_done'].sum()):,} / {int(report['variants_target'].sum()):,} "
              f"variants   |   arms complete: {n_complete}/{len(report)}")

        # A 40-cell text bar, so progress is legible without reading the table.
        filled = int(round(pct / 2.5))
        print(f"  [{'#' * filled}{'.' * (40 - filled)}]")

        hours = report["hours_left"].dropna()
        if n_complete == len(report):
            print("  STATUS: DONE — every arm has reached n_variants.")
        elif len(hours):
            total = float(hours.sum())
            covered = "all" if len(hours) == len(report) else f"{len(hours)}/{len(report)}"
            print(f"  STATUS: IN PROGRESS — ~{total:,.1f}h left ({covered} arm(s) timed)")
            if total > 11:
                print(f"          ~{total / 11:.1f} more Colab sessions at ~11h each. "
                      "Keep ACCUMULATE = True and re-run,")
                print("          or split the work with VARIANT_SLICE across machines.")
        else:
            print("  STATUS: IN PROGRESS — no timing data yet (no variant has finished).")

        if int(report["beyond_target"].sum()):
            print(f"  NOTE: {int(report['beyond_target'].sum())} shard(s) sit above the current "
                  "n_variants and are still counted by summarise_arm/check_variance.")
    return report


def parameter_coverage(
    results_dir,
    networks,
    methods,
    n_variants,
    *,
    master_seed,
    uncertainty,
    n_runs,
    proportion_edges_max=0.1,
    verbose=True,
):
    """Which parameter settings are banked, and which are still pending.

    One row per planned variant: its seed, the parameters it draws, and whether
    its shard is on disk. Derived from the seed sequences via
    :func:`draw_setting_params`, **not** from the shards — ``build_setting``
    draws its parameters before constructing the variant graph, so a pending
    variant's parameters are knowable without paying to build it. The table is
    therefore a plan for the entire grid, not a report on the finished part, and
    it costs one directory listing per arm.

    Use it to see whether the settings already banked span the intended range —
    a session stopped halfway, or a `variant_slice` claimed by another machine,
    both leave gaps that a bare count of completed variants would hide.
    """
    results_dir = Path(results_dir)
    rows = []
    for label, _ in networks:
        for method in methods:
            done = completed_variants(arm_dir(results_dir, label, method))
            sequences = variant_sequences(master_seed, label, method, n_variants)
            for i in range(n_variants):
                variation_seed, _ = variant_seeds(sequences[i], n_runs)
                drawn = draw_setting_params(
                    variation_seed,
                    uncertainty=uncertainty,
                    proportion_edges_max=proportion_edges_max,
                )
                rows.append({
                    "network": label,
                    "method": method,
                    "variant_index": i,
                    "variation_seed": variation_seed,
                    "uncertainty": drawn["uncertainty"],
                    "proportion_edges": drawn["proportion_edges"],
                    "done": i in done,
                })

    coverage = pd.DataFrame(rows)
    if verbose and len(coverage):
        swept = [
            c for c in ("uncertainty", "proportion_edges")
            if coverage[c].nunique() > 1
        ]
        print(f"Parameter coverage — {int(coverage['done'].sum()):,} of "
              f"{len(coverage):,} planned variants banked")
        if not swept:
            print("  (no parameter is swept in this option — every variant shares one setting)")
        for col in swept:
            done_vals = coverage.loc[coverage["done"], col]
            allv = coverage[col]
            print(f"  {col}: planned [{allv.min():.6g}, {allv.max():.6g}]")
            if len(done_vals):
                # Quartile occupancy of the planned range: a banked set that
                # fills only the low quartiles means the sweep is not yet
                # representative, however healthy the raw count looks.
                edges = np.linspace(allv.min(), allv.max(), 5)
                counts = [int(((done_vals >= edges[q]) & (done_vals <= edges[q + 1])).sum())
                          for q in range(4)]
                print(f"    banked [{done_vals.min():.6g}, {done_vals.max():.6g}]  "
                      f"quartile occupancy {counts}")
            else:
                print("    banked: none yet")

        pending = coverage.loc[~coverage["done"]]
        if len(pending):
            per_arm = pending.groupby(["network", "method"])["variant_index"]
            print(f"  {len(pending):,} pending; next index per arm: "
                  + ", ".join(f"{n}/{m}={int(v.min())}" for (n, m), v in per_arm))
    return coverage
