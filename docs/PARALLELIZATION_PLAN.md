# Variant generation & simulation parallelization — plan

**Status:** Option A implemented (2026-09-03). Step 0 (measure) was skipped by
decision; the two-phase cache is opt-in and falls back to the old inline build,
so it can land before the build-vs-run ratio is known.
**Owner:** Ignacio. **Last updated:** 2026-09-03.
**Related:** [`CLUSTER_PIPELINE_PLAN.md`](CLUSTER_PIPELINE_PLAN.md) — a two-phase
design composes with the cluster array-job split.

## Purpose

In the equality-study notebooks (`2a`–`2c`), each `(network, method)` arm runs a
loop over network variants. Within each iteration:

- **Variant generation** (`build_setting`) is **serial** — one core.
- **Replicate simulation** (`run_setting`) is **parallel** — `Pool(num_cores)`
  over `n_runs` replicate seeds.

So `num_cores - 1` cores sit idle during every build. This document works out
whether to parallelize variant generation, and if so how, given that all the
variant objects cannot be held in RAM at once.

## Current sequence

All logic is in [`model/equality_study.py`](../model/equality_study.py).
`run_arm` (one `(network, method)` cell):

```text
sequences = variant_sequences(master_seed, network_label, method, n_variants)
wanted    = range(n_variants)   or   range(*variant_slice)
todo      = [i in wanted if variant_i shard not already on disk]   # resume

for i in todo:                          # ── SERIAL loop over variants ──
    variation_seed, replicate_seeds = variant_seeds(sequences[i], n_runs)

    setting = build_setting(G, method, variation_seed, ...)        # ← SERIAL, 1 core
        ├─ generate_equalize_variant / randomize_network          #   Python rewiring loop
        │     (networks/variation_methods.py: while n_edges_added
        │      < n_edges, per-edge nx.clustering updates)
        └─ network_statistics(variant)                            #   nx.clustering + gini
        │                                                         #   + left-eigenvector solve
    build_sec = ...

    df = run_setting(setting, replicate_seeds, sim_kwargs,        # ← PARALLEL
                     num_cores, ...)                              #   Pool(num_cores) over n_runs,
        # graph handed to the pool ONCE via initializer,          #   fresh pool per variant
        # not per job (see run_setting docstring)
    run_sec = ...

    write_shard(df, shard_path(directory, i))                     # one parquet/csv per variant
    append row to cost.csv (build_sec, run_sec, sec_per_run, ...)
```

### What is already true and must be preserved

- **Determinism.** Every variant is a pure function of its `variation_seed`
  (keyed on `(network, method, variant_index)` via `variant_sequences`).
  `build_setting → draw_setting_params` reseeds both `random` and
  `numpy.random` from that seed before building. Variant *i* is bit-identical
  regardless of which process builds it or in what order — so parallelizing
  **across variants** needs no new seed plumbing.
- **Resume = a directory listing.** `completed_variants(directory)` is the whole
  resume state; a finished arm (≈1M rows) is never re-read to resume.
- **Atomic writes.** `write_shard` writes `*.tmp` then `os.replace` — safe under
  a mid-write kill and safe for multiple machines writing one filesystem.
- **`variant_slice`** restricts a session to `range(start, stop)`, so several
  machines divide an arm without coordinating.
- **`check_fingerprint`** separates identity keys (a mismatch = a different
  study, raises) from extent keys (`n_variants`, `methods` — may grow).
- **`cost.csv`** already records `build_sec` and `run_sec` per variant, in every
  arm directory, on Drive. **The data to decide this already exists.**

## The problem, precisely

Per variant, wall time is `build_sec` (1 core busy, `num_cores - 1` idle) then
`run_sec` (`num_cores` busy). The wasted resource is
`build_sec × (num_cores - 1)` core-seconds per variant. Whether that is worth
removing depends entirely on the ratio `build_sec / (build_sec + run_sec)`,
which varies by arm — `equalize` on the Ego network at high `proportion_edges`
is the expected worst case (the rewiring loop and `nx.clustering` both scale
with graph size and rewire count).

The obvious fix — a `Pool` over variant indices — needs somewhere to put the
built variants, because **all `n_variants` graph objects do not fit in RAM**
(hundreds of heavy `networkx` objects per arm). Hence an I/O hand-off: generate
in parallel, dump to disk, then load and simulate.

## Framing points

1. **The variant dump is a regenerable *cache*, not a checkpoint.** Each variant
   is a pure function of `variation_seed`. A lost cache entry is rebuilt, not
   lost data. Consequences:
   - It does **not** need Google Drive durability. It belongs on the *fastest*
     storage available: `/content` on Colab, local disk locally, node scratch on
     the cluster.
   - Only the **result shards** need to reach Drive.
   - Putting thousands of heavy graph files on a Drive mount and reading them
     back would likely cost more (mount latency, eventual-consistency lag) than
     the serial build costs today.

2. **Measure before building.** Read the existing `cost.csv` files, compute
   `build_sec / (build_sec + run_sec)` per arm and the total idle core-seconds.
   - < ~5 % → not worth the machinery; stop here.
   - ~5–15 % → prefer option **B** (prefetch), which needs no bulk disk.
   - \> ~15 % → option **A** (two-phase) is clearly justified.

3. **Parallel build is already determinism-safe** (see above) — no seed work.

4. **RAM is bounded by pool size, not variant count.** Phase A is
   `imap_unordered` over indices; each worker builds one variant, writes it,
   returns only a path + stats. Parent never holds more than a few. Peak RAM
   ≈ `k × (one variant + build scratch)`.

## Design options

| Option | Mechanism | Trade-off |
|---|---|---|
| **A. Two-phase generate → simulate** | Phase A: `Pool` over variant indices → dump each `setting` (graph + stats + drawn scalars) to a seed-keyed cache file on fast storage. Phase B: current `run_arm` loop, but `build_setting` replaced by a cache load. | Full answer; composes with the cluster split. Costs: a serialization format, extra disk, a second resume set, more moving parts. |
| **B. Prefetch / overlap** | Keep one phase. A background process builds variant *i+1* into a depth-1–2 queue while `run_setting` simulates variant *i*. Hides build latency behind run latency. | Small change, no bulk disk. Only helps while `build_sec < run_sec` per variant; adds concurrency logic to `run_arm`. |
| **C. Split phase A / phase B as separate cluster jobs** | Extension of A: a generate array job fills the cache, a simulate array job consumes it (`--dependency=afterok`). | Only meaningful once the cluster path is real. Lets generation and simulation use different node/résource profiles. |

A and B are not exclusive: B is a cheaper first step that captures most of the
benefit when build is the minority of wall time; A is the fuller answer and the
one that composes with `CLUSTER_PIPELINE_PLAN.md`.

## Open decisions (settle before implementing A)

- **D1. Serialization format for a cached variant.** Pickled `nx.Graph` is heavy
  and version-fragile. The vectorized model indexes a positional adjacency
  matrix anyway, so candidates: an edge list, or a sparse `.npz`, plus a small
  JSON sidecar carrying everything else `build_setting` returns in `setting`
  (`network_statistics` output, `uncertainty`, `proportion_edges`,
  `variation_seed`, `n_agents`, `n_edges`). Decide format + whether graph and
  metadata are one file or two.
  _Decision:_

- **D2. Cache location and key.** Fast/ephemeral storage; path convention;
  key = `variation_seed` or `(network, method, variant_index)`. Needs its own
  `completed_variant_caches()` resume set, distinct from `completed_variants()`
  (which stays the results resume set).
  _Decision:_

- **D3. Cache lifecycle.** Is it deleted after phase B consumes it, kept for the
  session, or kept across sessions? Kept-across-sessions turns a re-run into
  "skip build too", at the cost of disk. On ephemeral storage it is gone anyway.
  _Decision:_

- **D4. `variant_slice` interaction.** Phase A generates `[start, stop)` in
  parallel; phase B consumes the same range. Confirm both phases take the slice
  and that a partial phase A (some machine died) is just a smaller
  `completed_variant_caches()` set that a rerun tops up.
  _Decision:_

- **D5. Phase B pool churn (separate latent optimization).** Today a `Pool` is
  created and torn down once per variant. A persistent pool for phase B removes
  ~`n_variants` fork+import cycles. Worth doing alongside, or leave for later?
  _Decision:_

- **D6. Nested parallelism guard.** Keep phase A parallel-over-variants and
  phase B parallel-over-replicates / serial-over-variants. Do **not** parallelize
  both dimensions simultaneously (nested `Pool`s oversubscribe cores).
  _Decision:_

- **D7. `network_statistics` cost.** Confirm whether the expensive part of build
  is the rewiring loop or `network_statistics` (its `nx.clustering` +
  left-eigenvector solve). If it is the stats, that is independently
  parallelizable and might be the whole fix. `cost.csv` only has a combined
  `build_sec`; a one-off timing split is needed.
  _Decision:_

## Staged plan

Step 0 does not touch the pipeline and decides A vs B vs "do nothing".

0. **Measure.** Script (below) reads every `cost.csv` under a results tree and
   reports, per arm and overall: median `build_sec`, median `run_sec`, the
   `build / (build + run)` ratio, and total idle core-seconds
   (`Σ build_sec × (num_cores - 1)`). Run it against the Drive results tree.
1. **Decide.** Ratio < 5 % → close this plan. 5–15 % → option B. > 15 % →
   option A. Record the numbers and the choice in the Decision log.
2. **(Option A) Settle D1–D4** in this document.
3. **(Option A) Implement phase A** — `build_variants(...)` in
   `equality_study.py`: `Pool` over `todo` indices, worker builds via the
   existing `build_setting`, writes the cache artifact, returns
   `(index, path, build_sec)`. Its own resume set. Unit test: cached variant
   loads back bit-identical to a fresh `build_setting`.
4. **(Option A) Rewire `run_arm`** to load from cache instead of calling
   `build_setting`, falling back to an inline build if the cache entry is
   missing (so the change is safe to land before phase A always runs first).
5. **(Option A) Notebook wiring** — a phase-A cell before the run cell, gated by
   the same env/`SMOKE_TEST`/`VARIANT_SLICE` machinery. No new user-facing
   parameters beyond a cache directory.
6. **Verify** — a smoke run with phase A on/off produces byte-identical result
   shards; `2d` aggregation unaffected.
7. **(Option B instead)** — add a depth-1 prefetch to `run_arm`: a single
   `concurrent.futures.ProcessPoolExecutor(max_workers=1)` building the next
   variant while the current one simulates. No format decisions, no notebook
   changes. Same byte-identical verification.

### Step 0 script (to add as `scripts/analyze_build_cost.py` when we start)

```python
"""Read every cost.csv under a results tree; report build-vs-run cost.

Usage: python scripts/analyze_build_cost.py <results_dir>
Decides whether variant generation is worth parallelizing (see
docs/PARALLELIZATION_PLAN.md).
"""
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
frames = []
for p in root.rglob("cost.csv"):
    df = pd.read_csv(p)
    df["arm"] = p.parent.name
    frames.append(df)

if not frames:
    sys.exit(f"no cost.csv found under {root}")

cost = pd.concat(frames, ignore_index=True)
cost["ratio"] = cost["build_sec"] / (cost["build_sec"] + cost["run_sec"])
cost["idle_core_sec"] = cost["build_sec"] * (cost["num_cores"] - 1)

per_arm = cost.groupby("arm").agg(
    n_variants=("variant_index", "count"),
    build_med=("build_sec", "median"),
    run_med=("run_sec", "median"),
    ratio_med=("ratio", "median"),
    idle_core_hours=("idle_core_sec", lambda s: s.sum() / 3600),
)
print(per_arm.to_string(float_format=lambda x: f"{x:,.3f}"))
print(f"\noverall build/(build+run) median: {cost['ratio'].median():.3f}")
print(f"overall idle core-hours:          {cost['idle_core_sec'].sum() / 3600:,.1f}")
```

## What landed (2026-09-03) — Option A

`model/equality_study.py`:

- `build_variants(networks, cache_root, ...)` — phase A. Enumerates
  `(network, arm, variant)` exactly as `run_study`, builds each variant in a
  `Pool` (initializer stashes the base graph, mirroring `run_setting`), dumps the
  full `setting` dict with `dill` via an atomic tmp+`os.replace` write. Resumable
  through `completed_variant_caches()`; honours `variant_slice`. Returns a
  `build_sec` table. Accepts and ignores `sim_kwargs` so one kwargs dict serves
  both phases.
- `run_arm(..., variant_cache_root=None)` — when set, a variant whose cache file
  exists is loaded instead of built; a missing entry falls back to an inline
  build. `run_study` passes the argument straight through.
- Cache helpers: `variant_cache_dir`, `variant_cache_path`,
  `completed_variant_caches`, `write_variant_cache`, `read_variant_cache`,
  `VARIANT_CACHE_EXT = ".pkl"`.

`unit_tests/test_equality_study_variant_cache.py` — one cache file per variant
and the resume set finds them; a cached variant is bit-identical to a fresh
`build_setting`; resumable; `variant_slice` respected; and (slow) a shard built
via the cache equals the shard built without it.

Notebooks `2a` / `2b` / `2c`:

- `imports` — add `build_variants`.
- `config` — append `STUDY_KWARGS`, the one dict both phases consume (so they
  cannot enumerate the grid differently).
- `paths` — add `VARIANT_CACHE_DIR`: `.variant_cache/<option>/<run_tag>/`
  locally, `/content/variant_cache/...` on Colab. **Never Drive.**
- New `# Generate variants (phase A)` cell (`build-md` / `build-variants`) before
  the run cell.
- `run` — call becomes `run_study(NETWORKS, RESULTS_DIR,
  variant_cache_root=VARIANT_CACHE_DIR, **STUDY_KWARGS)`.

`.gitignore` — `.variant_cache/`.

### Not done / deferred

- **Step 0 (measure).** Still worth running `scripts/analyze_build_cost.py`
  (source above) against the Drive `cost.csv` files to quantify the speed-up and
  confirm phase A is worth its disk. `build_sec` in `run_arm` now reads ~0 on a
  cache hit, so the *new* `cost.csv` files no longer carry the real build cost —
  measure from a pre-2026-09-03 tree, or from the phase-A `build_cost` table.
- **D5 (persistent phase-B pool).** Untouched — still one `Pool` per variant.
- **D7 (`network_statistics` share of build).** Not separately profiled.
- **Cluster split (option C).** Composes with `CLUSTER_PIPELINE_PLAN.md` but not
  wired.

## Decision log

- _2026-09-03_ — Plan captured.
- _2026-09-03_ — Implemented Option A (two-phase disk cache) directly, per
  request, skipping step 0. Choices locked: serialize the whole `setting` dict
  with `dill` (D1); cache on fast/ephemeral storage, session-scoped, gitignored
  (D2/D3); `2a`/`2b`/`2c` + `equality_study.py` in scope, `2d` untouched. Cache
  is opt-in with an inline-build fallback, so the change is safe to land before
  the measurement. All 50 tests + 5 new pass.
