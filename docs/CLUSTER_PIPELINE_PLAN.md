# Cluster compute pipeline — plan and open questions

**Status:** draft, blocked on questions for the cluster-experienced collaborator.
**Owner:** Ignacio. **Last updated:** 2026-09-03.

## Purpose

The equality-study notebooks (`2a`–`2d`) are meant to run in three places:

1. **Local** — a laptop, from a repo clone.
2. **Google Colab** — force-fresh clone, `pip install`, Google Drive for persistent output.
3. **University cloud cluster** — the subject of this document.

Path (3) currently *works* only in the sense that the notebook can be run on a
cluster node through its **local** branch (`RUNNING_LOCALLY = True`). There is no
dedicated cluster mode, no headless entry point, and no batch/array submission.
This document records what is already in place, what is missing, and the
questions we need answered before we can commit to a design.

## Background — how the notebooks are built today

All of `2a`/`2b`/`2c` are the **same notebook**. They differ in exactly one code
cell (`config`: option slug, `UNCERTAINTY`, `CS_WINDOW`, `N_VARIANTS`). Every
piece of real logic lives in [`model/equality_study.py`](../model/equality_study.py);
the notebooks only orchestrate. `2d` aggregates the per-variant output shards and
hands the result to notebook 3.

Cell skeleton (2a is a more readable refactor of the same cells 2b/2c carry):

| Section | Cells | Role |
|---|---|---|
| Setup | `env-switch` → `clone` → `imports` | environment-dependent bootstrap |
| Config | `config` → `paths` → `fingerprint` | study params, output dir, reconcile with disk |
| Load networks | `load-networks` → `baseline-stats` | read PUD / Tobacco / Ego, assert |
| Progress report | `status` → `coverage` | read-only; safe to run before any compute |
| Run | `run` → `run_study(… variant_slice=VARIANT_SLICE …)` | the work |
| Post | `session-report` → `variance-check` → `summarise` → `plots` → `projection` | validation and summaries |
| Disconnect | `disconnect` | release the runtime |

### The environment switch

A single boolean, set at the top of the Setup section:

```python
RUNNING_LOCALLY = True   # True  → laptop: run from the repo clone, no clone/pip/Drive
                         # False → Colab: force-fresh clone, pip install, mount Drive
```

Three things branch on it:

- **`clone` cell** — `git clone` + `pip install -q dill`, or (local) walk up to the
  repo root.
- **`sys.path` / `cwd`** — set to the repo root either way.
- **`paths` cell** — `RESULTS_DIR` is `results/equality_study/<option>/<smoke|full>/`
  locally, or a Google Drive path under Colab, where it also calls `drive.mount`.

Because it is a boolean, it can only express (1) vs (2). **Cluster mode has
nowhere to live.**

## What is already in place for multi-machine runs

The hard parts of a distributed run are done. `model/equality_study.py` was
written with slicing in mind:

- **Deterministic seeds.** Every variant's seed is a pure function of
  `MASTER_SEED` keyed on `(network, arm, variant_index)`
  (`variant_sequences`, `variant_seeds`). Variant 457 is bit-identical no matter
  which machine builds it, or in what order. `MASTER_SEED` is the shared constant
  that makes "many machines, one study" coherent.
- **`VARIANT_SLICE`.** A `(start, stop)` pair, plumbed all the way through
  `run_study → run_arm`. The config-cell comment already says: *"so several
  machines can divide one arm without coordinating."* `None` → take the whole arm.
- **Atomic shard writes.** `write_shard` writes to a temp name and renames into
  place, so a killed job cannot leave a truncated shard that `resume` then trusts.
  Many machines writing one shared filesystem is safe.
- **`ACCUMULATE` + one shard per variant.** Re-running, overlapping slices, and
  requeued jobs are all idempotent — a variant already on disk is skipped.
- **`check_fingerprint`.** Identity keys (`uncertainty`, `MASTER_SEED`,
  `MAX_STEPS`, `CS_WINDOW`, `N_RUNS`, `SMOKE_TEST`) must match what is stamped on
  disk or the run raises; extent keys (`N_VARIANTS`, methods) may grow.
- **`2d` aggregation.** Reads the shard tree regardless of which machine wrote
  each shard.
- `study_report` itself prints *"split the work with VARIANT_SLICE across machines."*

## The gap — what cluster mode needs

1. **A third environment value.** `RUNNING_LOCALLY` (bool) → something that can
   say `local` / `colab` / `cluster`, read from an environment variable with a
   notebook-level override, so the same file runs unchanged on every array task.
2. **A headless entry point.** Compute nodes should not run Jupyter. A small
   `scripts/run_equality_study.py` that imports the *same* `equality_study`
   functions and does: load networks → `check_fingerprint` → `run_study` → save
   `cost`. Arguments: `--option`, `--results-dir`, `--variant-slice START STOP`,
   `--cores`, `--smoke/--full`, `--master-seed`.
3. **One source for the per-option parameters.** Move the `UNCERTAINTY` /
   `CS_WINDOW` / `N_VARIANTS` table into an `OPTIONS = {…}` dict in
   `equality_study.py`, so the notebook `config` cell and the driver read the same
   values and cannot drift.
4. **A batch/array submission script.** Maps a task index to a `VARIANT_SLICE`,
   builds the environment, runs the driver, and triggers the `2d` merge once the
   array finishes. Scheduler-specific — hence the questions below.
5. **Documentation** of the resulting workflow (this file, once filled in; and a
   general how-to in the knowledge base once the pattern is proven).

## Open questions for the collaborator

Grouped so they can be answered in one pass. Answers go inline under each
question; turn resolved ones into prose in the relevant section above.

### A. The cluster itself

- **A1.** Which scheduler / resource manager? (SLURM, PBS/Torque, PBS Pro,
  SGE/Grid Engine, LSF, HTCondor, Kubernetes, other.)
  _Answer:_
- **A2.** Is there a job-array facility, and what is its syntax and its per-user
  array-size cap? (SLURM `--array=0-N%K`, PBS `-J`, SGE `-t`, …)
  _Answer:_
- **A3.** Typical limits per job: max wall time, max cores per node, memory per
  core, and whether jobs are pre-empted / requeued.
  _Answer:_
- **A4.** Do compute nodes have outbound internet? (Decides whether `git clone` /
  `pip` / `uv sync` can run inside the job, or the env must be prebuilt on a
  login node.)
  _Answer:_

### B. How the study is run there today

- **B1.** Is it run at all on the cluster yet, or only locally and on Colab?
  _Answer:_
- **B2.** If run there: interactively (Jupyter / notebook server), or headless
  (`jupyter nbconvert --execute`, `papermill`, a hand-written script)?
  _Answer:_
- **B3.** One node doing the whole run, or several nodes each with a hand-set
  `VARIANT_SLICE`? If split, how is the split decided and recorded?
  _Answer:_
- **B4.** Which branch of the env switch is used — `RUNNING_LOCALLY = True`?
  _Answer:_

### C. Code and environment on the node

- **C1.** How does the repo get onto the cluster — `git clone`, `git pull` on a
  shared path, `rsync` from a laptop, a release tarball?
  _Answer:_
- **C2.** How is the Python environment created — `uv sync` from
  `pyproject.toml` / `uv.lock`, a `module load python/…`, Conda, a prebuilt
  virtualenv on a shared filesystem, a container (Singularity/Apptainer)?
  _Answer:_
- **C3.** Is there a shared filesystem visible from all compute nodes, or is
  per-node scratch the only option? Path(s)?
  _Answer:_
- **C4.** Any purge policy on scratch (files deleted after N days)? Where should
  results live so they survive?
  _Answer:_

### D. Output and hand-off

- **D1.** Where should `RESULTS_DIR` point on the cluster — repo `results/`, a
  project directory, scratch then copied out?
  _Answer:_
- **D2.** How do results get back to a machine that runs notebook 3 — `rsync`,
  cluster-to-Drive upload, a shared mount?
  _Answer:_
- **D3.** Who runs the `2d` aggregation and notebook 3 — on the cluster, or after
  pulling results down?
  _Answer:_

### E. Scale

- **E1.** Target grid for the real (non-smoke) run: `N_VARIANTS` per option, and
  which options (`option1_literature`, `option2_harder`, `option3_phase`) are in
  scope.
  _Answer:_
- **E2.** Rough per-variant runtime observed so far (from `%%time` on the `run`
  cell or `study_report`), to size the array chunking.
  _Answer:_

## Proposed design (contingent on the answers above)

Sketch only — revise once questions A–E are answered.

### Environment switch

```python
# One of: "local", "colab", "cluster". Read from EQSTUDY_ENV, default "local".
ENV = os.environ.get("EQSTUDY_ENV", "local")
```

| Concern | `local` | `colab` | `cluster` |
|---|---|---|---|
| Clone + pip | no | yes | no (job builds the env) |
| Drive mount | no | yes | no |
| `RESULTS_DIR` | `results/equality_study/…` | Drive path | `$EQSTUDY_RESULTS_DIR` |
| `VARIANT_SLICE` | `None` (notebook) | `None` | from `$EQSTUDY_VARIANT_SLICE` |
| `MAX_CORES`, `SMOKE_TEST` | notebook literals | notebook literals | from env vars |

The notebook keeps a manual override at the top so an interactive session can
still force any mode.

### Headless driver

`scripts/run_equality_study.py` — thin argparse wrapper over the same
`equality_study` functions the notebook calls. No Jupyter on compute nodes. The
notebook `run` cell can optionally shell out to it so there is exactly one code
path.

### Array job (scheduler TBD — placeholder is SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=eqstudy
#SBATCH --array=0-19          # 20 chunks; %K concurrency cap per A2
#SBATCH --cpus-per-task=8     # per A3
#SBATCH --mem-per-cpu=2G      # per A3; workers are memory-bound
#SBATCH --time=24:00:00       # per A3

CHUNK=25                      # variants per task; N_VARIANTS / array size
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
STOP=$((  START + CHUNK ))

export EQSTUDY_ENV=cluster
export EQSTUDY_RESULTS_DIR=/path/to/shared/results   # per C3/C4/D1

# env build per C2 — one of:
#   uv sync --frozen         (if C4 says internet is available)
#   source /shared/venvs/eqstudy/bin/activate   (prebuilt)
#   module load python/3.10 && ...

srun python scripts/run_equality_study.py \
    --option option1_literature \
    --results-dir "$EQSTUDY_RESULTS_DIR" \
    --variant-slice "$START" "$STOP" \
    --cores "$SLURM_CPUS_PER_TASK" \
    --full
```

A dependent merge job (`--dependency=afterok:<arrayjobid>` or the scheduler's
equivalent) runs `2d` / a `scripts/aggregate.py` once every chunk is banked.

## Implementation plan

Ordered; steps 1–3 do not depend on the cluster answers and can start now if we
want. Steps 4–6 need questions A–E resolved.

1. Add `OPTIONS = {…}` to `equality_study.py`; point the notebook `config` cells
   at it. Pure refactor, covered by existing tests.
2. Add `scripts/run_equality_study.py` (headless driver). Verify it reproduces a
   smoke run bit-for-bit against the notebook.
3. Introduce the `ENV` tri-state switch in `2a`–`2d`, keeping a manual override.
   `cluster` initially behaves like `local` plus the env-var reads.
4. Write `scripts/cluster/` submission scripts for the actual scheduler (A1/A2).
5. Do a smoke array run on the cluster; confirm shards from parallel tasks merge
   cleanly in `2d`.
6. Fill in this document as prose, then lift the general pattern into a
   knowledge-base how-to.

## Decision log

- _2026-09-03_ — Document lives in `docs/`, not the knowledge base, until the
  cluster workflow is actually proven. KB how-to comes later (step 6).
