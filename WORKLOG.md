# e_network_inequality — Working Log
- status: active
- type: log
- description: Append-only chronological log of significant agent interventions, difficult problems solved, and major changes to this repository.
- injection: informational
- volatility: evolving
- last_checked: 2026-05-08
<!-- content -->
Most recent event comes first.

---

### 2026-07-17: Choice-Stability Stopping Criterion + Study
- id: worklog.2026_07_17_choice_stability_stopping
- status: done
- type: log
- last_checked: 2026-07-17
<!-- content -->
**AI Assistant**: Claude Opus 4.8 (Claude Code, VSCode extension)
**Task**: Add a decision-stability ("choice-stability") stopping criterion to `VectorizedModel` and a Study-2-style evaluation. Stop when *every* agent's chosen theory (`argmax` credence, i.e. `credences[:,1] > credences[:,0]` since `epsilon=0`) has been unchanged for the last `W` steps — a direct response to the false-convergence / theory-flip failure mode of tolerance stopping (`STOPPING_CONDITION_ANALYSIS.md` §3.1).

#### Changes made
- **`model/vectorized_model.py`** (user-authorized core edit; documented, backwards-compatible): new `choice_stability_stopping` / `choice_stability_window` / `record_choice_flips` params, all default OFF. New mutually-exclusive `elif` branch in `run_simulation` tracks `_last_flip_step`; `record_choice_flips` logs `(step, truth_share)` to `choice_flip_history` so all windows derive from one *record-once* run offline.
- **`model/vectorized_simulation_functions.py`**: threaded the three params through the wrapper; `choice_flip_history` added to `result_dict`.
- **`testing/unit_tests/test_stopping_conditions.py`**: +5 tests (gap==window per W; native==offline equivalence; determinism; oscillation→cap; defaults-off). Full suite **32 passed**.
- **New** `model/convergence_analysis/stopping_condition/`: `choice_stability_stopping.py` (reference driver), `A. Choice Stability Stopping Study.ipynb` (grid + OLS H1′/H2′ + tolerance comparison, mirrors Study 2), `CHOICE_STABILITY_STOPPING_PLAN.md` (plan).
- **Docs**: `STOPPING_CONDITION_ANALYSIS.md` §9 + open-questions; `MC_AGENT.md` immutability exception reconciled.

#### Verification
- Native and offline (record-once) stops match exactly on the real PUD network across `W ∈ {100,250,500,1000}` and in unit tests.
- Full notebook ran end-to-end headlessly (smoke mode). Early results already support **H2′** (truth-share ~invariant to W: `log_window` p≈0.65) and the post-stop-drift hypothesis (tolerance stopping → flips in ~100% of resumed runs; choice-stability → near-zero).

---

### 2026-05-30: Housekeeping Run
- id: worklog.2026_05_30_housekeeping
- status: done
- type: log
- last_checked: 2026-05-30
<!-- content -->
**Tests:** 27 passed, 0 failed (`unittest discover -s testing/unit_tests`)
**Imports:** OK — core modules + `utils.imports` re-export hub load cleanly
**Networks:** OK — PUD 90n/160e, Tobacco 289n/1229e, Ego 503n/2933e
**Notebooks:** 21 parsed OK, 0 failed (run after the nbformat 4.5 cell-ID migration)
**Notes:**
- Trigger: sanity check following the nbformat 4.5 cell-ID migration earlier today.
- PUD network now reports **90 nodes** (160 edges) vs **87 nodes** in the 2026-05-08 report — appears regenerated since; worth confirming this is expected.
- Optional Phase 5 (ruff + vulture) run but **no code changed**: F401 flags 93 unused imports (most in `utils/imports.py`, the intentional re-export hub); vulture flags `args`/`kwargs` signature padding, immutable `model/model.py` internals, re-export imports in `utils/sa_network_variation_directed.py`, and **unreachable code after `return` at `networks/variation_methods.py:480`** (the one genuine smell — left for review).

---

### 2026-05-30: Migrate Legacy Notebooks to nbformat 4.5 (cell IDs)
- id: worklog.2026_05_30_nbformat_45_migration
- status: done
- type: log
- last_checked: 2026-05-30
<!-- content -->
**AI Assistant**: Claude Opus 4.8 (Claude Code, VSCode extension)
**Task**: Migrate all legacy notebooks to nbformat 4.5 and assign stable cell IDs so AI-assisted editing tools (and stable diffs / addressable cells) can target specific cells. Closes Goal 4 "Migrate Legacy Notebooks to nbformat 4.5".

#### Changes made
- One-shot idempotent migration over all 21 notebooks: bumped `nbformat_minor` → 5 where < 5, and assigned an 8-char hex `id` to every cell missing one (canonical position: after `cell_type`, after `execution_count` for code cells).
- **312 cell IDs added across 14 notebooks**; 11 of those also bumped from nbformat 4.0/4.2 → 4.5. Seven already-compliant notebooks were left untouched.
- Serialization auto-detected per file (`indent` 1 vs 2, `ensure_ascii` true/false) so the diff is strictly id additions + version bumps — no code, outputs, or metadata altered.

#### Verification
- No-op JSON round-trip confirmed byte-identical to originals before migration, isolating the intended changes.
- `git diff` filter returns zero changed lines that are not an `id` add or `nbformat_minor` bump (323 insertions, 11 deletions).
- All 21 notebooks post-migration: `nbformat_minor == 5`, zero cells missing `id`, no duplicate IDs, `nbformat.validate` passes.
- Pre-existing Colab artifact (`errorDetails` in an error output of `phase_dynamics/Colab_Ignacio_Convergence_Study.ipynb`) confirmed present in `HEAD`; not introduced here.

---

### 2026-05-08: Snapshot Feature + Convergence Analysis Notebook Update
- id: worklog.2026_05_08_snapshot_feature
- status: done
- type: log
- last_checked: 2026-05-08
<!-- content -->
**AI Assistant**: Claude Sonnet 4.6 (Antigravity, VSCode extension)
**Task**: Add a snapshot feature to the vectorized simulation model to track truth share and max absolute belief change at fixed intervals; update analysis notebooks to visualize the trajectories.

#### Changes made

**`model/vectorized_model.py`**
- Added `snapshot_interval: int = 0` parameter to `VectorizedModel.__init__`.
- Initialized `self.snapshots = {"step": [], "truth_share": [], "max_belief_change": []}`.
- Inside `run_simulation`, at every `snapshot_interval`-th step: computes current truth share via `determine_conclusion()` and max absolute belief change (`np.max(np.abs(credences_prior - self.credences))`) for beta agents. Appends all three to `self.snapshots`.

**`model/vectorized_simulation_functions.py`**
- Added `snapshot_interval=0` to `run_vectorized_simulation_with_params` signature.
- Passed it through to `VectorizedModel(...)`.
- When `snapshot_interval > 0`, attaches `my_model.snapshots` to `result_dict["snapshots"]`.

**`model/convergence_analysis/stopping_condition/A. 100k Stopping Study.ipynb`**
- Added `SNAPSHOT_INTERVAL = 5_000` constant.
- Patched `run_compute_time_study` to pass `snapshot_interval=SNAPSHOT_INTERVAL` and return `all_snapshots`.
- Updated the runner cell to unpack the new return value.
- Added new section **"Snapshot Analysis: Belief Dynamics over Time"** with:
  - `aggregate_snapshots()` helper (stacks runs into matrices, computes mean ± std).
  - Per-network trajectory plots (truth share + max belief change on log scale, ± 1 SD shading).
  - Combined all-networks comparison plot.

**`testing/notebooks/basic_model_testing_v2.ipynb`** (new)
- Duplicated from `basic_model_testing.ipynb`.
- Switched imports to `VectorizedModel`.
- Added `snapshot_interval=1000` to simulation calls.
- Added snapshot plot cell (two-panel: truth share and max belief change).

**`README.md`**
- Added "Codebase Architecture and Execution Flow" section documenting the vectorized paradigm, parallel execution via `multiprocessing`, and the directory map for future coding agents.

**`ADD_SNAPSHOT_PLAN.md`** (new)
- Design document saved at project root: records design decisions (fixed interval, max absolute change, beta agents only), code changes, usage example, and verification checklist.

#### Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Snapshot frequency | Fixed interval | Easier to plot as continuous timeline |
| Belief change metric | Max absolute change | Strictly defines the stopping tolerance |
| Agent scope | Beta only | Bayes credences don't have the same delta interpretation |

#### Smoke test result
5 snapshots captured at steps [1000, 2000, 3000, 4000, 5000] on PUD network (87 nodes, 160 edges); truth shares in [0, 1]; max belief change decreased monotonically from 0.000522 → 0.000142.

---
