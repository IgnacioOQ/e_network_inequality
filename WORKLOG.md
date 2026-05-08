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
