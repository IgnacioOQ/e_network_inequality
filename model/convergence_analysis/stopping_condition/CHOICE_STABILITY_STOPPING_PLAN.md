# Choice-Stability Stopping Study Plan
- status: active
- type: plan
- owner: agent
- last_checked: 2026-07-17
<!-- content -->

> **Scope reminder:** All scripts and outputs referenced here belong in `model/convergence_analysis/stopping_condition/`. The one exception is the core-engine edit to `model/vectorized_model.py`, explicitly authorized by the user for this work (see §1) — it must be well documented and backwards-compatible. See `MC_AGENT.md`.

Introduce a new stopping criterion for `VectorizedModel`: halt when **every** agent's chosen theory has been unchanged for the last `W` steps. Because `epsilon = 0` is hardcoded, an agent's choice is the deterministic greedy pick `argmax(credence)` — equivalently the indicator `credences[:,1] > credences[:,0]` — so the criterion is exactly "no agent has crossed the 0.5 decision boundary in the last `W` steps." This targets the *decision*-stability failure mode (false convergence and post-stop theory flips, `STOPPING_CONDITION_ANALYSIS.md` §3.1) that the current credence-magnitude tolerance rule misses.

---

## Locked Design Decisions

Decisions agreed with the user, recorded so a resuming agent does not re-litigate them.

| Decision | Choice |
|---|---|
| **Strictness** | Strict — 100% of agents flip-free for `W` steps. A lone oscillating agent legitimately blocks stopping; the run hits `max_steps` and is recorded with `stabilized = False` (real data, not a failure). |
| **Window `W`** | Swept `{100, 250, 500, 1000}`, derived offline from a single run per (params, seed) via the record-once trick (§2). The window axis costs no extra simulation. |
| **Choice definition** | The *intended* greedy choice `credences[:,1] > credences[:,0]` (robust even if `epsilon` were raised above 0), not the realized pull. |
| **Study shape** | Full Study-2-style grid, in a **new** notebook. |
| **Core edits** | `vectorized_model.py` may be modified, provided changes are documented and backwards-compatible. `MC_AGENT.md`'s "immutable" line is superseded and reconciled in §8. |

**Honest boundary to document:** choice-stability guarantees *decision* stability, not *parameter* stability (`STOPPING_CONDITION_ANALYSIS.md` §3.2). Agents may keep accumulating evidence (α, β growing) while never flipping — acceptable, because the study's outcome of interest (truth-share / consensus) depends only on choices.

---

## Task Dependency Overview

| # | Task | Blocked by |
|---|---|---|
| 1 | Backwards-compat audit + green baseline | — |
| 2 | Core engine: `choice_stability_stopping` mode + `record_choice_flips` | 1 |
| 3 | Unit tests | 2 |
| 4 | Reference driver script | 2 |
| 5 | New study notebook (mirror Study 2) | 4 |
| 6 | Data analysis — H1′ / H2′ | 5 |
| 7 | Comparison vs tolerance stopping + post-stop drift | 5 |
| 8 | Documentation updates | 6, 7 |

---

## 1. Backwards-Compatibility Audit and Green Baseline

Establish a green baseline and prove the core change will be purely additive before touching the engine.

1. Run the existing suite: `.venv/bin/python -m unittest discover -s testing/unit_tests -v`. It must be green; the 3 tests in `testing/unit_tests/test_stopping_conditions.py` cover the `tolerance_stopping` / `tstep_stopping` precedence chain.
2. Confirm defaults: `tolerance_stopping=True`, `tstep_stopping=False` (`vectorized_model.py` `__init__`). The stop-dispatch in `run_simulation` is a mutually-exclusive `if/elif` chain: `tolerance_stopping` → `tstep_stopping` → `auc_stopping` (the last is a `run_simulation` parameter, not an `__init__` attribute).
3. Enumerate call sites (`grep -rn "VectorizedModel(" --include='*.py'`): `utils/mc_analysis.py`, the unit tests, `model/vectorized_simulation_functions.py`, and every script under `model/convergence_analysis/`. All omit the new params, so defaulting them OFF preserves current behavior everywhere.

**Verification:** suite green; a written list of call sites confirming none pass the new params.

---

## 2. Core Engine — Choice-Stability Stopping Mode

Edit `model/vectorized_model.py`, documented inline, keeping every change additive.

- **New `__init__` params (default OFF):** `choice_stability_stopping: bool = False`, `choice_stability_window: int = 500`, `record_choice_flips: bool = False`. Docstring block explaining the criterion, the `epsilon=0` assumption, and the decision-vs-parameter-stability caveat.
- **Stop dispatch:** add `elif self.choice_stability_stopping:` after the `tstep_stopping` branch. Reachable only when the new flag is True and both existing flags are False — so no existing combination changes.
- **State tracked (beta agents):** initialize `self._prev_choices` (bool vector) and `self._last_flip_step`. Each step recompute `choices = self.credences[:,1] > self.credences[:,0]`; if any differ from `_prev_choices`, set `_last_flip_step = self.n_steps` and update `_prev_choices`. Stop when `self.n_steps - self._last_flip_step >= self.choice_stability_window`. O(N) memory, O(N)/step.
- **Record-once instrumentation:** when `record_choice_flips`, append `(step, truth_share)` to `self.choice_flip_history` on every step where ≥1 agent flips. Truth-share is constant between flips, so this list reconstructs the stop-step and truth-share for *any* `W` offline: stop `= (start of first inter-flip gap ≥ W) + W`. This flag is orthogonal to the stopping mode — the study runs in `tstep` mode to `max_steps` with `record_choice_flips=True` and derives all four windows from one run.
- **Wrapper:** thread the three new params additively through `model/vectorized_simulation_functions.py` (defaults preserve behavior).

**Verification:** all existing call sites still construct and run identically; a manual `choice_stability_stopping=True` run halts on a synthetic case at the expected step.

---

## 3. Unit Tests

Extend `testing/unit_tests/test_stopping_conditions.py` (keep the 3 existing tests unchanged):

- Synthetic/short run with a known last-flip step → assert stop at `last_flip + W` for each `W ∈ {100,250,500,1000}`.
- Never-stabilizing case (agent forced to oscillate) → assert `n_steps` reaches the cap and `stabilized` is False.
- Equivalence: native-mode stop-step equals the offline-derived stop-step (from `choice_flip_history`) for the same seed.
- Determinism: a seeded run reproduces the stop-step.

**Verification:** `.venv/bin/python -m unittest discover -s testing/unit_tests -v` green, including the new cases.

---

## 4. Reference Driver Script

Create `choice_stability_stopping.py` in this folder — a single-network local driver exercising the native mode, matching the existing reference-script pattern (`post_stopping_drift.py`, `stopping_tolerance_sensitivity.py`). Serves local testing/debugging before the Colab grid.

**Verification:** runs locally on `pud_network.pkl`, prints stop-step and truth-share, writes its CSV inside the folder.

---

## 5. New Study Notebook (mirror Study 2)

Author `A. Choice Stability Stopping Study.ipynb` in this folder, mirroring the structure of `A. Stopping Condition Study Final.ipynb` (Study 2). Match the existing notebooks' Colab flavor (`nbformat_minor: 0`, ids at `metadata.id`, every cell carries a `metadata` dict).

Apply the notebook skill (`content/how-to/NOTEBOOK_WRITING_SKILL.md` in the knowledge base): `RUNNING_LOCALLY` switch, `SMOKE_TEST` + tiered compute budgets, force-fresh git-clone bootstrap, Drive I/O via one path constant per role, `multiprocessing.Pool` with per-worker `SeedSequence.spawn` seeding (seed written into each result row), `%%time` + `tqdm` on heavy cells, disconnect cell last. Heavy logic imported from modules; the notebook orchestrates.

**Grid:** uncertainty `{1e-4, 1e-3, 5e-3, 1e-2}` × n_experiments `{100, 500, 1000}` × 500 runs × networks `{pud, tobacco}`, run once per (params, seed) with `record_choice_flips=True`; derive all `W ∈ {100,250,500,1000}` offline. Record `steps_taken(W)`, `truth_share(W)`, `stabilized(W)`. Outputs (folder-local, `{network}_` prefixed): raw CSV, summary CSV, stop-time distribution and window-sensitivity plots.

**Verification:** Restart-and-Run-All passes under `SMOKE_TEST=True`; `python3 -c "import json; json.load(open(...))"` succeeds; outputs land inside the folder.

---

## 6. Data Analysis — H1′ / H2′

Add a data-analysis section mirroring Study 2's, using `utils/data_analysis_utils.py` (OLS, Pearson/VIF, standardized coefficients, Cohen's f², diagnostics):

- **H1′ (Steps):** steps-to-stabilization driven by `W` (additive offset), `uncertainty`, and `n_experiments` (all shorten time-to-last-flip).
- **H2′ (Truth-share):** truth-share at stabilization driven mainly by `uncertainty` and **near-invariant to `W`** — the headline testable prediction and the criterion's main selling point.

**Verification:** regression tables + f² bar chart produced; H2′ W-invariance visible in the output.

---

## 7. Comparison vs Tolerance Stopping + Post-Stop Drift

Reusing the `post_stopping_drift.py` structure, compare choice-stability vs tolerance stopping on shared seeds: stop-step, truth-share, and **post-stop flip rate** after resuming `K` steps. Hypothesis: choice-stability yields near-zero post-stop flips versus a nonzero rate for tolerance stopping.

**Verification:** a comparison table/plot showing the flip-rate difference across continuation lengths.

---

## 8. Documentation Updates

- Add a new study section (and a §4-style reasoning block) to `STOPPING_CONDITION_ANALYSIS.md`; mark open questions #2 (post-stopping drift) and #4 (credence vs parameter stopping) as addressed, and add a new one on the W-invariance of truth-share.
- Reconcile `MC_AGENT.md`: `vectorized_model.py` is no longer immutable for this work — record the user authorization and the backwards-compatibility guarantee.
- Append entries to `WORKLOG.md` and `AI_AGENTS/AGENTS_LOG.md` per repo convention. All markdown follows `AI_AGENTS/MD_CONVENTIONS.md`.

**Verification:** docs updated and internally consistent; close the tracking task (`cp_mcp_task_close`, `todo.choice_stability_stopping`).

---

## Cross-References

- `STOPPING_CONDITION_ANALYSIS.md` — the analysis this study extends; §3.1 (false convergence), §3.2 (credence vs parameter stability), §7 (Study 2 structure this notebook mirrors).
- `MC_AGENT.md` — folder scope and constraints; the immutability line reconciled in §8.
- `post_stopping_drift.py` — structural template for §7.
- Central tracking task: `todo.choice_stability_stopping` (cp_mcp).
