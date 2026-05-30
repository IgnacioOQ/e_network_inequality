# TODO_WORKFLOW.md
- status: active
- type: workflow
- last_checked: 2026-05-30
<!-- content -->
Tasks assigned to Ignacio, Hein, and Max, ordered by priority.
Technical context for each item is in [OBSERVATIONS.md](OBSERVATIONS.md).

---

## Goal 1: Get Networks Ready for Simulation
- status: in-progress
- priority: critical
<!-- content -->

### Fix Ego Depletion — Equalize Variation
- status: in-progress
- owner: Hein
- priority: critical
<!-- content -->
The equalize variant for ego depletion is still not working. Try:
- Milder equalization: target a mix of uniform + original distribution instead of purely uniform.
- Tune `p_conditional` upward to reduce rejection-filter bias.

### Visual Inspection of All Three Networks
- status: todo
- owner: Ignacio
- priority: high
<!-- content -->
Check PUD, Tobacco, and Ego depletion for suspicious structural features before running
simulations. Use [`A. Visualizations.ipynb`](A.%20Visualizations.ipynb).

---

## Goal 2: Run Simulations
- status: blocked
- priority: high
- blocked_by: [Goal 1]
<!-- content -->

### Investigate Theory-Flip Window Stopping Condition
- status: todo
- owner: Ignacio
- priority: medium
<!-- content -->
Investigate a stopping condition of the form: "stop if no agent has changed theory (crossed the
0.5 credence boundary) in the last X rounds." This is a macroscopic, window-based criterion that
directly targets the False Convergence concern in
[STOPPING_CONDITION_ANALYSIS.md](model/convergence_analysis/stopping_condition/STOPPING_CONDITION_ANALYSIS.md)
§3.1 — single-step `allclose` on credences can fire on a quiet step while the network is still
drifting. A theory-flip window measures whether the *choice landscape* has stabilised rather than
whether one step happened to be small.

Scope:
- Implement as a new stopping mode in `VectorizedModel` (alongside `tolerance_stopping`,
  `tstep_stopping`, `auc_stopping`); do **not** modify the immutable OO model.
- Sweep window size `X` (e.g. `{50, 100, 500, 1000, 5000}`) on `pud_network.pkl` and
  `tobacco_network.pkl` using the existing Colab harness in
  `model/convergence_analysis/stopping_condition/`.
- Compare against tolerance-based stopping on: steps-to-stop, truth-share at stop, and
  post-stopping drift (resume simulation, check whether any agent eventually flips).
- Relates to Open Questions 2 (post-stopping drift) and 5 (minimum safe tolerance) in the
  analysis doc.

### Hyperparameter Optimization for Variation Methods
- status: todo
- owner: Hein
- priority: medium
- blocked_by: [Fix Ego Depletion]
<!-- content -->
Use [hyperopt](https://hyperopt.github.io/hyperopt/) to tune `p_conditional`, `p_max_edges`,
`p_max_rewired`. Objective: minimize correlation between network statistics across varied networks.

---

## Goal 3: Revise Paper
- status: in-progress
- priority: high
<!-- content -->
Address FEW reviewer feedback. Writing can proceed in parallel with simulation work.

### Explain Why Results Differ from Theoretical Networks
- status: todo
- owner: Ignacio, Max
<!-- content -->
Identify which structural property (degree heterogeneity, clustering, diameter, hubs) drives the
disappearance of the Zollman and equality effects on empirical networks.

---

## Goal 4: Repository Hygiene
- status: active
- priority: low
<!-- content -->
Non-scientific maintenance work that improves tooling and developer experience. Can be done
opportunistically when scientific tasks are blocked.
