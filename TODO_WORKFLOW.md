# TODO_WORKFLOW.md
- status: active
- type: workflow
- last_checked: 2026-05-08
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

### Decide on Stopping Condition
- status: todo
- owner: Max, Ignacio
- priority: high
<!-- content -->
Agree on fixed-step count (likely 1,000,000). Must replicate Zollman (2007) as a correctness
check. Optionally: add early-stop check every 10k steps after a minimum run.

### Hyperparameter Optimization for Variation Methods
- status: todo
- owner: Hein
- priority: medium
- blocked_by: [Fix Ego Depletion]
<!-- content -->
Use [hyperopt](https://hyperopt.github.io/hyperopt/) to tune `p_conditional`, `p_max_edges`,
`p_max_rewired`. Objective: minimize correlation between network statistics across varied networks.

### Run Large-Scale Simulations
- status: blocked
- owner: Max, Ignacio
- priority: high
- blocked_by: [Goal 1, Decide on Stopping Condition]
<!-- content -->
Run on all three networks × variants (original, densified, equalized) via
[`2. GColab Simulations.ipynb`](2.%20GColab%20Simulations.ipynb). Use Utrecht University cloud.

---

## Goal 3: Revise Paper
- status: in-progress
- priority: high
<!-- content -->
Address FEW reviewer feedback. Writing can proceed in parallel with simulation work.

### Explain Why Results Differ from Theoretical Networks
- status: todo
- owner: Ignacio, Max
- blocked_by: [Run Large-Scale Simulations]
<!-- content -->
Identify which structural property (degree heterogeneity, clustering, diameter, hubs) drives the
disappearance of the Zollman and equality effects on empirical networks.

### Clarify Dynamics-of-Inquiry Takeaways
- status: todo
- owner: Ignacio, Max
<!-- content -->
Rewrite the discussion to clearly state what the results teach about how scientific inquiry proceeds.

### Strengthen "How Actually" Framing
- status: todo
- owner: Ignacio, Max
<!-- content -->
Clarify what phenomenon the model explains and how empirical topologies advance the mechanistic
explanation beyond theoretical models.

---

## Goal 4: Repository Hygiene
- status: active
- priority: low
<!-- content -->
Non-scientific maintenance work that improves tooling and developer experience. Can be done
opportunistically when scientific tasks are blocked.

### Migrate Legacy Notebooks to nbformat 4.5
- status: todo
- owner: Ignacio
- priority: low
<!-- content -->
13 of 20 notebooks are still on nbformat 4.0 / 4.2 and have no cell IDs, which prevents
AI-assisted editing tools (and stable diffs / addressable cells in general) from targeting
specific cells. See the KB skill `content/how-to/NOTEBOOK_WRITING_SKILL.md` §1 for context
(nbformat 4.5 + cell IDs).

Affected notebooks:
- Root: `1. Citation Data and Networks Generation.ipynb`, `2. GColab Simulations.ipynb`,
  `A. GColab Simulations Playground.ipynb`
- `model/convergence_analysis/`: `phase_dynamics/Colab_Ignacio_Convergence_Study.ipynb`,
  `stopping_condition/A. 100k Stopping Study.ipynb`,
  `stopping_condition/A. Stopping Condition Study Final.ipynb`,
  `stopping_condition/A. Stopping Condition Study v2.ipynb`
- `testing/notebooks/`: `basic_model_testing.ipynb`, `basic_model_testing_v2.ipynb`,
  `vectorized_basic_model_testing.ipynb`

Migration is a one-shot script: walk every `.ipynb`, bump `nbformat_minor` to `5`, and
assign an 8-char hex `id` to every cell missing one (idempotent — already-IDed cells are
left alone). Will produce a noisy diff (every cell gains an `id` field), so do it as a
single dedicated commit on a quiet day.
