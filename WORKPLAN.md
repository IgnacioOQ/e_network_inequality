# WORKPLAN.md
- status: active
- last_checked: 2026-04-15
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
