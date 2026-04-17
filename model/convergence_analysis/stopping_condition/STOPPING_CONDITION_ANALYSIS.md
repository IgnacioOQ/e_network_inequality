# Stopping Condition Analysis
- status: active
- type: research
- owner: user
- last_checked: 2026-04-17
<!-- content -->

> **Scope reminder:** All scripts and outputs referenced here belong in `model/convergence_analysis/`. Do not modify any file outside this folder. See `MC_AGENT.md` for constraints.

This document examines the tolerance-based stopping condition used in `VectorizedModel`, identifies a potential problem with it, and describes three empirical evaluation scripts.

---

## 1. What the Stopping Condition Does

**Location:** `model/vectorized_model.py`, function `stop_condition()` (nested inside `run_simulation()`).

For **Beta agents**, the condition is:

```python
np.allclose(credences_prior, self.credences, rtol=self.tolerance, atol=self.tolerance)
```

where `credences` = `α / (α + β)` for each agent and theory (shape `(N, 2)`). The simulation halts when **all** agents' credences change by less than `tolerance` in a single step.

Default parameter values (`VectorizedModel.__init__`):
- `tolerance = 5e-3`
- `tolerance_stopping = True`

Analysis scripts in this folder use `tolerance = 1e-6`, which is 5000× stricter.

For **Bayes agents** the condition is binary: `all ≤ 0.5` **or** `all > 0.99` (consensus check, not tolerance-based).

Three stopping modes exist:

| Mode | Flag | Behaviour |
|------|------|-----------|
| Tolerance-based | `tolerance_stopping=True` | Break when `allclose` satisfied |
| Fixed steps | `tstep_stopping=True` | Run exactly `number_of_steps` iterations |
| AUC-based | `auc_stopping=True` | Break when node-level AUC ≥ threshold |

---

## 2. What "Truly Stable" Would Mean

A simulation is in a **true absorbing state** when, for all future steps `k > 0`:

$$\text{credences}_{t+k} = \text{credences}_t \quad \text{(in expectation)}$$

For Beta agents this requires:
1. Every agent has committed to a single theory (epsilon-greedy exploration is negligible).
2. Evidence accumulated from experiments is no longer shifting the Alpha/Beta parameters enough to move the posterior mean.
3. All signals propagating through the network have reached their destinations.

The `allclose` check is a **local, 1-step approximation** of this: it asks whether *one particular step* was quiet, not whether the process has globally settled.

---

## 3. The Problem

### 3.1 False Convergence

A single quiet step (due to a low-variance draw) can satisfy `allclose(tol=5e-3)` even while the network is still drifting slowly. The stochastic nature of the bandit means any step may be atypically quiet. After the stop, if the simulation were resumed, credences could continue to shift — including agents flipping their theory preference (crossing the 0.5 credence boundary).

### 3.2 Credence Stability ≠ Parameter Stability

Credences are ratios: `α/(α+β)`. A credence can be flat even while `α` and `β` are both growing proportionally — the agent is accumulating strong evidence, but the *ratio* has stabilised. The `allclose` check on credences misses this. A stricter check on the raw parameters (`alphas_betas`) would catch continued evidence accumulation.

**Example:** An agent with `(α, β) = (100, 1)` has credence `0.990`. If it experiments and observes a success, `(α, β) → (101, 1)`, credence → `0.990` (unchanged to many decimal places). The stopping condition fires, but the agent just processed new evidence.

### 3.3 Tolerance Sensitivity Hypothesis

We suspect that outputs (truth-share, fraction of correct consensus, stopping time) vary **significantly** across the tolerance range `[1e-1, 1e-6]` — even though all values in this range are "low." If so, the reported simulation results are sensitive to an arbitrary implementation choice.

---

## 4. Speed of Convergence vs. Tolerance Threshold

### 4.1 NumPy Defaults vs. Simulation Default

`np.allclose` has the following built-in defaults:

```
np.allclose(a, b, rtol=1e-05, atol=1e-08)
```

The condition it evaluates is:

$$|a - b| \leq \texttt{atol} + \texttt{rtol} \cdot |b|$$

The simulation overrides **both** parameters with a single `tolerance = 5e-3`, so the effective condition is:

$$|\text{credences}_{t-1} - \text{credences}_t| \leq 5 \times 10^{-3} + 5 \times 10^{-3} \cdot |\text{credences}_t|$$

Compared to numpy's defaults, the simulation's tolerance is:
- **500× looser** than numpy's default `rtol` (`5e-3` vs `1e-5`)
- **500,000× looser** than numpy's default `atol` (`5e-3` vs `1e-8`)

In practice, since credences are bounded in `(0, 1)`, the `atol` term dominates (the `rtol` contribution is at most `5e-3 × 1 = 5e-3`), so the effective condition is approximately `|Δcredence| ≤ 1e-2`. This means the simulation can stop even when credences are still changing by up to ~1 percentage point per step.

### 4.2 Effect on Steps to Convergence

Tighter tolerances demand smaller per-step credence changes before stopping, so the simulation must run longer. The relationship is not simply proportional: because the Beta posteriors concentrate over time (evidence accumulates), per-step credence changes decay roughly as `1/t`, so cutting the tolerance by a factor of 10 roughly multiplies the stopping time by ~10 (in the slow exploitation phase, after the transition `t*`).

Concretely, we expect stopping times to follow a pattern roughly like:

| Tolerance | Relative to default | Expected stopping time (qualitative) |
|-----------|--------------------|-----------------------------------------|
| `1e-1`    | 20× looser          | Very few steps — likely stops in exploration phase |
| `1e-2`    | 2× looser           | Stops early in exploitation phase |
| `5e-3`    | **default**         | Baseline |
| `1e-3`    | 5× stricter         | ~5–10× more steps than default |
| `1e-5`    | ≈ numpy `rtol`      | Significantly longer run; well into exploitation |
| `1e-8`    | ≈ numpy `atol`      | Near-asymptotic; may approach `max_steps` cap |
| `1e-6`    | 5000× stricter      | Used in `two_phase_dynamics.py`; very long runs |

The exact multipliers depend on network topology, `n_experiments`, and the bandit gap `Δ = p_1 - p_0`. The empirical measurement is the primary goal of `stopping_tolerance_sensitivity.py`.

### 4.3 Why This Matters

If stopping times differ by an order of magnitude across the plausible tolerance range, then:

1. **Comparability breaks down.** Simulations run with different tolerance values are not sampling the same stage of the belief-formation process and cannot be directly compared.
2. **Truth-share is a time-dependent quantity.** From `root_influence_analysis.py` (§ Gap Analysis), the actual truth share at 1,000 steps is ~28% below the long-run prediction. Stopping too early — at a loose tolerance — captures a transient, not the asymptotic outcome.
3. **Numpy defaults are not safe defaults here.** The numpy `atol=1e-8` would require credence changes smaller than `1e-8`, which for the Beta agent likely never occurs in finite runs (Alpha and Beta parameters grow monotonically, so the posterior mean always shifts by at least `O(1/t)`). Using numpy defaults without a `max_steps` cap could result in simulations that run until the cap regardless of tolerance.

### 4.4 Measurement

`stopping_tolerance_sensitivity.py` records `steps_taken` for each tolerance level across `N_RUNS` independent runs. The expected output is a distribution of stopping times per tolerance, from which mean and variance can be compared. A useful derived quantity is the **stopping-time ratio**: `mean_steps(tol) / mean_steps(default)`, which quantifies the computational cost of tightening the criterion.

---

## 5. Open Questions

These are proposed additions to `OPEN_QUESTIONS.md`:

1. **Stopping-time sensitivity**: Does truth-share vary monotonically with tolerance, or is there a plateau below which outputs stabilise?
2. **Post-stopping drift**: After `tolerance_stopping` fires, how much do credences drift if the simulation is resumed? Do agents flip theories?
3. **Topology interaction**: Is sensitivity higher on sparse networks (where signals take longer to propagate) than on dense ones?
4. **Credence vs. parameter stopping**: Does checking `np.allclose` on `alphas_betas` instead of `credences` meaningfully delay stopping and change output?
5. **Minimum safe tolerance**: Is there a tolerance value below which simulation outputs converge to the same distribution as fixed-step runs?

---

## 6. Evaluation Scripts

Three scripts in this folder test the above concerns empirically. All scripts:
- Import via `sys.path.insert` from project root.
- Use `networks/network_generation.py` for topology.
- Save outputs to `results/` at the project root.
- Do **not** modify any core model file.

### Script 1: `stopping_tolerance_sensitivity.py`

Sweeps `tolerance ∈ {1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6}` plus a fixed-step baseline (`tstep_stopping=True`, `max_steps=10**6`). Runs `N_RUNS` independent simulations per tolerance. Records `steps_taken`, `truth_share`, `mean_credence_correct`, and `fraction_consensus`.

**Output:** `results/tolerance_sensitivity.csv`

### Script 2: `post_stopping_drift.py`

Runs the model with default tolerance (`5e-3`) until it stops, saves the state (`alphas_betas`, `credences`), then resumes from that exact state and runs for additional `K ∈ {10_000, 50_000, 100_000}` steps. Measures drift in credences and theory-flip counts.

**Output:** `results/post_stopping_drift.csv`

### Script 3: `tolerance_vs_alphabeta.py`

Runs the same simulation under two parallel stopping criteria: (A) current credence-based `allclose`, (B) the same check applied to `alphas_betas` directly. Compares stopping times and truth-shares.

**Output:** `results/alphabeta_stopping_comparison.csv`

---

## 7. Cross-References

- `OPEN_QUESTIONS.md` — Questions 2 (stability vs. convergence speed) and 3 (two-phase dynamics) are directly related.
- `STOCHASTIC_HYPOTHESIS.md` — The Markov Chain framing assumes absorption; this analysis tests whether the simulation actually reaches absorption before stopping.
- `convergence_studies.py` — Existing script tracking per-step belief change; its output motivates the tolerance sweep here.
- `two_phase_dynamics.py` — Uses a stricter `1e-6` tolerance with max-credence-change check; an implicit acknowledgment that the default `5e-3` may be insufficient.
