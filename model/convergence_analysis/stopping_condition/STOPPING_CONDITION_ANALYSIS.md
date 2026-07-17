# Stopping Condition Analysis
- status: active
- type: research
- owner: user
- last_checked: 2026-04-23
<!-- content -->

> **Scope reminder:** All scripts and outputs referenced here belong in `model/convergence_analysis/`. Do not modify any file outside this folder. See `MC_AGENT.md` for constraints.

This document examines the tolerance-based stopping condition used in `VectorizedModel`, identifies a potential problem with it, and describes the empirical evaluation studies. Studies are run on Google Colab via `A. Stopping Condition Study.ipynb`, covering both `pud_network.pkl` and `tobacco_network.pkl`.

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

**Study 1** in `A. Stopping Condition Study v2.ipynb` (§ "Convergence Speed Analysis") is the dedicated study for this question. It runs the tolerance sweep on both `pud_network.pkl` and `tobacco_network.pkl`, using `N_RUNS=100` and `MAX_STEPS=10^5`, records only `steps_taken`, and produces:

> **Note:** Study 1 was removed from `A. Stopping Condition Study Final.ipynb` because Study 2's full grid already sweeps tolerance as one of its axes. Refer to `A. Stopping Condition Study v2.ipynb` for the standalone tolerance sweep.

- **`{network}_convergence_speed.csv`** — raw stopping times (one row per run × tolerance)
- **`{network}_convergence_speed_boxplot.png`** — box plots of stopping-time distributions per tolerance on a log scale
- **`{network}_convergence_speed_ratio.png`** — bar chart of mean steps + line plot of stopping-time ratio vs. default

The **stopping-time ratio** (`mean_steps(tol) / mean_steps(5e-3)`) is the key output: it directly quantifies the computational cost of tightening the criterion and reveals whether there is a plateau below which additional strictness yields no further delay.

The reference implementation is `convergence_speed_analysis.py` (serial, single-network version).

---

## 5. Open Questions

These are proposed additions to `OPEN_QUESTIONS.md`:

1. **Stopping-time sensitivity**: Does truth-share vary monotonically with tolerance, or is there a plateau below which outputs stabilise?
2. **Post-stopping drift**: After `tolerance_stopping` fires, how much do credences drift if the simulation is resumed? Do agents flip theories? *(Addressed in §9 — the choice-stability criterion is a direct response, and its comparison study measures post-stop flip rate against tolerance stopping.)*
3. **Topology interaction**: Is sensitivity higher on sparse networks (where signals take longer to propagate) than on dense ones?
4. **Credence vs. parameter stopping**: Does checking `np.allclose` on `alphas_betas` instead of `credences` meaningfully delay stopping and change output? *(Related to §9: choice-stability sidesteps the credence-magnitude question entirely by checking the decision `argmax`, not the credence value.)*
5. **Minimum safe tolerance**: Is there a tolerance value below which simulation outputs converge to the same distribution as fixed-step runs?
6. **Window-invariance of truth-share**: Under choice-stability stopping (§9), is the truth-share at stabilisation invariant to the window `W` (H2′), even as the stopping time grows with `W` (H1′)?

---

## 6. The Role of Uncertainty and Experiment Count

### 6.1 What `uncertainty` Controls

In `VectorizedBandit`, `uncertainty` is the **bandit gap**:

```python
p_bad_theory  = 0.5
p_good_theory = 0.5 + uncertainty
```

A higher `uncertainty` makes the true theory easier to distinguish from the false one, so agents accumulate decisive evidence faster and credences move more sharply per step. This shortens the time to convergence and increases truth-share.

**Note on epsilon:** The epsilon-greedy exploration rate is hardcoded to `self.epsilon = 0` in `VectorizedModel` — agents are always greedy (exploit best credence). The "exploration phase" seen in `two_phase_dynamics.py` arises not from epsilon-greedy randomness but from uncertain initial priors: agents initialized near `credence ≈ 0.5` behave as if exploring until evidence pushes them to commit.

### 6.2 What `n_experiments` Controls

`n_experiments` is the number of bandit pulls per agent per step. More pulls per step means a larger evidence update each round, so credences move faster and the simulation converges in fewer steps — at the cost of higher computational load per step.

### 6.3 Expected Interactions

| Effect | Direction |
|--------|-----------|
| Higher `uncertainty` → larger bandit gap | Faster convergence, higher truth-share |
| Higher `n_experiments` → more evidence per step | Fewer steps to convergence |
| Stricter `tolerance` | More steps required |
| Interaction `uncertainty × tolerance` | Large gap may allow loose tolerance to still be "safe"; small gap may require strict tolerance to capture the correct outcome |

The variance in truth-share (not just the mean) is the key diagnostic: high variance at a given (tolerance, uncertainty) combination signals that the stopping point is unreliable — some runs stop while still in a transient state.

---

## 7. Evaluation Studies

### Primary Entry Point: Colab Notebook

All active studies are run on Google Colab via **`A. Stopping Condition Study Final.ipynb`** in this folder. Compute-intensive runs require Colab; the `.py` scripts serve as reference implementations for local testing.

**Shared notebook parameters:**
- Networks: `pud_network.pkl` and `tobacco_network.pkl`
- `N_RUNS = 100` per condition
- `MAX_STEPS = 100_000` (10⁵)
- All outputs saved to Google Drive, prefixed by network name (`pud_*`, `tobacco_*`)

---

### Study 1: Convergence Speed Analysis (notebook § "Study 1")

Dedicated to §4. Sweeps `tolerance ∈ {1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6}` at fixed `uncertainty=0.00001` and `n_experiments=10`. Records only `steps_taken` and produces stopping-time distributions and the stopping-time ratio table.

**Reference script:** `convergence_speed_analysis.py`

**Outputs (per network):**
- `{network}_convergence_speed.csv` — raw stopping times
- `{network}_convergence_speed_boxplot.png` — stopping-time distributions (log scale)
- `{network}_convergence_speed_ratio.png` — mean steps + ratio vs. default

---

### Study 2: Parameter Search (notebook § "Study 2")

Dedicated to §6. Full grid search over `tolerance × uncertainty × n_experiments`. Records `steps_taken` and `truth_share` for each combination.

**Grid:** tolerances `{1e-3, 1e-4, 1e-5, 1e-6, 5e-7}` × uncertainties `{0.0001, 0.001, 0.005, 0.01}` × n_experiments `{100, 500, 1000}` × 500 runs = **30,000 simulations per network**.

**Reference script:** `parameter_search.py`

**Outputs (per network):**
- `{network}_parameter_search.csv` — raw results
- `{network}_parameter_search_summary.csv` — group means and variances
- `{network}_parameter_search_heatmap_nexp{5,10,20}.png` — heatmaps (one per n_experiments value)
- `{network}_parameter_search_lines.png` — mean steps vs. tolerance, faceted by n_experiments

---

### Study 2 Data Analysis (notebook § "Study 2: Parameter Search — Data Analysis")

Dedicated to testing the two hypotheses about which parameters drive outcomes. Loads the CSV output of Study 2 and applies OLS regression via `utils/data_analysis_utils.py`.

**Hypotheses:**
- **H1 (Steps):** Steps to convergence is driven mainly by `tolerance`; `uncertainty` and `n_experiments` have little effect.
- **H2 (Truth share):** Mean truth share is driven mainly by `uncertainty`; `tolerance` and `n_experiments` have little effect.

**Predictors** (all log₁₀-transformed): `log_tolerance`, `log_uncertainty`, `log_n_experiments`.

**Per-network outputs:**
- Pearson correlation matrix + VIF table (multicollinearity check; expect VIF ≈ 1.0 for orthogonal grid)
- OLS for `steps`: R², standardised coefficients, Cohen's f² per predictor, regression diagnostics
- OLS for `truth_share`: same
- Grouped bar chart comparing Cohen's f² across both outcomes

---

### Reference Scripts (local, single-network)

These `.py` scripts were the original implementations and remain useful for local testing and debugging.

| Script | Purpose |
|--------|----------|
| `convergence_speed_analysis.py` | Tolerance sweep → stopping-time distributions (§4) |
| `parameter_search.py` | Full grid: tolerance × uncertainty × n_experiments (§6) |
| `stopping_tolerance_sensitivity.py` | Tolerance sweep + fixed-step baseline |
| `post_stopping_drift.py` | Resume after stop; measure drift and theory flips |
| `tolerance_vs_alphabeta.py` | Credence-based vs. parameter-based stopping comparison |

---

## 8. Cross-References

- `OPEN_QUESTIONS.md` — Questions 2 (stability vs. convergence speed) and 3 (two-phase dynamics) are directly related.
- `STOCHASTIC_HYPOTHESIS.md` — The Markov Chain framing assumes absorption; this analysis tests whether the simulation actually reaches absorption before stopping.
- `convergence_studies.py` — Existing script tracking per-step belief change; its output motivates the tolerance sweep here.
- `two_phase_dynamics.py` — Uses a stricter `1e-6` tolerance with max-credence-change check; an implicit acknowledgment that the default `5e-3` may be insufficient.

---

## 9. Choice-Stability Stopping (Decision-Stability Criterion)

A different answer to the false-convergence problem (§3.1): stop when the network's **decisions** have settled, not when a single step was quiet.

### 9.1 The Criterion

Stop when **every** agent's chosen theory has been unchanged for the last `W` steps. An agent's choice is the greedy pick `argmax(credence)`; since `epsilon = 0` is hardcoded (§6.1), this is the deterministic indicator `credences[:, 1] > credences[:, 0]`. So the rule is equivalent to:

> No agent has crossed the 0.5 decision boundary in the last `W` steps.

This targets exactly what the tolerance rule misses. Where `allclose` asks "was *this* step quiet?" (and a low-variance draw can spuriously satisfy it while the network still drifts), choice-stability asks "has the *consensus* stopped moving?" — the quantity the study's truth-share output actually depends on.

**Implementation** (`model/vectorized_model.py`, documented, backwards-compatible — all flags default OFF):

- `choice_stability_stopping=True`, `choice_stability_window=W` — a fourth stopping mode, a mutually-exclusive `elif` after `tolerance_stopping` / `tstep_stopping`. It tracks `_last_flip_step` (the most recent step any agent flipped) and stops when `n_steps - _last_flip_step >= W`.
- `record_choice_flips=True` — orthogonal instrumentation appending `(step, truth_share)` to `choice_flip_history` on every flip. Because truth-share is constant between flips, one run recorded at the largest window yields the stop step and truth-share for **every** `W` offline (the *record-once* trick). Native and offline agree exactly (`testing/unit_tests/test_stopping_conditions.py`).

### 9.2 Boundary — Decision vs. Parameter Stability

Choice-stability guarantees *decision* stability, not *parameter* stability (§3.2). An agent at `(α, β) = (100, 1)` never flips even as α keeps growing. This is acceptable — indeed desirable — because the epistemic outcome of interest (which theory the network settles on) is a function of decisions, not of the raw parameter magnitudes.

### 9.3 Hypotheses and Evaluation

Run via **`A. Choice Stability Stopping Study.ipynb`** (mirrors the Study-2 grid on `pud_network.pkl` and `tobacco_network.pkl`), with `choice_stability_stopping.py` as the local reference driver. The grid sweeps `uncertainty × n_experiments`, derives `W ∈ {100,250,500,1000}` offline, and tests:

- **H1′ (Steps):** steps-to-stabilisation grow with the window `W` (and shrink with `uncertainty`, `n_experiments`).
- **H2′ (Truth-share):** truth-share at stabilisation is driven by `uncertainty` and is **~invariant to `W`** — the headline prediction. Smoke-scale OLS already shows `log_window` insignificant (p ≈ 0.65) for truth-share while `log_uncertainty` and `log_n_experiments` dominate.

A comparison sub-study (Study 2 in the notebook, reusing the `post_stopping_drift.py` resume pattern) pits choice-stability against tolerance stopping on shared seeds, measuring the **post-stop flip rate** after resuming `K` steps — directly answering open question #2. Early runs show tolerance stopping firing extremely early with post-stop flips in ~100% of runs, versus a near-zero flip rate under choice-stability.

**Outputs (per network, folder-local):** `{network}_choice_stability.csv`, `_summary.csv`, `_boxplot.png`, `_steps_lines.png`, `_truth_share_lines.png`, and `{network}_stopping_comparison.csv` / `.png`.

See `CHOICE_STABILITY_STOPPING_PLAN.md` for the full task decomposition.
