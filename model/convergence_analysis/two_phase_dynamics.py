#!/usr/bin/env python
# coding: utf-8

# Two-Phase Dynamics: Exploration -> Exploitation Transition
# ==========================================================
# Tracks per-agent belief-change statistics (mean + std dev) over time
# to characterize the structural transition predicted in Q3 of OPEN_QUESTIONS.md.
#
# Key signals:
#   mean |delta_belief|  -- average rate of learning per agent
#   std  |delta_belief|  -- heterogeneity across agents; peaks at transition t*
#   frac_active          -- share of agents still making non-trivial updates
#
# t* is identified as the step of peak std dev (Theory 1).
#
# The script runs model.step() directly so it can capture per-agent statistics
# that run_simulation() does not expose (it stores only the mean).

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from model.vectorized_model import VectorizedModel

plt.style.use("seaborn-v0_8-whitegrid")

# --- Load Network ---

print("Loading pud_final.json network...")
network_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "networks",
    "citation_data",
    "pud_final.json",
)
with open(network_path, "r") as f:
    network_data = json.load(f)

network = nx.node_link_graph(network_data, edges="links")
print(f"Network: {len(network.nodes())} nodes, {len(network.edges())} edges")

# --- Initialize Model ---

model = VectorizedModel(
    network=network,
    n_experiments=10,
    uncertainty=0.1,
    agent_type="beta",
    tolerance_stopping=False,  # we control stopping in the loop below
    compute_convergence=False,  # we track our own statistics
)

# --- Step-by-Step Simulation ---

MAX_STEPS = 500_000
TOLERANCE = 1e-6  # stop when max per-agent credence change falls below this
ACTIVE_THR = 1e-4  # agents with |delta_belief| > this are counted as "active"
LOG_EVERY = 20_000

print(f"\nRunning step-by-step (max {MAX_STEPS} steps, tolerance {TOLERANCE})...")

mean_T0, mean_T1 = [], []
std_T0, std_T1 = [], []
frac_active = []

for _ in range(MAX_STEPS):
    prior = model.credences.copy()  # shape (N, 2)

    model.step()

    abs_change = np.abs(model.credences - prior)  # shape (N, 2)

    mean_T0.append(np.mean(abs_change[:, 0]))
    mean_T1.append(np.mean(abs_change[:, 1]))
    std_T0.append(np.std(abs_change[:, 0]))
    std_T1.append(np.std(abs_change[:, 1]))
    frac_active.append(np.mean(np.max(abs_change, axis=1) > ACTIVE_THR))

    if model.n_steps % LOG_EVERY == 0:
        print(
            f"  step {model.n_steps:>7d}: "
            f"mean_T1={mean_T1[-1]:.2e}  "
            f"std_T1={std_T1[-1]:.2e}  "
            f"frac_active={frac_active[-1]:.3f}"
        )

    if np.max(abs_change) < TOLERANCE:
        print(f"\nConverged at step {model.n_steps}")
        break

n_steps = model.n_steps
conclusion = np.mean(model.credences[:, 1] > model.credences[:, 0])
print(f"\nSteps: {n_steps}  |  Conclusion (truth share): {conclusion:.4f}")

# --- Identify t* ---

steps = np.arange(1, len(mean_T1) + 1)
mean_T0 = np.array(mean_T0)
mean_T1 = np.array(mean_T1)
std_T0 = np.array(std_T0)
std_T1 = np.array(std_T1)
frac_active = np.array(frac_active)

# t* = step where frac_active first drops below 0.5:
# half the network has committed to exploitation.
# (Peak std at step 1 is just an initialization artifact.)
t_star_idx = np.argmax(frac_active < 0.5)
if t_star_idx == 0:
    t_star_idx = len(steps) - 1  # fallback: never crossed
t_star = steps[t_star_idx]
print(f"t* (frac_active < 0.5): step {t_star}")

# --- Plot ---

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
C0, C1 = "#e74c3c", "#2980b9"
VLINE_KW = dict(color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.8)

# Panel 1: Mean belief change (same signal as convergence_studies.py)
ax = axes[0]
ax.plot(steps, mean_T0, color=C0, label="Theory 0 (Bad)", alpha=0.8, linewidth=0.9)
ax.plot(steps, mean_T1, color=C1, label="Theory 1 (Good)", alpha=0.8, linewidth=0.9)
ax.axvline(t_star, **VLINE_KW, label=f"t* = {t_star}")
ax.set_yscale("log")
ax.set_ylabel("Mean |Δbelief|", fontsize=11)
ax.set_title(
    f"Two-Phase Dynamics: Exploration → Exploitation (PUD Network, N={len(network.nodes())})",
    fontsize=12,
)
ax.legend(fontsize=10)

# Panel 2: Std dev of belief change (reveals t*)
ax = axes[1]
ax.plot(steps, std_T0, color=C0, label="Theory 0 (Bad)", alpha=0.8, linewidth=0.9)
ax.plot(steps, std_T1, color=C1, label="Theory 1 (Good)", alpha=0.8, linewidth=0.9)
ax.axvline(t_star, **VLINE_KW, label=f"t* = {t_star}")
ax.set_yscale("log")
ax.set_ylabel("Std Dev |Δbelief|", fontsize=11)
ax.legend(fontsize=10)

# Panel 3: Fraction of agents still actively updating
ax = axes[2]
ax.plot(
    steps,
    frac_active,
    color="#27ae60",
    linewidth=1.2,
    label=f"|Δbelief| > {ACTIVE_THR:.0e} (any theory)",
)
ax.axvline(t_star, **VLINE_KW, label=f"t* = {t_star}")
ax.set_xlabel("Time Step", fontsize=11)
ax.set_ylabel("Fraction Active", fontsize=11)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=10)

# Log x-axis: the interesting dynamics are front-loaded (first ~1000 steps);
# a linear axis would compress them into an invisible sliver.
for ax in axes:
    ax.set_xscale("log")

plt.tight_layout()
output_path = os.path.join(
    os.path.dirname(__file__), f"two_phase_dynamics_{n_steps}steps.png"
)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to '{output_path}'")
