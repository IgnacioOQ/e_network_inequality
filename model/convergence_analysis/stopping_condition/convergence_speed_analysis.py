#!/usr/bin/env python
# coding: utf-8

# Convergence Speed vs. Tolerance Threshold
# ==========================================
# Measures how many steps the simulation takes to stop under each tolerance
# level on the pud_final.pkl network. Produces stopping-time distributions
# (box plots) and stopping-time ratios relative to the default (5e-3).
# See STOPPING_CONDITION_ANALYSIS.md §4.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import dill
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.vectorized_model import VectorizedModel

# --- Load network ---
network_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'networks', 'citation_data', 'pud_final.pkl')
with open(network_path, 'rb') as f:
    network = dill.load(f)
print(f"Network loaded: {len(network.nodes())} nodes, {len(network.edges())} edges")

# --- Configuration ---
N_EXPERIMENTS = 10
UNCERTAINTY = 0.1
N_RUNS = 200
MAX_STEPS = 10 ** 6
DEFAULT_TOLERANCE = 5e-3

TOLERANCES = [1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6]
LABELS = [f"{t:.0e}" for t in TOLERANCES]

output_dir = os.path.dirname(__file__)

# --- Run sweep ---
print(f"\nRunning {N_RUNS} simulations per tolerance level...")

steps_by_tolerance = {}

for tol in TOLERANCES:
    label = f"{tol:.0e}"
    steps_list = []
    for seed in range(N_RUNS):
        model = VectorizedModel(
            network=network,
            n_experiments=N_EXPERIMENTS,
            agent_type="beta",
            uncertainty=UNCERTAINTY,
            tolerance=tol,
            tolerance_stopping=True,
            seed=seed,
            seeded=True,
        )
        model.run_simulation(number_of_steps=MAX_STEPS, show_bar=False)
        steps_list.append(model.n_steps)
    steps_by_tolerance[tol] = steps_list
    mean_s = np.mean(steps_list)
    hit_cap = sum(s == MAX_STEPS for s in steps_list)
    print(f"  tol={label}: mean={mean_s:.0f}  median={np.median(steps_list):.0f}  hit_cap={hit_cap}/{N_RUNS}")

# --- Build DataFrame ---
rows = []
for tol, steps_list in steps_by_tolerance.items():
    for s in steps_list:
        rows.append({"tolerance": tol, "tolerance_label": f"{tol:.0e}", "steps": s})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(output_dir, 'convergence_speed.csv'), index=False)
print(f"\nRaw data saved to convergence_speed.csv")

# --- Stopping-time ratio table ---
default_mean = np.mean(steps_by_tolerance[DEFAULT_TOLERANCE])
print(f"\nStopping-time ratio (mean_steps / mean_steps at default {DEFAULT_TOLERANCE:.0e}):")
print(f"{'Tolerance':<12} {'Mean steps':>12} {'Median steps':>14} {'Ratio':>8} {'Hit cap':>8}")
print("-" * 58)
for tol in TOLERANCES:
    s = steps_by_tolerance[tol]
    ratio = np.mean(s) / default_mean
    hit_cap = sum(x == MAX_STEPS for x in s)
    marker = " *" if tol == DEFAULT_TOLERANCE else ""
    print(f"{tol:<12.0e} {np.mean(s):>12.0f} {np.median(s):>14.0f} {ratio:>8.2f} {hit_cap:>8}{marker}")

# --- Plot 1: Box plot of stopping-time distributions ---
fig, ax = plt.subplots(figsize=(11, 5))

data_for_plot = [steps_by_tolerance[t] for t in TOLERANCES]
bp = ax.boxplot(data_for_plot, labels=LABELS, patch_artist=True, notch=False)

default_idx = TOLERANCES.index(DEFAULT_TOLERANCE)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor('#2980b9' if i == default_idx else '#bdc3c7')
    patch.set_alpha(0.8)

ax.set_yscale('log')
ax.set_xlabel('Tolerance', fontsize=12)
ax.set_ylabel('Steps to convergence (log scale)', fontsize=12)
ax.set_title(
    f'Stopping-Time Distribution by Tolerance — PUD Network (N={len(network.nodes())}, {N_RUNS} runs each)',
    fontsize=12,
)
ax.axvline(default_idx + 1, color='#2980b9', linestyle='--', alpha=0.5, linewidth=1.2,
           label=f'Default ({DEFAULT_TOLERANCE:.0e})')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_speed_boxplot.png'), dpi=150, bbox_inches='tight')
print("\nPlot saved: convergence_speed_boxplot.png")

# --- Plot 2: Mean stopping time + ratio vs. tolerance ---
fig, ax1 = plt.subplots(figsize=(9, 4))

means = [np.mean(steps_by_tolerance[t]) for t in TOLERANCES]
ratios = [m / default_mean for m in means]

color_bar = '#2980b9'
color_line = '#e74c3c'

ax1.bar(LABELS, means, color=color_bar, alpha=0.7, label='Mean steps')
ax1.set_yscale('log')
ax1.set_xlabel('Tolerance', fontsize=12)
ax1.set_ylabel('Mean steps to convergence (log)', fontsize=12, color=color_bar)
ax1.tick_params(axis='y', labelcolor=color_bar)

ax2 = ax1.twinx()
ax2.plot(LABELS, ratios, 'o-', color=color_line, linewidth=2, markersize=6, label='Stopping-time ratio')
ax2.axhline(1.0, color=color_line, linestyle='--', alpha=0.4, linewidth=1)
ax2.set_ylabel('Ratio vs. default (5e-3)', fontsize=12, color=color_line)
ax2.tick_params(axis='y', labelcolor=color_line)

ax1.set_title(
    f'Mean Convergence Steps & Stopping-Time Ratio — PUD Network ({N_RUNS} runs each)',
    fontsize=12,
)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_speed_ratio.png'), dpi=150, bbox_inches='tight')
print("Plot saved: convergence_speed_ratio.png")

plt.close('all')
print("\nDone.")
