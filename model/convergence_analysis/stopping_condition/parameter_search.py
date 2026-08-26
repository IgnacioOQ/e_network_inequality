#!/usr/bin/env python
# coding: utf-8

# This file is a script, not an importable module: the analysis below runs at
# module level, so importing it would execute a full simulation (minutes) and
# write plots into the working tree. Fail fast and say so instead.
if __name__ != "__main__":
    raise ImportError(
        f"{__name__} is an analysis script, not a library module. "
        f"Run it directly: python {__file__}"
    )

# Parameter Search: Tolerance × Uncertainty × n_experiments
# ==========================================================
# For each combination of (tolerance, uncertainty, n_experiments), runs N_RUNS
# independent simulations on pud_network.pkl and records:
#   (a) truth_share  — fraction of agents converging to the correct theory
#   (b) steps_taken  — steps until the stopping condition fires
#
# uncertainty = bandit gap: p_good = 0.5 + uncertainty, p_bad = 0.5.
# epsilon (exploration) is hardcoded to 0 in VectorizedModel — agents are
# always greedy; the "exploration phase" arises solely from uncertain priors.
#
# See STOPPING_CONDITION_ANALYSIS.md §4 and §7.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import dill
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

from model.vectorized_model import VectorizedModel

# --- Load network ---
network_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'networks', 'citation_data', 'pud_network.pkl')
with open(network_path, 'rb') as f:
    network = dill.load(f)
print(f"Network loaded: {len(network.nodes())} nodes, {len(network.edges())} edges")

# --- Parameter grid ---
TOLERANCES    = [1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6]
UNCERTAINTIES = [0.01, 0.05, 0.1, 0.2, 0.5]
N_EXPERIMENTS_GRID = [5, 10, 20]

N_RUNS    = 100
MAX_STEPS = 10 ** 6

output_dir = os.path.dirname(__file__)

# --- Run grid ---
total = len(TOLERANCES) * len(UNCERTAINTIES) * len(N_EXPERIMENTS_GRID) * N_RUNS
print(f"\nTotal simulations: {total}")

rows = []
done = 0

for n_exp, uncertainty, tolerance in itertools.product(N_EXPERIMENTS_GRID, UNCERTAINTIES, TOLERANCES):
    for seed in range(N_RUNS):
        model = VectorizedModel(
            network=network,
            n_experiments=n_exp,
            agent_type="beta",
            uncertainty=uncertainty,
            tolerance=tolerance,
            tolerance_stopping=True,
            seed=seed,
            seeded=True,
        )
        model.run_simulation(number_of_steps=MAX_STEPS, show_bar=False)

        credences = model.credences
        truth_share = float(np.mean(credences[:, 1] > credences[:, 0]))

        rows.append({
            "tolerance":     tolerance,
            "uncertainty":   uncertainty,
            "n_experiments": n_exp,
            "seed":          seed,
            "steps":         model.n_steps,
            "truth_share":   truth_share,
            "hit_cap":       int(model.n_steps == MAX_STEPS),
        })
        done += 1

    if done % (N_RUNS * len(TOLERANCES)) == 0:
        print(f"  {done}/{total} done  "
              f"(n_exp={n_exp}, uncertainty={uncertainty:.2f}, tol={tolerance:.0e} complete)")

df = pd.DataFrame(rows)
csv_path = os.path.join(output_dir, 'parameter_search.csv')
df.to_csv(csv_path, index=False)
print(f"\nRaw data saved to parameter_search.csv ({len(df)} rows)")

# --- Summary table ---
summary = (
    df.groupby(["n_experiments", "uncertainty", "tolerance"])
    .agg(
        mean_steps=("steps", "mean"),
        std_steps=("steps", "std"),
        mean_truth=("truth_share", "mean"),
        std_truth=("truth_share", "std"),
        hit_cap_frac=("hit_cap", "mean"),
    )
    .reset_index()
)
summary_path = os.path.join(output_dir, 'parameter_search_summary.csv')
summary.to_csv(summary_path, index=False)
print("Summary saved to parameter_search_summary.csv")

# --- Plots (one figure per n_experiments value) ---

def make_heatmaps(n_exp_val, df_sub, suffix):
    tol_labels  = [f"{t:.0e}" for t in TOLERANCES]
    unc_labels  = [str(u) for u in UNCERTAINTIES]

    # Build 2D arrays: rows = uncertainty (ascending), cols = tolerance (descending strictness)
    mean_steps  = np.zeros((len(UNCERTAINTIES), len(TOLERANCES)))
    std_steps   = np.zeros_like(mean_steps)
    mean_truth  = np.zeros_like(mean_steps)
    std_truth   = np.zeros_like(mean_steps)

    for i, unc in enumerate(UNCERTAINTIES):
        for j, tol in enumerate(TOLERANCES):
            sub = df_sub[(df_sub.uncertainty == unc) & (df_sub.tolerance == tol)]["steps"]
            ts  = df_sub[(df_sub.uncertainty == unc) & (df_sub.tolerance == tol)]["truth_share"]
            mean_steps[i, j] = sub.mean()
            std_steps[i, j]  = sub.std()
            mean_truth[i, j] = ts.mean()
            std_truth[i, j]  = ts.std()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Parameter Search — n_experiments={n_exp_val}, PUD network ({N_RUNS} runs each)",
        fontsize=13,
    )

    panels = [
        (axes[0, 0], mean_steps,  "Mean steps to convergence",          "YlOrRd"),
        (axes[0, 1], std_steps,   "Std dev of steps",                   "YlOrRd"),
        (axes[1, 0], mean_truth,  "Mean truth share",                   "RdYlGn"),
        (axes[1, 1], std_truth,   "Std dev of truth share (variance)",  "RdYlGn_r"),
    ]

    for ax, data, title, cmap in panels:
        im = ax.imshow(data, aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(TOLERANCES)))
        ax.set_xticklabels(tol_labels, fontsize=9)
        ax.set_yticks(range(len(UNCERTAINTIES)))
        ax.set_yticklabels(unc_labels, fontsize=9)
        ax.set_xlabel("Tolerance", fontsize=10)
        ax.set_ylabel("Uncertainty (bandit gap)", fontsize=10)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax)
        # Annotate cells
        for i in range(len(UNCERTAINTIES)):
            for j in range(len(TOLERANCES)):
                val = data[i, j]
                fmt = f"{val:.0f}" if "steps" in title.lower() else f"{val:.3f}"
                ax.text(j, i, fmt, ha='center', va='center', fontsize=7,
                        color='black' if 0.2 < (data[i,j] - data.min()) / (data.max() - data.min() + 1e-9) < 0.8 else 'white')

    plt.tight_layout()
    out = os.path.join(output_dir, f'parameter_search_heatmap{suffix}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: parameter_search_heatmap{suffix}.png")

for n_exp_val in N_EXPERIMENTS_GRID:
    df_sub = df[df.n_experiments == n_exp_val]
    make_heatmaps(n_exp_val, df_sub, f"_nexp{n_exp_val}")

# --- Line plots: steps vs tolerance, one line per uncertainty, faceted by n_experiments ---
fig, axes = plt.subplots(1, len(N_EXPERIMENTS_GRID), figsize=(5 * len(N_EXPERIMENTS_GRID), 5), sharey=True)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(UNCERTAINTIES)))

for ax, n_exp_val in zip(axes, N_EXPERIMENTS_GRID):
    df_sub = df[df.n_experiments == n_exp_val]
    for unc, color in zip(UNCERTAINTIES, colors):
        means = [
            df_sub[(df_sub.uncertainty == unc) & (df_sub.tolerance == tol)]["steps"].mean()
            for tol in TOLERANCES
        ]
        ax.plot([f"{t:.0e}" for t in TOLERANCES], means, 'o-', color=color,
                label=f"unc={unc}", linewidth=1.8, markersize=5)
    ax.set_yscale('log')
    ax.set_title(f"n_experiments={n_exp_val}", fontsize=11)
    ax.set_xlabel("Tolerance", fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel("Mean steps (log)", fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
fig.suptitle("Convergence Speed: Steps vs. Tolerance by Uncertainty — PUD Network", fontsize=12)
plt.tight_layout()
out = os.path.join(output_dir, 'parameter_search_lines.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print("Line plot saved: parameter_search_lines.png")

print("\nDone.")
