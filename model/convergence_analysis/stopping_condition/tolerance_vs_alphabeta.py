#!/usr/bin/env python
# coding: utf-8

# Compare stopping on credences (current behaviour) vs. stopping on raw
# alpha/beta parameters (stricter criterion). Same tolerance sweep for both.
# See STOPPING_CONDITION_ANALYSIS.md §5 Script 3.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd

from model.vectorized_model import VectorizedModel
from networks.network_generation import barabasi_albert_directed

# --- Configuration ---
N_NODES = 30
M_EDGES = 2
N_EXPERIMENTS = 10
UNCERTAINTY = 0.1
N_RUNS = 200
MAX_STEPS = 10 ** 6

TOLERANCES = [1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6]

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results'))
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'alphabeta_stopping_comparison.csv')


def run_credence_stopping(network, tolerance, seed):
    """Criterion A: current behaviour — allclose on credences."""
    model = VectorizedModel(
        network=network,
        n_experiments=N_EXPERIMENTS,
        agent_type="beta",
        uncertainty=UNCERTAINTY,
        tolerance=tolerance,
        tolerance_stopping=True,
        seed=seed,
        seeded=True,
    )
    model.run_simulation(number_of_steps=MAX_STEPS, show_bar=False)
    truth_share = np.mean(model.credences[:, 1] > model.credences[:, 0])
    return model.n_steps, truth_share


def run_alphabeta_stopping(network, tolerance, seed):
    """Criterion B: allclose on alphas_betas — stricter proposed criterion.

    Implemented as a manual loop using tstep_stopping=False and tolerance_stopping=False,
    with our own check each step.
    """
    model = VectorizedModel(
        network=network,
        n_experiments=N_EXPERIMENTS,
        agent_type="beta",
        uncertainty=UNCERTAINTY,
        tolerance_stopping=False,
        tstep_stopping=False,
        seed=seed,
        seeded=True,
    )

    for _ in range(MAX_STEPS):
        ab_prior = model.alphas_betas.copy()
        model.step()
        if np.allclose(ab_prior, model.alphas_betas, rtol=tolerance, atol=tolerance):
            break

    truth_share = np.mean(model.credences[:, 1] > model.credences[:, 0])
    return model.n_steps, truth_share


rows = []
total = len(TOLERANCES) * N_RUNS

done = 0
for tolerance in TOLERANCES:
    print(f"Running tolerance={tolerance:.0e} ({N_RUNS} runs)...")
    for run_idx in range(N_RUNS):
        network = barabasi_albert_directed(N_NODES, M_EDGES)

        steps_A, ts_A = run_credence_stopping(network, tolerance, seed=run_idx)
        steps_B, ts_B = run_alphabeta_stopping(network, tolerance, seed=run_idx)

        rows.append({
            "tolerance": tolerance,
            "run": run_idx,
            "steps_credence": steps_A,
            "truth_share_credence": ts_A,
            "steps_alphabeta": steps_B,
            "truth_share_alphabeta": ts_B,
            "steps_ratio": steps_B / steps_A if steps_A > 0 else np.nan,
            "truthshare_diff": ts_B - ts_A,
        })
        done += 1

    print(f"  done. ({done}/{total})")

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows to {output_path}")

print("\nSummary (mean per tolerance):")
summary = df.groupby("tolerance")[
    ["steps_credence", "steps_alphabeta", "steps_ratio",
     "truth_share_credence", "truth_share_alphabeta", "truthshare_diff"]
].mean()
print(summary.to_string())
