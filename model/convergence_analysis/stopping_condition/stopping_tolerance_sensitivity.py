#!/usr/bin/env python
# coding: utf-8

# Sweep tolerance values and compare final simulation outputs.
# See STOPPING_CONDITION_ANALYSIS.md §5 Script 1.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import dill
import numpy as np
import pandas as pd

from model.vectorized_model import VectorizedModel

# --- Load network ---
network_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'networks', 'citation_data', 'pud_network.pkl')
with open(network_path, 'rb') as f:
    network = dill.load(f)
print(f"Network loaded: {len(network.nodes())} nodes, {len(network.edges())} edges")

# --- Configuration ---
N_EXPERIMENTS = 10
UNCERTAINTY = 0.1
N_RUNS = 200
MAX_STEPS = 10 ** 6

TOLERANCES = [1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6]

output_path = os.path.join(os.path.dirname(__file__), 'tolerance_sensitivity.csv')


def run_single(network, tolerance, tstep, seed):
    model = VectorizedModel(
        network=network,
        n_experiments=N_EXPERIMENTS,
        agent_type="beta",
        uncertainty=UNCERTAINTY,
        tolerance=tolerance,
        tolerance_stopping=(not tstep),
        tstep_stopping=tstep,
        seed=seed,
        seeded=True,
    )
    model.run_simulation(number_of_steps=MAX_STEPS, show_bar=False)

    credences = model.credences  # (N, 2)
    believes_truth = credences[:, 1] > credences[:, 0]
    truth_share = np.mean(believes_truth)
    mean_credence_correct = np.mean(credences[believes_truth, 1]) if believes_truth.any() else 0.0
    # Consensus: all agents agree on the same theory
    fraction_consensus = 1.0 if (believes_truth.all() or (~believes_truth).all()) else 0.0

    return {
        "tolerance": "fixed" if tstep else tolerance,
        "seed": seed,
        "steps_taken": model.n_steps,
        "truth_share": truth_share,
        "mean_credence_correct": mean_credence_correct,
        "fraction_consensus": fraction_consensus,
    }


rows = []
configs = [(tol, False) for tol in TOLERANCES] + [(None, True)]
total = len(configs) * N_RUNS

done = 0
for tolerance, tstep in configs:
    label = "fixed" if tstep else f"{tolerance:.0e}"
    print(f"Running tolerance={label} ({N_RUNS} runs)...")
    for run_idx in range(N_RUNS):
        seed = run_idx  # same seeds across all tolerances for comparability
        row = run_single(network, tolerance if not tstep else 5e-3, tstep, seed)
        rows.append(row)
        done += 1
    print(f"  done. ({done}/{total})")

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows to {output_path}")

# Summary
print("\nSummary (mean ± std per tolerance):")
summary = df.groupby("tolerance")[["steps_taken", "truth_share", "fraction_consensus"]].agg(["mean", "std"])
print(summary.to_string())
