#!/usr/bin/env python
# coding: utf-8

# After the model stops (default tol=5e-3), resume from that exact state and
# measure how much credences drift. Directly tests whether the stopping point
# is truly stable. See STOPPING_CONDITION_ANALYSIS.md §5 Script 2.

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
STOP_TOLERANCE = 5e-3

# Steps to continue after initial stop
CONTINUATION_STEPS = [10_000, 50_000, 100_000]

output_path = os.path.join(os.path.dirname(__file__), 'post_stopping_drift.csv')


def resume_from_state(network, alphas_betas_saved, credences_saved, extra_steps, uncertainty):
    """Create a fresh model and inject saved state, then run extra_steps."""
    model = VectorizedModel(
        network=network,
        n_experiments=N_EXPERIMENTS,
        agent_type="beta",
        uncertainty=uncertainty,
        tolerance=STOP_TOLERANCE,
        tolerance_stopping=False,
        tstep_stopping=True,
        seeded=False,
    )
    # Inject saved state directly — model internals support this since step()
    # only reads self.alphas_betas and self.credences.
    model.alphas_betas = alphas_betas_saved.copy()
    model.credences = credences_saved.copy()
    model.run_simulation(number_of_steps=extra_steps, show_bar=False)
    return model.credences.copy()


rows = []

for run_idx in range(N_RUNS):
    # Phase 1: run until tolerance stopping fires
    model = VectorizedModel(
        network=network,
        n_experiments=N_EXPERIMENTS,
        agent_type="beta",
        uncertainty=UNCERTAINTY,
        tolerance=STOP_TOLERANCE,
        tolerance_stopping=True,
        seed=run_idx,
        seeded=True,
    )
    model.run_simulation(number_of_steps=MAX_STEPS, show_bar=False)

    stop_step = model.n_steps
    credences_at_stop = model.credences.copy()  # (N, 2)
    alphas_betas_at_stop = model.alphas_betas.copy()  # (N, 2, 2)

    believes_truth_at_stop = credences_at_stop[:, 1] > credences_at_stop[:, 0]
    truth_share_at_stop = np.mean(believes_truth_at_stop)

    # Phase 2: resume for each continuation length
    for K in CONTINUATION_STEPS:
        credences_after = resume_from_state(network, alphas_betas_at_stop, credences_at_stop, K, UNCERTAINTY)

        drift = np.mean(np.abs(credences_after - credences_at_stop))
        max_drift = np.max(np.abs(credences_after - credences_at_stop))

        believes_truth_after = credences_after[:, 1] > credences_after[:, 0]
        truth_share_after = np.mean(believes_truth_after)

        # Agents that flipped theory (crossed 0.5 boundary)
        flipped = np.any(believes_truth_after != believes_truth_at_stop)
        n_flipped = np.sum(believes_truth_after != believes_truth_at_stop)

        rows.append({
            "run": run_idx,
            "stop_step": stop_step,
            "truth_share_at_stop": truth_share_at_stop,
            "continuation_steps": K,
            "mean_drift": drift,
            "max_drift": max_drift,
            "truth_share_after": truth_share_after,
            "any_flipped": int(flipped),
            "n_flipped": n_flipped,
        })

    if (run_idx + 1) % 50 == 0:
        print(f"  {run_idx + 1}/{N_RUNS} runs done")

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows to {output_path}")

print("\nSummary (mean drift and flip rate per continuation length):")
summary = df.groupby("continuation_steps")[["mean_drift", "max_drift", "any_flipped", "n_flipped"]].mean()
print(summary.to_string())
