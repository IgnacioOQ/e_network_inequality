#!/usr/bin/env python
# coding: utf-8

# Reference (local, single-network) driver for the CHOICE-STABILITY stopping
# criterion: stop when *every* agent's chosen theory (greedy argmax of its
# credence, i.e. credences[:,1] > credences[:,0] since epsilon=0) has been
# unchanged for the last W steps -- equivalently, no agent has crossed the 0.5
# decision boundary in the last W steps. See CHOICE_STABILITY_STOPPING_PLAN.md
# and STOPPING_CONDITION_ANALYSIS.md (the Colab notebook is the scaled entry
# point; this script is for local testing/debugging).
#
# Efficiency (record-once): each run is executed ONCE with the LARGEST window
# under native choice_stability_stopping plus record_choice_flips=True. It stops
# as soon as W_max flip-free steps accumulate (or hits MAX_STEPS if it never
# stabilizes), capturing the (step, truth_share) flip history. That history
# contains every flip needed to derive the stop step and truth share for EVERY
# smaller window offline -- the window axis costs no extra simulation, and
# truth_share is constant between flips. A subset is cross-checked against
# per-window native runs to confirm the two agree.

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
UNCERTAINTY = 0.001
N_RUNS = 50
MAX_STEPS = 100_000
WINDOWS = [100, 250, 500, 1000]
CROSS_CHECK_RUNS = 5   # runs to also execute in native mode as an equivalence check

output_path = os.path.join(os.path.dirname(__file__), 'choice_stability_stopping.csv')


def offline_stop(flip_history, window, max_steps):
    """Derive (stop_step, truth_share, stabilized) for a window from a
    record_choice_flips history [(step, truth_share), ...] (baseline entry at
    run start plus one per flip step). The first inter-event gap of length
    >= window fixes the stop; if none exists the run did not stabilize."""
    steps = [s for s, _ in flip_history] + [max_steps]
    shares = [t for _, t in flip_history]
    for i in range(len(flip_history)):
        if steps[i + 1] - steps[i] >= window:
            return steps[i] + window, shares[i], True
    return max_steps, shares[-1] if shares else np.nan, False


rows = []

for run_idx in range(N_RUNS):
    # Single record-once run: native stopping at the LARGEST window, recording
    # flips. Stops early once W_max flip-free steps accumulate. seed=run_idx
    # matches the sibling reference scripts in this folder; the scaled notebook
    # uses numpy.random.SeedSequence.spawn for independent reproducible streams.
    model = VectorizedModel(
        network=network,
        n_experiments=N_EXPERIMENTS,
        agent_type="beta",
        uncertainty=UNCERTAINTY,
        tolerance_stopping=False,
        choice_stability_stopping=True,
        choice_stability_window=max(WINDOWS),
        record_choice_flips=True,
        seed=run_idx,
        seeded=True,
    )
    model.run_simulation(number_of_steps=MAX_STEPS, show_bar=False)
    flips = model.choice_flip_history

    for window in WINDOWS:
        stop_step, truth_share, stabilized = offline_stop(flips, window, MAX_STEPS)
        rows.append({
            "run": run_idx,
            "window": window,
            "stop_step": stop_step,
            "truth_share": truth_share,
            "stabilized": int(stabilized),
        })

    # Equivalence cross-check on the first few runs: native mode must reproduce
    # the offline-derived stop step and truth share for the same seed.
    if run_idx < CROSS_CHECK_RUNS:
        for window in WINDOWS:
            native = VectorizedModel(
                network=network,
                n_experiments=N_EXPERIMENTS,
                agent_type="beta",
                uncertainty=UNCERTAINTY,
                tolerance_stopping=False,
                choice_stability_stopping=True,
                choice_stability_window=window,
                seed=run_idx,
                seeded=True,
            )
            native.run_simulation(number_of_steps=MAX_STEPS, show_bar=False)
            off_step, off_share, _ = offline_stop(flips, window, MAX_STEPS)
            assert native.n_steps == off_step, (
                f"native/offline stop mismatch run={run_idx} W={window}: "
                f"{native.n_steps} vs {off_step}")
            assert np.isclose(native.conclusion, off_share), (
                f"native/offline truth-share mismatch run={run_idx} W={window}")

    if (run_idx + 1) % 10 == 0:
        print(f"  {run_idx + 1}/{N_RUNS} runs done")

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows to {output_path}")
print(f"Equivalence cross-check passed on the first {CROSS_CHECK_RUNS} runs.")

print("\nSummary per window (mean stop step, mean truth share, stabilization rate):")
summary = df.groupby("window").agg(
    mean_stop_step=("stop_step", "mean"),
    mean_truth_share=("truth_share", "mean"),
    stabilized_rate=("stabilized", "mean"),
)
print(summary.to_string())
