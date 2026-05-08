# Snapshot Feature Implementation Plan

## Goal

Implement a snapshot feature within the vectorized model to record the **mean truth share** and **maximum absolute belief change** at fixed intervals during long-running simulations (≥ 100,000 steps). This enables analysis of how quickly the belief-updating process stabilizes over time — both in terms of convergence to the truth and magnitude of belief adjustments.

---

## Design Decisions

| Decision | Choice |
|---|---|
| **Snapshot interval** | Fixed frequency (e.g., every `snapshot_interval` steps) |
| **Belief change metric** | Maximum absolute change across all agents at a given step |
| **Agent scope** | `beta` agents only |

---

## Changes Made

### `model/vectorized_model.py`

- **`__init__`**:
  - Added `snapshot_interval: int = 0` parameter.
  - Initialized `self.snapshots` as a list of dicts: `{"step": int, "truth_share": float, "max_belief_change": float}`.
- **`run_simulation`**:
  - Inside the main step loop, at every `snapshot_interval`-th step:
    - Computes the current **truth share** (fraction of beta agents holding the true hypothesis).
    - Computes the **maximum absolute belief change** (max over all agents of `|credence_now - credence_prior|`).
    - Appends `{"step": step_num, "truth_share": ..., "max_belief_change": ...}` to `self.snapshots`.

---

### `model/vectorized_simulation_functions.py`

- Updated `run_vectorized_simulation_with_params` to:
  - Accept a new `snapshot_interval` keyword argument (default `0`).
  - Pass it through to `VectorizedModel(...)`.
  - After simulation, attach `my_model.snapshots` to the returned `result_dict` under the key `"snapshots"`.

---

## Usage Example

```python
from model.vectorized_simulation_functions import run_vectorized_simulation_with_params

params = {
    "network": my_graph,
    "n_experiments": 100,
    "uncertainty": 0.0001,
    "agent_type": "beta",
    "number_of_steps": 100000,
    "snapshot_interval": 1000,  # Record every 1000 steps
}

result = run_vectorized_simulation_with_params(params)

# Access snapshots:
snapshots = result["snapshots"]
# snapshots is a list of dicts:
# [{"step": 1000, "truth_share": 0.72, "max_belief_change": 0.003}, ...]
```

---

## Testing Notebook

A new notebook `testing/notebooks/basic_model_testing_v2.ipynb` is planned to:

1. Run a simulation with `snapshot_interval=1000`.
2. Plot **truth share** and **max absolute belief change** over time as two side-by-side panels.
3. Serve as a visual sanity check for the snapshot feature.

---

## Verification Checklist

- [ ] `self.snapshots` has exactly `number_of_steps // snapshot_interval` entries.
- [ ] `truth_share` values are in `[0, 1]`.
- [ ] `max_belief_change` values decrease monotonically (or near-monotonically) toward `0`.
- [ ] Overhead of snapshot computation is negligible vs. total simulation time.
- [ ] `snapshot_interval=0` (default) produces no snapshots and no performance cost.
