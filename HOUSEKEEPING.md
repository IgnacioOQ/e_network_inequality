# Housekeeping Workflow
- status: active
- type: workflow
- description: Routine sanity check for the e_network_inequality repository — runs the unit test suite, checks for import errors, verifies notebooks are intact, and keeps the codebase clean and functional.
- injection: procedural
- volatility: evolving
- last_checked: 2026-05-08
<!-- content -->
This workflow is the routine sanity check for the `e_network_inequality` repository. It covers three concerns: (1) running the unit test suite to verify model correctness; (2) checking that the codebase imports cleanly and has no dead dependencies; and (3) ensuring notebooks and data files remain intact. Run this workflow after any significant batch of code changes, before committing, and before running large-scale simulations.

---

## Phase 1 — Context Load

Before running checks, orient yourself:

1. Read [README.md](README.md) — understand the project structure and current status.
2. Read [TODO_WORKFLOW.md](TODO_WORKFLOW.md) — check for any blocked or in-progress tasks that might affect what to test.
3. Read [WORKLOG.md](WORKLOG.md) — check the most recent entry to understand what changed since the last housekeeping run.

**Exit criterion:** You understand the current state of the project and which modules were recently modified.

---

## Phase 2 — Run Unit Tests

**Goal:** Verify that all model logic, stopping conditions, agent behavior, and vectorization are correct.

### Step 1 — Full test suite

Run from the project root:

```bash
.venv/bin/python -m unittest discover -s testing/unit_tests -v
```

Expected test files and what they cover:

| File | Coverage |
|:---|:---|
| `test_vectorization.py` | Validates that `VectorizedModel` produces identical results to the original `Model` given the same seed |
| `test_stopping_conditions.py` | Verifies tolerance-stopping, step-stopping, and early-exit behavior of `VectorizedModel` |
| `test_agents.py` | Checks that `BetaAgent` and `BayesAgent` update beliefs correctly |
| `test_mc_analysis.py` | Validates Markov Chain analysis utilities |

**Exit criterion:** All tests pass with `OK`. No failures or errors.

### Step 2 — Snapshot feature regression

After any change to `VectorizedModel` or `vectorized_simulation_functions.py`, run a quick snapshot smoke test:

```bash
.venv/bin/python -c "
import sys, json
sys.path.insert(0, '.')
import networkx as nx
import pickle

with open('networks/citation_data/pud_network.pkl', 'rb') as f:
    G = pickle.load(f)
mapping = {node: idx for idx, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

from model.vectorized_model import VectorizedModel
model = VectorizedModel(G, n_experiments=5, uncertainty=0.001,
                        agent_type='beta', snapshot_interval=1000,
                        tolerance_stopping=False)
model.run_simulation(number_of_steps=5000, show_bar=False)

snaps = model.snapshots
assert len(snaps['step']) == 5, f'Expected 5 snapshots, got {len(snaps[\"step\"])}'
assert all(0 <= v <= 1 for v in snaps['truth_share']), 'Truth share out of range'
assert all(v >= 0 for v in snaps['max_belief_change']), 'Negative belief change'
print('Snapshot smoke test PASSED')
"
```

**Exit criterion:** Script prints `Snapshot smoke test PASSED`.

---

## Phase 3 — Import and Dependency Check

**Goal:** Ensure all modules import cleanly from the project root with no broken dependencies.

### Step 1 — Core module imports

```bash
.venv/bin/python -c "
from utils.imports import *
from model.agents import BetaAgent, BayesAgent
from model.model import Model
from model.vectorized_model import VectorizedModel
from model.vectorized_simulation_functions import run_vectorized_simulation_with_params
from utils.network_utils import *
from networks.network_generation import *
print('All core imports OK')
"
```

### Step 2 — Network files check

Verify the three empirical networks are loadable:

```bash
.venv/bin/python -c "
import pickle, networkx as nx

networks = [
    'networks/citation_data/pud_network.pkl',
    'networks/citation_data/tobacco_network.pkl',
    'networks/citation_data/ego_network.pkl',
]
for path in networks:
    with open(path, 'rb') as f:
        G = pickle.load(f)
    print(f'{path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
print('All networks loaded OK')
"
```

**Exit criterion:** All imports succeed and all three networks load with expected node/edge counts.

---

## Phase 4 — Notebook Integrity Check

**Goal:** Verify that key notebooks are valid JSON and have not been accidentally corrupted.

```bash
.venv/bin/python -c "
import json, glob

notebooks = glob.glob('**/*.ipynb', recursive=True)
notebooks = [nb for nb in notebooks if '.venv' not in nb]
errors = []
for nb_path in notebooks:
    try:
        with open(nb_path) as f:
            nb = json.load(f)
        n_cells = len(nb.get('cells', []))
        print(f'  OK  [{n_cells} cells] {nb_path}')
    except Exception as e:
        errors.append((nb_path, str(e)))
        print(f'  ERR {nb_path}: {e}')

if errors:
    print(f'\n{len(errors)} notebook(s) failed to parse.')
else:
    print(f'\nAll {len(notebooks)} notebooks OK.')
"
```

**Exit criterion:** All notebooks parse as valid JSON with no errors.

---

## Phase 5 — Code Cleanliness Check (Optional)

**Goal:** Catch obvious unused imports and dead code. Run only when doing a deeper cleanup pass.

### Unused imports (ruff)

```bash
.venv/bin/pip install ruff --quiet
.venv/bin/ruff check model/ utils/ networks/ --select F401 --statistics
```

### Dead code (vulture)

```bash
.venv/bin/pip install vulture --quiet
.venv/bin/python -m vulture model/ utils/ networks/ --min-confidence 80
```

Review output carefully — some "dead code" may be intentionally kept as API surface. Do not delete without verifying.

**Exit criterion:** No obvious unused public functions in core model modules.

---

## Phase 6 — Report and Log

After completing the checks, append a brief entry to [WORKLOG.md](WORKLOG.md):

```markdown
### YYYY-MM-DD: Housekeeping Run
- id: worklog.YYYY_MM_DD_housekeeping
- status: done
- type: log
- last_checked: YYYY-MM-DD
<!-- content -->
**Tests:** <N passed, N failed>
**Imports:** <OK / issues>
**Networks:** <OK / issues>
**Notebooks:** <N OK, N failed>
**Notes:** <Any notable findings or fixes applied>
```

**Exit criterion:** WORKLOG.md updated with today's run.

---

## Quick Reference — Housekeeping Checklist

```
[ ] Phase 1: README, TODO_WORKFLOW, WORKLOG reviewed
[ ] Phase 2a: .venv/bin/python -m unittest discover -s testing/unit_tests -v — all pass
[ ] Phase 2b: Snapshot smoke test — PASSED
[ ] Phase 3a: Core module imports — OK
[ ] Phase 3b: Network files loadable — OK
[ ] Phase 4:  All notebooks parse as valid JSON — OK
[ ] Phase 5:  (Optional) ruff + vulture checks reviewed
[ ] Phase 6:  WORKLOG.md updated
```

---

## Latest Report

**Date:** 2026-05-08
**Trigger:** Initial HOUSEKEEPING.md creation after snapshot feature implementation.

### Test results
- `unittest discover -s testing/unit_tests`: Not run (initial creation). Run before next commit.

### Import check
- All core imports: ✅ (verified via smoke test during snapshot implementation)
- Networks: PUD (87 nodes, 160 edges), Tobacco, Ego — ✅

### Snapshot smoke test
- 5 snapshots at steps [1000, 2000, 3000, 4000, 5000] ✅
- Truth shares in [0, 1] ✅
- Max belief change decreasing ✅

### Files modified in this session
- `model/vectorized_model.py`: Added `snapshot_interval`, `self.snapshots`, per-step capture logic.
- `model/vectorized_simulation_functions.py`: Threaded `snapshot_interval` through wrapper.
- `model/convergence_analysis/stopping_condition/A. 100k Stopping Study.ipynb`: New snapshot analysis section.
- `testing/notebooks/basic_model_testing_v2.ipynb`: New notebook with snapshot plots.
- `README.md`: Added "Codebase Architecture and Execution Flow" section.
- `ADD_SNAPSHOT_PLAN.md`: Design document saved at project root.
- `WORKLOG.md`: Created (append-only working log).
- `TODO_WORKFLOW.md`: Created (renamed from WORKPLAN.md with type: workflow).
- `HOUSEKEEPING.md`: This file created.
