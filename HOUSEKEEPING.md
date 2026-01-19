# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md

# Current Project Housekeeping

## Dependency Network

- **Core Dependencies**:
    - `src/net_epistemology/utils/imports.py`: Used by all other modules.
- **Agent Definitions**:
    - `src/net_epistemology/core/agents.py`: Depends on `imports`.
    - `src/net_epistemology/core/vectorized_agents.py`: Depends on `imports`.
- **Model Definitions**:
    - `src/net_epistemology/core/model.py`: Depends on `imports`, `agents`.
    - `src/net_epistemology/core/vectorized_model.py`: Depends on `imports`, `vectorized_agents`.
- **Simulation Functions**:
    - `src/net_epistemology/simulation/simulation_functions.py`: Depends on `imports`, `core.agents`, `core.model`, `utils.network_utils`, `utils.network_generation`.
    - `src/net_epistemology/simulation/vectorized_simulation_functions.py`: Depends on `imports`, `core.vectorized_model`, `utils.network_utils`, `utils.network_generation`.
- **Network Handling**:
    - `src/net_epistemology/utils/network_generation.py`: Depends on `imports`.
    - `src/net_epistemology/utils/network_utils.py`: Depends on `imports`.
- **Testing**:
    - `tests/unit_tests.py`: Depends on `imports`, `core.agents`.
    - `tests/test_vectorization.py`: Depends on `core.model`, `core.vectorized_model`.

## Latest Report

**Execution Date:** 2026-01-18

**Test Results:**

*   **Unit Tests (`tests/unit_tests.py`):** PASSED (8 tests)
*   **Vectorization Tests (`tests/test_vectorization.py`):** PASSED (4 tests)
*   **Script Verification (Smoke Tests):**
    *   `notebooks/basic_model_testing.py` (derived from notebook): PASSED.
    *   `notebooks/vectorized_basic_model_testing.py` (derived from notebook): PASSED.
    *   `notebooks/run_simulations_test.py` (derived from notebook): PASSED (with modifications for headless environment and reduced parameters).

**Summary:**
All core tests pass successfully. The notebook verification scripts run without errors when adapted for the headless environment (reduced steps/simulations, headless matplotlib, fixed paths). The `run_simulations_test.py` script was modified to use `run_simulation_with_params` instead of the missing wrapper and to handle the missing `perc_pruned_lcc.pkl` file (used `perceptron_final.pkl` instead).
