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

**Execution Date:** 2026-01-20 (Current)

**Test Results:**

*   **Unit Tests (`tests/unit_tests.py`):** PASSED (8 tests)
*   **Vectorization Tests (`tests/test_vectorization.py`):** PASSED (4 tests)
*   **Script Verification:**
    *   `basic_model_testing_script.py`: PASSED.
    *   `vectorized_basic_model_testing_script.py`: PASSED.
    *   `run_simulations_test_script.py`: PASSED (Testing Generated section). "Testing Empirical" section skipped due to missing `perc_pruned_lcc.pkl` file.

**Summary:**
All core tests pass successfully. Verification scripts for notebooks ran successfully in headless mode after being converted and adjusted for paths and matplotlib backend. The simulation logic is robust. The empirical network test was skipped because the specific data file is missing from the repository, but the generated network simulations confirm functionality.
