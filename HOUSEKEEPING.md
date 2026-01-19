# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md

# Current Project Housekeeping

## Dependency Network

- **Core Dependencies**:
    - `src/net_epistemology/utils/imports.py`: Depends on (none).
- **Agent Definitions**:
    - `src/net_epistemology/core/agents.py`: Depends on `imports`.
    - `src/net_epistemology/core/vectorized_agents.py`: Depends on `imports`.
- **Model Definitions**:
    - `src/net_epistemology/core/model.py`: Depends on `imports`, `core.agents`.
    - `src/net_epistemology/core/vectorized_model.py`: Depends on `imports`.
- **Simulation Functions**:
    - `src/net_epistemology/simulation/vectorized_simulation_functions.py`: Depends on `core.vectorized_model`, `imports`, `utils.network_generation`, `utils.network_utils`.
    - `src/net_epistemology/simulation/simulation_functions.py`: Depends on `core.agents`, `core.model`, `imports`, `utils.network_generation`, `utils.network_utils`.
- **Network Handling**:
    - `src/net_epistemology/utils/network_generation.py`: Depends on (none).
    - `src/net_epistemology/utils/network_utils.py`: Depends on (none).
- **Testing**:
    - `tests/unit_tests.py`: Depends on `imports`, `core.agents`.
    - `tests/test_vectorization.py`: Depends on `core.model`, `core.vectorized_model`.

## Latest Report

**Execution Date:** 2026-01-19

**Test Results:**

*   **Unit Tests (`tests/unit_tests.py`):** PASSED (8 tests)
*   **Vectorization Tests (`tests/test_vectorization.py`):** PASSED (4 tests)
*   **Notebook Verification (Smoke Tests):**
    *   `tests/basic_model_testing_smoke.py`: PASSED (Verified Bayes and Beta agents on dummy/loaded network)
    *   `tests/run_simulations_smoke.py`: PASSED (2 parallel simulations completed successfully)

**Summary:**
All core tests and new smoke tests pass successfully. Notebook logic verified via converted smoke test scripts.
