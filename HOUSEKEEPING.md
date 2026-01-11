# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md

# Current Project Housekeeping

## Dependency Network

- **Core Dependencies (`imports.py`)**: Used by all other modules.
- **Agent Definitions**:
    - `agents.py`: Depends on `imports.py`.
    - `vectorized_agents.py`: Depends on `imports.py`.
- **Model Definitions**:
    - `model.py`: Depends on `imports.py`, `agents.py`.
    - `vectorized_model.py`: Depends on `imports.py`, `vectorized_agents.py`.
- **Simulation Functions**:
    - `simulation_functions.py`: Depends on `imports.py`, `agents.py`, `model.py`, `network_utils.py`, `network_generation.py`.
    - `vectorized_simulation_functions.py`: Depends on `imports.py`, `vectorized_model.py`, `network_utils.py`, `network_generation.py`.
- **Network Handling**:
    - `network_generation.py`: Depends on `imports.py`.
    - `network_utils.py`: Depends on `imports.py`.
- **Testing**:
    - `unit_tests.py`: Depends on `imports.py`, `agents.py`.
    - `test_vectorization.py`: Depends on `model.py`, `vectorized_model.py`.

## Latest Report

**Execution Date:** 2026-01-11

**Test Results:**

*   **Unit Tests (`unit_tests.py`):** PASSED (8 tests)
*   **Vectorization Tests (`test_vectorization.py`):** PASSED (4 tests)
*   **Notebook Verification:**
    *   `basic_model_testing.ipynb`: FAILED (`NameError: name 'df_bayes' is not defined`).
    *   `vectorized_basic_model_testing.ipynb`: PASSED.

**Summary:**
The project dependencies are correctly mapped. Core unit tests and vectorization tests are passing. The basic model testing notebook has a runtime error (undefined variable) that needs addressing, but the vectorized version is functional.
