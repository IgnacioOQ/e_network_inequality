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
- **Analysis Tools**:
    - `src/net_epistemology/analysis/mc_analysis.py`: Depends on `core.vectorized_model`. Provides Markov Chain analysis utilities.
- **Testing**:
    - `tests/unit_tests.py`: Depends on `imports`, `core.agents`.
    - `tests/test_vectorization.py`: Depends on `core.model`, `core.vectorized_model`.
    - `tests/test_mc_analysis.py`: Depends on `core.vectorized_model`, `analysis.mc_analysis`.

## Latest Report

**Execution Date:** 2026-01-21 (13:45)

**Test Results:**

*   **Unit Tests (`tests/unit_tests.py`):** PASSED (8 tests)
*   **Vectorization Tests (`tests/test_vectorization.py`):** PASSED (4 tests)
*   **Script Verification:**
    *   `tests/basic_model_testing_script.py`: PASSED (conclusion: 0.86/0.95)
    *   `tests/vectorized_basic_model_testing_script.py`: PASSED (conclusion: 0.91)
    *   `notebooks/convergence_analysis/convergence_studies.py`: PASSED (conclusion: 0.5929)
    *   `notebooks/convergence_analysis/root_influence_analysis.py`: PASSED (Gap +0.0096 at 1M steps)

**Summary:**
Notebooks organized into subfolders. Testing scripts patched to use Agg backend and fixed attribute access (`credences_history` -> `agent_histories`). All tests passed.
