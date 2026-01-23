# Housekeeping Protocol
- status: active

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md.

# Current Project Housekeeping
- status: active

## Dependency Network
- status: active

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
- status: active

**Execution Date:** 2026-01-23

**Test Results:**

*   **Unit Tests (`tests/unit_tests.py`):** PASSED (8/8 tests)
    - All Bandit and BetaAgent tests passed successfully
    - Test coverage: initialization, experiment execution, beta updates, greedy choice
    - Execution time: ~0.003s

*   **Vectorization Tests (`tests/test_vectorization.py`):** PASSED (4/4 tests)
    - Legacy vs Vectorized implementation equivalence verified
    - Beta and Bayes agent initialization and update logic match
    - Execution time: ~0.011s

*   **Markov Chain Analysis Tests (`tests/test_mc_analysis.py`):** PASSED (12/12 tests)
    - StateSnapshot fingerprint generation and consistency verified
    - MarkovChainAnalyzer core functionality validated
    - Convergence diagnostics, trajectory summaries, and Markov property checks working
    - Support for both Beta and Bayes agents confirmed
    - Execution time: ~0.162s

*   **Basic Model Verification:** PASSED
    - `tests/basic_model_testing_script.py` passed (Fixed deprecated `applymap` usage).
    - `tests/vectorized_basic_model_testing_script.py` passed.

**Summary:**
Complete test suite passes with 100% success rate. All core functionality verified:
- Legacy agent implementation working correctly.
- Vectorized implementation maintains equivalence with legacy code.
- Markov Chain analysis module fully functional.
- Basic model verification scripts running successfully.
- Dependency graph verified and matches documentation.

**System Status:** All systems operational. Fixed `applymap` deprecation in `tests/basic_model_testing_script.py`.
