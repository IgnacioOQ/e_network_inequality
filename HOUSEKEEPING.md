# Housekeeping Protocol
- status: active

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. Make sure that the report uses the proper syntax protocol as defined in MD_REPRESENTATION_CONVENTIONS.md. If necessary, you can always use the scripts in the language folder to help you with this.
6. Print that report in the Latest Report subsection below, overwriting previous reports.
7. Add that report to the AGENTS_LOG.md.

# Current Project Housekeeping
- status: active

## Dependency Network
- status: active

Based on analysis of source code imports:
- **Core Modules:**
  - `src/net_epistemology/core/agents.py` (Base agent classes)
  - `src/net_epistemology/core/model.py` (Legacy model, depends on `agents.py`)
  - `src/net_epistemology/core/vectorized_model.py` (Fast model, independent of `agents.py` and `model.py`)
  - `src/net_epistemology/core/vectorized_agents.py` (Used by `vectorized_model.py`)
- **Simulation Engine:**
  - `src/net_epistemology/simulation/simulation_functions.py` (Wraps `Model`)
  - `src/net_epistemology/simulation/vectorized_simulation_functions.py` (Wraps `VectorizedModel`)
- **Analysis:**
  - `src/net_epistemology/analysis/mc_analysis.py` (Markov Chain analysis, uses `VectorizedModel` for checks)
- **Utilities:**
  - `src/net_epistemology/utils/imports.py` (Centralized imports)
  - `src/net_epistemology/utils/network_generation.py`
  - `src/net_epistemology/utils/network_utils.py`

## Latest Report
- status: active

**Execution Date:** 2026-01-26

**Test Results:**
1.  `tests/unit_tests.py`: **Passed** (8 tests).
2.  `tests/test_vectorization.py`: **Passed** (4 tests).
3.  `tests/test_mc_analysis.py`: **Passed** (12 tests).
4.  `tests/vectorized_basic_model_testing_script.py`: **Passed** (Beta Conclusion: 0.96, Bayes Conclusion: 0.72).
5.  `tests/basic_model_testing_script.py`: **Passed** (Beta Conclusion: 0.87, Bayes Conclusion: 0.85).

**Total: 24/24 unit tests passed.**

**Summary:**
All unit tests and smoke tests passed. The system is stable. The dependency graph remains consistent with the previous architecture.
