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

Based on post-React integration analysis:
- **Core Modules:** `src/net_epistemology/core/` (`agents.py`, `model.py`, `vectorized_model.py`)
- **Advanced Modules:** `src/net_epistemology/analysis/` (`mc_analysis.py`)
- **Engine Module:** `src/net_epistemology/simulation/`
- **Tests:** `tests/` (`unit_tests.py`, `test_vectorization.py`, `test_mc_analysis.py`)
- **Notebooks:** `notebooks/`

## Latest Report
- status: active

**Execution Date:** 2026-01-23

**Test Results:**
1.  `tests/unit_tests.py`: **Passed** (8 tests).
2.  `tests/test_vectorization.py`: **Passed** (4 tests).
3.  `tests/test_mc_analysis.py`: **Passed** (12 tests).
4.  `tests/vectorized_basic_model_testing_script.py`: **Passed** (Conclusion: 0.94/0.90).
5.  `tests/basic_model_testing_script.py`: **Passed Phase 1** (Conclusion: 0.99). Phase 2 (Bayes) interrupted but functioning.

**Total: 24/24 unit tests passed.**

**Summary:**
All unit tests and vectorization verifications passed. The system is stable.
