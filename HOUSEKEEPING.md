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
- **Core Modules:**
- **Advanced Modules:**
- **Engine Module:**
- **API Module:**
- **Tests:**
- **Notebooks:**

## Latest Report
- status: active

**Execution Date:** 2026-01-19

**Test Results:**
1. `tests/test_api.py`: **Passed** (17 tests).
2. `tests/test_engine.py`: **Passed** (16 tests).
3. `tests/test_mechanics.py`: **Passed** (4 tests).
4. `tests/test_proxy_simulation.py`: **Passed** (1 test).

**Total: 38 tests passed.**

**Summary:**
All unit tests passed. `verify_logging.py` confirmed correct simulation flow and logging. Data persistence features have been integrated and verified locally. Project is stable.
