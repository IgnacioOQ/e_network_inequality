# AI Agents Log & Reference

This file serves as a persistent memory for AI agents working on the project. It includes a reference for common tasks, guidelines, and a chronological log of major interventions.

## Reference

### Common Bash Commands
*   **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually:
    pip install numpy scipy pandas networkx tqdm matplotlib seaborn dill
    ```
*   **Run Unit Tests:**
    ```bash
    python -m unittest unit_tests.py
    ```
*   **Code Formatting:**
    ```bash
    black .
    # Always manually verify after formatting!
    ```

### Code Style & Conventions
*   **Formatting:** The project uses `black` via a PostToolUse hook, but manual verification is required.
*   **Immutability:** `agents.py` and `model.py` are **immutable**. Do not modify them directly. Create subclasses or new files for changes.
*   **Consistency:** Maintain coding style consistent with the `main` branch.

### Architecture & Design
*   **State Management:** The `Model` class (`model.py`) manages the simulation state, specifically the list of `agents`. Agent belief states (`alphas_betas`) are stored within `BetaAgent` instances.
*   **Error Handling:** Use unit tests (`unit_tests.py`) to catch regressions. Ensure `try-except` blocks are used where runtime variability (e.g., network generation) might cause issues.

## Pull Request Template

When submitting changes, please use the following structure:

```markdown
**Title:** [Short, descriptive title]

**Description:**
[Detailed explanation of the changes]

**Changes:**
- [File modified]: [Brief description of change]
- [New file]: [Purpose]

**Verification:**
- [ ] Ran `python -m unittest unit_tests.py` and passed.
- [ ] Manually verified code formatting.
- [ ] Confirmed no changes to immutable files (`agents.py`, `model.py`).
```

## Intervention History

### [DATE] - Initial Setup & Test Fixes (Jules)
*   **Task:** Fix failing unit tests and establish documentation.
*   **Actions:**
    *   Modified `unit_tests.py` to handle random initialization of `BetaAgent` (used shape/type checks instead of hardcoded values).
    *   Renamed `greedy_choice` to `egreedy_choice` in tests.
    *   Created `AGENTS.md` with project context and rules.
    *   Created `AI_AGENTS/` directory for sub-agent context.
    *   Documented development rules (immutability, consistency).

### [2026-01-06] - Sync with Main & Environment Setup (Jules)
*   **Task:** Update branch with code from `main`, add `requirements.txt`, and preserve local documentation/tests.
*   **Actions:**
    *   Synced all tracked files from `main` (commit `583ddb1`) to the current branch.
    *   Restored local `unit_tests.py` to preserve test logic, but updated it to call `agent.update()` instead of `agent.beta_update()` to match the `main` codebase.
    *   Created `requirements.txt` with project dependencies.
    *   Verified simulations: `basic_model_testing.ipynb` (timed out but ran) and `run_simulations_test.ipynb` (failed due to missing `network_randomization.py` in `main`).
    *   Verified unit tests: `python -m unittest unit_tests.py` passed.

### [2026-01-06] - Linearization / Vectorization (Jules)
*   **Task:** Create vectorized implementation of the simulation to improve performance.
*   **Actions:**
    *   Created `vectorized_model.py`: Implements `VectorizedModel` which replaces object-oriented agent state with NumPy matrices `(N_agents, 2, 2)`. Replaces loop-based updates with matrix multiplication (`Adjacency.T @ Outcomes`).
    *   Created `vectorized_agents.py`: Implements `VectorizedBandit` for batch experiment generation.
    *   Created `vectorized_simulation_functions.py`: Wrapper for running vectorized simulations.
    *   Created `test_vectorization.py`: Verified initialization match and update logic equivalence between `Model` and `VectorizedModel`.
    *   **Results:** Benchmarking showed a **~125x speedup** (from 14.5s to 0.11s for 100 agents / 500 steps).

### [2026-01-06] - Documentation Update (Jules)
*   **Task:** Update `AGENTS.md` with explicit Git Management rules.
*   **Actions:**
    *   Added "Git Management" subsection to `AGENTS.md`.
    *   Specified: "All commits should be merged to the 'ai-agents-branch' branch".

### [2026-01-06] - Vectorized Bayes Agent Fixes & Integration (Jules)
*   **Task:** Finalize Vectorized Bayes Agent implementation, fix bugs, and update tests.
*   **Actions:**
    *   **Fixed `vectorized_model.py`:**
        *   Corrected the indentation of the `if self.sampling_update:` block in `step()` method to prevent `ValueError: a <= 0` when running with `agent_type='bayes'`.
        *   Implemented `agent_type='bayes'` logic in `__init__` and `step` (choice, masking experiments, Bayesian update).
        *   Ensured Bayesian update only considers evidence for Theory 1, matching the object-oriented implementation.
        *   **Refined Stopping Logic:** Integrated user-provided refinements to `stop_condition` (respecting `tstep_stopping` and using consensus check for Bayes) and class initialization.
    *   **Updated `test_vectorization.py`:**
        *   Added `test_bayes_initialization_match` to verify random state alignment with `Model`.
        *   Added `test_bayes_update_logic` to verify the mathematical correctness of the vectorized Bayesian update against the scalar `Model`.
    *   **Updated `vectorized_basic_model_testing.ipynb`:**
        *   Added a new section to run and visualize the Bayes Agent simulation.
    *   **Verified:**
        *   Ran unit tests: All passed.
        *   Ran notebook: Successfully executed Bayes simulation without errors.
    *   **Documentation:**
        *   Updated `AI_AGENTS/LINEARIZE_AGENT.md` with Bayes details.
        *   Updated `AGENTS_LOG.md`.

### [2026-01-06] - Workflow & Context Documentation (Jules)
*   **Task:** Add HUMAN-ASSISTANT WORKFLOW and CONTEXT FINE-TUNING to `AGENTS.md`.
*   **Actions:**
    *   Inserted detailed "HUMAN-ASSISTANT WORKFLOW" section at the top of `AGENTS.md`, outlining steps for repo loading, git branching, and PR merging.
    *   Added "CONTEXT FINE-TUNING" section explaining how to teach agents via context files instead of weight updates.
    *   Synced local workspace with `origin/ai-agents-branch` to incorporate user's manual fixes to `vectorized_model.py`.

### [2026-01-06] - Documentation Reorganization (Jules)
*   **Task:** Reorganize `AGENTS.md` sections and add specific advice/rules.
*   **Actions:**
    *   Added `SHORT ADVICE` section at the top.
    *   Reordered sections: Short Advice -> Human-Assistant Workflow -> Workflow & Tooling -> Development Rules -> Context Fine-Tuning.
    *   Updated `WORKFLOW & TOOLING` to remove Git Management (now in Human-Assistant Workflow).
    *   Updated `DEVELOPMENT RULES & CONSTRAINTS` with coding convention rule.
    *   Added `LOCAL PROJECT DESCRIPTION` header.

### [2026-01-06] - Expand Key Files Description (Jules)
*   **Task:** Expand the "Key Files and Directories" section in `AGENTS.md` with detailed structure and dependency info.
*   **Actions:**
    *   Renamed "Key Files" to "Key Files and Directories".
    *   Added "Directory Structure" subsection.
    *   Added "File Dependencies & Logic" subsection mapping out imports.
    *   Expanded descriptions for both Legacy and Vectorized implementations.

### [2026-01-11] - Housekeeping and Dependency Mapping (Jules)
*   **Task:** Execute Housekeeping protocol, map dependencies, and update reports.
*   **Actions:**
    *   Mapped dependency network using `grep` on import statements.
    *   Executed `unit_tests.py` (Passed).
    *   Executed `test_vectorization.py` (Passed).
    *   Executed `vectorized_basic_model_testing.ipynb` via conversion script (Passed).
    *   Executed `basic_model_testing.ipynb` via conversion script (Failed: `NameError: name 'df_bayes' is not defined`).
    *   Updated `HOUSEKEEPING.md` with the corrected dependency network and test report.
    *   **Errors Logged:** `basic_model_testing.ipynb` failed with `NameError: name 'df_bayes' is not defined` at line `mean_credence = df_bayes.mean(axis=1)`.

### [2026-01-12] - Housekeeping and Verification (Antigravity)
*   **Task:** Executed comprehensive housekeeping protocol, including dependency mapping and verification testing.
*   **Actions:**
    *   **Dependency Analysis:** Verified and mapped current file structure (`src/net_epistemology`).
    *   **Unit Testing:** `tests/unit_tests.py` confirmed 8/8 tests passed.
    *   **Vectorization Verification:** `tests/test_vectorization.py` confirmed 4/4 tests passed.
    *   **Notebook Verification:** Executed conversion scripts.
        *   `basic_model_testing_script.py`: Failed (`NameError`).
        *   `vectorized_basic_model_testing_script.py`: Failed (`AttributeError` in pandas usage).
    *   **Environment:** Installed missing `dill` dependency.
    *   **Documentation:** Updated `HOUSEKEEPING.md` with new dependency graph and latest test report.

### [2026-01-12] - Notebook Fixes and Verification (Jules)
*   **Task:** Fix identified errors in notebooks and `agents.py` to pass housekeeping checks.
*   **Actions:**
    *   **Branch Sync:** Merged `ai-agents-branch` to ensure latest state.
    *   **`basic_model_testing.ipynb` Fix:**
        *   Identified `AttributeError: 'Model' object has no attribute 'credences_history'` and `BayesAgent` lacking history tracking.
        *   Modified `src/net_epistemology/core/agents.py`: Updated `BayesAgent` to accept `histories` parameter and store `credences_history` (overriding immutability rule with user permission).
        *   Modified `src/net_epistemology/core/model.py`: Updated `Model.__init__` to pass `histories` to `BayesAgent`.
        *   Patched `notebooks/basic_model_testing.ipynb`: Set `histories=True` for Bayes agent, updated attribute access to `agent_histories`, uncommented plotting code, and updated deprecated `applymap` to `map`.
        *   Verified execution via extracted script.
    *   **`vectorized_basic_model_testing.ipynb` Fix:**
        *   Identified `KeyError: 'edges'` due to JSON format mismatch (`links` vs `edges`).
        *   Patched `notebooks/vectorized_basic_model_testing.ipynb`: Added logic to rename `links` key to `edges` before loading into NetworkX.
        *   Verified execution via extracted script.
    *   **Housekeeping:** Updated `HOUSEKEEPING.md` with fully passing report.
    *   **Clean Up:** Removed temporary scripts.
