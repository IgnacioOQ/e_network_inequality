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
