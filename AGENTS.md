# AGENTS.md

## HUMAN-ASSISTANT WORKFLOW
1.  **Open the assistant and load the ai-agents-branch into their local repositories.** Do this by commanding them to first of all read the AGENTS.md file.
2.  **Work on the ASSISTANT, making requests, modifying code, etc.**
3.  **IMPORTANT: GIT MECHANISM**
    3.1. Jules (and maybe Claude) push the changes into a newly generated branch. In my case, this is `jules-sync-main-v1-15491954756027628005`. **This is different from the `ai-agents-branch`!!**
    3.2. So what you need to do is merge the newly generated branch and the `ai-agents-branch` often. Usually in the direction from `jules-sync-main-v1-15491954756027628005` to `ai-agents-branch`. I do this by:
        3.2.1. Going to pull requests.
        3.2.2. New Pull request
        3.2.3. Base: `ai-agents-branch`, Compare: `jules-sync-main-v1-15491954756027628005` (arrow in the right direction).
        3.2.4. Follow through. It should allow to merge and there should not be incompatibilities. If there are incompatibilities, you can delete the `ai-agents-branch` and create a new one cloning the `jules-sync-main-v1-15491954756027628005` one. After deleting `ai-agents-branch`, go to the `jules-sync-main-v1-15491954756027628005` branch, look at the dropdown bar with the branches (not the link), and create a new copy.
4.  **Enjoy!**

## CONTEXT FINE-TUNING
You cannot "fine-tune" an AI agent (change its underlying neural network weights) with files in this repository. **However**, you **CAN** achieve a similar result using **Context**.

**How it works (The "Context" Approach):**
If you add textbooks or guides to the repository (preferably as Markdown `.md` or text files), agents can read them. You should then update the relevant agent instructions (e.g., `AI_AGENTS/LINEARIZE_AGENT.md`) to include a directive like:

> "Before implementing changes, read `docs/linearization_textbook.md` and `docs/jax_guide.md`. Use the specific techniques described in Chapter 4 for sparse matrix operations."

**Why this is effective:**
1.  **Specific Knowledge:** Adding a specific textbook helps if you want a *specific style* of implementation (e.g., using `jax.lax.scan` vs `vmap` in a particular way).
2.  **Domain Techniques:** If the textbook contains specific math shortcuts for your network types, providing the text allows the agent to apply those exact formulas instead of generic ones.

**Recommendation:**
If you want to teach an agent a new language (like JAX) or technique:
1.  Add the relevant chapters as **text/markdown** files.
2.  Update the agent's instruction file (e.g., `AI_AGENTS/LINEARIZE_AGENT.md`) to reference them.
3.  Ask the agent to "Refactor the code using the techniques in [File X]".

This file provides context for AI agents (and humans) working on this codebase.

## Project Overview
This project is a simulation framework for agent-based models on various network structures, specifically focusing on network epistemology and theory choice using Bandit problems.

## Setup & Testing
*   **Install Dependencies:** `pip install -r requirements.txt` (or manually install `numpy`, `scipy`, `pandas`, `networkx`, `tqdm`, `matplotlib`, `seaborn`, `dill`).
*   **Run Tests:** `python -m unittest unit_tests.py`

## Development Rules & Constraints
1.  **Immutable Core Files:** Do not modify `agents.py`, `model.py`, or `simulation_functions.py`.
    *   If you need to change the logic of an agent or the model, you must create a **new version** (e.g., a subclass or a new file) rather than modifying the existing classes in place.
2.  **Consistency:** Ensure any modifications or new additions remain as consistent as possible with the logic and structure of the `main` branch.

## Workflow & Tooling
*   **PostToolUse Hook (Code Formatting):**
    *   **Context:** A "hook" is configured to run automatically after specific events.
    *   **The Event:** "PostToolUse" triggers immediately after an agent uses a tool to modify a file (e.g., writing code or applying an edit).
    *   **The Action:** The system automatically runs a code formatter ("beautifier") on the modified file.
    *   **Implication:** The hook handles the heavy lifting of styling.

*   **Verification (Manual Check):**
    *   **Rule:** Despite the automatic hook, agents must **manually double-check** that the code is correctly formatted and syntactically valid before submitting.
    *   **Action:** Agents should explicitly run a formatter (e.g., `black .` or `black <file>`) or a linter to verify compliance. Do not blindly rely on the hook; ensure the final output is clean.

*   **Visual Verification (Notebook):**
    *   **Rule:** Every agent that performs a significant intervention or modifies the codebase **MUST** run the following notebooks:
        1.  `basic_model_testing.ipynb`
        2.  `run_simulations_test.ipynb`
    *   **Rationale:** Visual inspection of the output (plots, dataframes) is required to verify correct simulation behavior.
    *   **Action:** Execute the notebooks and inspect the results. Do not rely solely on headless unit tests.

*   **Logging Changes:**
    *   **Rule:** Every agent that performs a significant intervention or modifies the codebase **MUST** update the `AGENTS_LOG.md` file.
    *   **Action:** Append a new entry under the "Intervention History" section summarizing the task, the changes made, and the date.

*   **Git Management:**
    *   **Rule:** All commits should be merged to the 'ai-agents-branch' branch.
    *   **Constraint:** Do **NOT** create new branches unless explicitly requested by the user. Always work on and submit to `ai-agents-branch`.

## Key Architecture & Logic

### 1. Directed Graphs & Information Flow
*   **Directionality:** In the directed networks used here, a directed edge `A -> B` signifies information flow from A to B.
*   **Observation:** Consequently, B observes A (receives information from A).
*   **Implementation:** In `model.py`, the `agents_update` method uses `self.network.predecessors(agent.id)` to identify the neighbors that the agent observes. This aligns with the logic that if `A -> B`, A is a predecessor of B, and B gets information from A.

### 2. Agents
*   **BetaAgent:** Defined in `agents.py`.
*   **Initialization:** `alphas_betas` are initialized with **random values** from a uniform distribution (`rd.uniform(0, 4)`), not fixed constants.
*   **Choice Mechanism:** The agent uses `egreedy_choice` (Epsilon-Greedy) to select theories.

## Key Files
*   `agents.py`: Defines `Bandit`, `BetaAgent`, and `BayesAgent`.
*   `model.py`: Defines the `Model` class which manages the simulation loop, agent interactions, and graph updates.
*   `simulation_functions.py`: Defines the wrapper functions for running parallel simulations.
*   `unit_tests.py`: Contains unit tests for the agents. Note that tests must account for random initialization.
*   `network_utils.py`: Helper functions for network manipulation.
