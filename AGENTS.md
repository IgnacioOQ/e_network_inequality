# AGENTS.md

## HUMAN-ASSISTANT WORKFLOW
*   **Open the assistant and load the ai-agents-branch into their local repositories.** Do this by commanding them to first of all read the AGENTS.md file.
*   **Work on the ASSISTANT, making requests, modifying code, etc.**

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
