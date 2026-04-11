# AGENTS.md
- status: active

## SHORT ADVICE
- status: active

- The whole trick is providing the AI Assistants with context, and this is done using the *.md files (AGENTS.md, AGENTS_LOG.md, and the AI_AGENTS folder)
- Make sure that when writing *.md files, you use the proper syntax protocol as defined in MD_REPRESENTATION_CONVENTIONS.md. If necessary, you can always use the scripts in the language folder to help you with this.
- Learn how to work the Github, explained below.
- Keep logs of changes in AGENTS_LOG.md
- Make sure to execute the HOUSEKEEPING.md protocol often.
- Always ask several forms of verification, so because the self-loop of the chain of thought improves performance.
- Impose restrictions and constraints explicitly in the context.
- 
## HUMAN-ASSISTANT WORKFLOW
- status: active

1. Open the assistant and load the ai-agents-branch into their local repositories. Do this by commanding them to first of all read the AGENTS.md file.
2. Work on the ASSISTANT, making requests, modifying code, etc.
3. IMPORTANT: GIT MECHANISM
    3.1. Jules (and maybe Claude) push the changes into a newly generated branch. In my case, this is `jules-sync-main-v1-15491954756027628005`. **This is different from the `ai-agents-branch`!!**
    3.2. So what you need to do is merge the newly generated branch and the `ai-agents-branch` often. Usually in the direction from `jules-sync-main-v1-15491954756027628005` to `ai-agents-branch`. I do this by:
        3.2.1. Going to pull requests.
        3.2.2. New Pull request
        3.2.3. Base: `ai-agents-branch`, Compare: `jules-sync-main-v1-15491954756027628005` (arrow in the right direction).
        3.2.4. Follow through. It should allow to merge and there should not be incompatibilities. If there are incompatibilities, you can delete the `ai-agents-branch` and create a new one cloning the `jules-sync-main-v1-15491954756027628005` one. After deleting `ai-agents-branch`, go to the `jules-sync-main-v1-15491954756027628005` branch, look at the dropdown bar with the branches (not the link), and create a new copy.
4. Enjoy!

## WORKFLOW & TOOLING
- status: active

*   **PostToolUse Hook (Code Formatting):**
    *   **Context:** A "hook" is configured to run automatically after specific events.
    *   **The Event:** "PostToolUse" triggers immediately after an agent uses a tool to modify a file (e.g., writing code or applying an edit).
    *   **The Action:** The system automatically runs a code formatter (like `black` for Python) on the modified file.
    *   **Implication for Agents:** You do not need to manually run a formatter. The system handles it. However, be aware that the file content might slightly change (whitespace, indentation) immediately after you write to it.

*   **Jupyter Notebooks (`.ipynb`):**
    *   **Rule:** Do not attempt to read or edit `.ipynb` files directly with text editing tools. They are JSON structures and easy to corrupt.
    *   **Action:** If you need to verify or modify logic in a notebook, ask the user to export it to a Python script, or create a new Python script to reproduce the logic.
    *   **Exception:** You may *run* notebooks if the environment supports it (e.g., via `nbconvert` to execute headless), but avoid editing the source.

*   **Documentation Logs (`AGENTS_LOG.md`):**
    *   **Rule:** Every agent that performs a significant intervention or modifies the codebase **MUST** update the `AGENTS_LOG.md` file.
    *   **Action:** Append a new entry under the "Intervention History" section summarizing the task, the changes made, and the date.

## DEVELOPMENT RULES & CONSTRAINTS
- status: active

1.  **Immutable Core Files:** Do not modify `model/agents.py`, `model/model.py`, or `model/simulation_functions.py`.
    *   If you need to change the logic of an agent or the model, you must create a **new version** (e.g., a subclass or a new file) rather than modifying the existing classes in place.
2.  **Consistency:** Ensure any modifications or new additions remain as consistent as possible with the logic and structure of the `main` branch.
3.  **Coding Conventions:** Always keep the coding conventions pristine.

## CONTEXT FINE-TUNING
- status: active

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

## LOCAL PROJECT DESCRIPTION
- status: active

### Project Overview
- status: active

This project is a simulation framework for agent-based models on various network structures, specifically focusing on network epistemology and theory choice using Bandit problems.

### Setup & Testing
- status: active

*   **Install Dependencies:** `pip install -r requirements.txt` (or manually install `numpy`, `scipy`, `pandas`, `networkx`, `tqdm`, `matplotlib`, `seaborn`, `dill`).
*   **Run Tests:** `python -m unittest unit_tests.py`

### Key Architecture & Logic
- status: active

#### 1. Directed Graphs & Information Flow
- status: active

*   The simulation uses **NetworkX** graphs.
*   **Directionality:** The graph is treated as **Directed** (`nx.DiGraph`).
*   **Interpretation of Edges (`A -> B`):**
    *   In NetworkX, an edge `(u, v)` means `u` points to `v`.
    *   **Information Flow:** In this simulation, an edge from A to B means **A listens to B**.
    *   **Code Implication:** When Agent A updates their belief, they check their **neighbors**.
    *   If using `G.neighbors(A)` in a DiGraph, it returns successors (nodes A points to).
    *   If using `G.predecessors(A)`, it returns nodes pointing to A.
    *   **Convention:** The code typically iterates over `G.predecessors(agent.id)` (or neighbors if undirected) to find the agents that the current agent "observes".
    *   **Summary:** If A observes B, the graph should have an edge `A -> B` (A is the source, B is the target, but information flows B -> A in terms of observation).

#### 2. Agents
- status: active

*   **`Bandit`:** The environment. Returns success/failure based on probabilities.
*   **`BetaAgent`:** Uses Beta distributions to model beliefs about two theories (0 and 1).
    *   `alphas_betas`: Stores `[alpha, beta]` for both theories.
    *   `credences`: Mean of the beta distribution.
    *   `choice`: Epsilon-greedy.

#### 3. Simulation Loop (`Model` class)
- status: active

*   **Step:**
    1.  **Experiment:** Every agent chooses a theory and runs an experiment (getting success/failure).
    2.  **Update:** Every agent observes the results of their **predecessors** (neighbors who point to them).
    3.  **Bayesian Update:** Agents update their Alpha/Beta parameters based on their own *and* their neighbors' results.

### Key Files and Directories
- status: active

#### Directory Structure
- status: active

```
e_network_inequality/
│
├── 1. Citation Data and Networks Generation.ipynb   # Entry point 1: build networks
├── 2. Colab Simulations.ipynb                       # Entry point 2: run simulations
├── 3. Simulations Data Analysis.ipynb               # Entry point 3: analyse results
├── A. Visualizations.ipynb                          # Entry point 4: visualisations
│
├── model/                          # All model and simulation code
│   ├── agents.py                   # Legacy OO agent classes (IMMUTABLE)
│   ├── bandit.py                   # Vectorized multi-armed bandit
│   ├── model.py                    # Legacy OO Model class (IMMUTABLE)
│   ├── vectorized_model.py         # Fast vectorized simulation (primary)
│   ├── simulation_functions.py     # Wrappers for Model (IMMUTABLE)
│   ├── vectorized_simulation_functions.py
│   └── convergence_analysis/       # Convergence & MC analysis (scripts, notebooks, docs)
│
├── networks/                       # Network generation and manipulation
│   ├── network_generation.py       # Synthetic graph generators
│   ├── variation_methods.py        # Network variation utilities
│   └── citation_data/              # Empirical network files (.pkl, .json)
│
├── utils/                          # Shared utilities
│   ├── imports.py                  # Central external library imports
│   ├── network_utils.py            # Network statistics and helpers
│   └── mc_analysis.py              # Markov Chain analysis utilities
│
├── testing/                        # All tests
│   ├── unit_tests/                 # Automated test suite (unittest discover)
│   │   ├── test_agents.py
│   │   ├── test_vectorization.py
│   │   ├── test_mc_analysis.py
│   │   └── test_stopping_conditions.py
│   └── notebooks/                  # Interactive testing notebooks
│       ├── basic_model_testing.ipynb
│       ├── vectorized_basic_model_testing.ipynb
│       ├── reproducing_zollman.ipynb
│       └── variation_methods_test.ipynb
│
├── results/                        # Output datasets and analysis notebooks
├── figures/                        # Visualisation notebooks
│
├── AI_AGENTS/                      # Context files for AI agents
│   ├── AGENTS.md                   # This file
│   ├── AGENTS_LOG.md               # Log of AI interventions
│   ├── HOUSEKEEPING.md             # Housekeeping protocol
│   ├── REPOSITORY_MAP.md           # Detailed module/import map
│   ├── MD_CONVENTIONS.md           # Markdown-JSON schema conventions
│   └── ...
│
├── README.md
├── requirements.txt
└── setup.py
```

#### File Dependencies & Logic
- status: active

The project relies on a central imports file to manage dependencies across modules.
*   **`imports.py`**: Imports all necessary external libraries (`numpy`, `scipy`, `networkx`, `pandas`, etc.) and sets up seeds. It is imported by `agents.py`, `model.py`, and simulation scripts.

**Legacy/Reference Implementation (Immutable):**
*   **`agents.py`**: Defines the object-oriented agent classes:
    *   `Bandit`: The environment returning experiment results.
    *   `BetaAgent`: Bayesian learner using Beta distributions.
    *   `BayesAgent`: Simplified Bayesian learner.
*   **`model.py`**: Defines the `Model` class. It manages the graph (`self.network`), the list of agents, and the time loop (`run_simulation`). It handles the interaction between agents (observing neighbors).
*   **`simulation_functions.py`**: Wrappers to initialize parameters (generating networks) and run the `Model`. Used for parallel execution.

**Vectorized Implementation (Fast):**
*   **`vectorized_model.py`**: The high-performance, matrix-based replacement for `Model`. It stores agent states in NumPy arrays `(N, 2, 2)` instead of objects. Includes convergence tracking and root analysis.
*   **`bandit.py`**: Contains `VectorizedBandit` for batch processing of experiments.
*   **`vectorized_simulation_functions.py`**: Wrappers for running `VectorizedModel`.

**Analysis Tools:**
*   **`utils/mc_analysis.py`**: Markov Chain analysis utilities including:
    *   `MarkovChainAnalyzer`: Main class for trajectory analysis
    *   `StateSnapshot`: Immutable state records with fingerprinting
    *   Convergence diagnostics, mixing time estimation, absorption analysis

**Network Handling:**
*   **`network_generation.py`**: Functions to generate synthetic networks (e.g., `barabasi_albert_directed`, `directed_watts_strogatz`).
*   **`network_utils.py`**: Helper functions for calculating network statistics and metrics.

**Testing & Verification:**
*   **`tests/unit_tests.py`**: Unit tests for the reference implementation (`agents.py`, `model.py`).
*   **`tests/test_vectorization.py`**: Regression tests ensuring `VectorizedModel` matches `Model`.
*   **`tests/test_mc_analysis.py`**: Unit tests for Markov Chain analysis utilities.
*   **`notebooks/basic_testing/`**: Visual verification notebooks for both model implementations.
*   **`model/convergence_analysis/`**: Convergence studies, root influence analysis scripts, and Markov Chain documentation (`MC_AGENT.md`, `ROOTNODE_HYPOTHESIS.md`, `OPEN_QUESTIONS.md`).

