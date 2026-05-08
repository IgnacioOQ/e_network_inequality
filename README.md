# Network Epistemology Simulation
- status: active
- owner: user

This project is a simulation framework for agent-based models on various network structures, specifically focusing on network epistemology and theory choice using Bandit problems. It allows for studying how agents update their beliefs in networked environments using Bayesian inference.

## Project Overview
- status: active

### Simulation Core
- status: active

The core of this project is a flexible agent-based modeling framework designed to simulate the spread of beliefs and the evolution of scientific consensus in networked communities.

- **Bayesian Agents**: Agents model their beliefs using Beta distributions (Beta(α, β)), representing their confidence in two competing theories (Theory 0 vs. Theory 1).
- **Network Structure**: The population is structured as a directed graph where edges represent observational channels (e.g., Agent A listens to Agent B).
- **Simulation Loop**:
    1.  **Experiment**: Agents perform "experiments" on their chosen theory (Two-Armed Bandit problem).
    2.  **Observation**: Agents observe the success/failure outcomes of their neighbors (predecessors in the graph).
    3.  **Update**: Agents update their belief parameters (α/β) using Bayesian inference based on their own results and the observed evidence.

Two parallel implementations exist to balance flexibility and performance:
- **Object-Oriented**: Logical, easy to extend (`model/model.py`).
- **Vectorized**: High-performance, matrix-based implementation using NumPy for large-scale simulations (`model/vectorized_model.py`).

### Empirical Networks
- status: active

To validate the models against real-world scientific dynamics, the project incorporates empirical datasets derived from bibliometric data.

- **Source**: Data is obtained from **OpenAlex**, specifically focusing on the history of **Peptic Ulcer Disease (PUD)** research (1900-1978).
- **Network Construction**:
    - **Nodes**: Authors working on the topic.
    - **Edges**: Citations between authors (directed from cited to citing).
- **Processing**: The raw data undergoes rigorous cleaning:
    - Pruning of "twins" (authors who always co-author, treated as a single epistemic unit).
    - Extraction of the **Largest Weakly Connected Component (LCC)** to ensure network integrity.
    - Removal of self-loops.
- **Purpose**: These networks serve as realistic topologies for running the belief dynamics simulations, allowing for the comparison of theoretical predictions with historical consensus shifts.

## Installation
- status: active

### Prerequisites
- status: active

Ensure you have Python 3.8+ installed.

### Setup
- status: active

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Manual install*: `numpy`, `scipy`, `pandas`, `networkx`, `tqdm`, `matplotlib`, `seaborn`, `dill`, `statsmodels`, `joblib`.

## Usage
- status: active

### Running Tests
- status: active

To verify the installation and core logic (run from project root):
```bash
.venv/bin/python -m unittest discover -s testing/unit_tests -v
```

For a deeper sanity check (unit tests + import check + network integrity + notebook validation + snapshot smoke test), follow the [Housekeeping Workflow](HOUSEKEEPING.md). Recommended after any non-trivial change to the model or before committing.

### Running Simulations
- status: active

The main entry-point notebooks are at the **project root**, following this workflow:

1. `1. Citation Data and Networks Generation.ipynb` — Fetch data from OpenAlex and build empirical citation networks.
2. `2. GColab Simulations.ipynb` — Run large-scale simulations on Google Colab (primary simulation entry point).
3. `3. Results Data Analysis.ipynb` — Load simulation outputs, analyse and plot results.

Appendices (standalone, may be incomplete):

- `A. Visualizations.ipynb` — Network and result visualisations.
- `A. GColab Simulations Playground.ipynb` — Experimental simulation runs and sandbox work on Colab.

For testing and debugging refer to `testing/notebooks/`.

## Codebase Architecture and Execution Flow
- status: active

This section is designed to provide a comprehensive structural overview for developers and coding agents to understand how to navigate and modify the codebase.

### 1. Object-Oriented vs Vectorized Paradigms
The simulation logic is bifurcated into two separate implementations to guarantee both theoretical clarity and computational speed:
- **Legacy Object-Oriented Implementation (`model/model.py`, `model/agents.py`)**: This is the original, easy-to-read implementation where each agent is an instantiated Python object. It is strictly **immutable** to serve as a baseline source of truth.
- **Vectorized Implementation (`model/vectorized_model.py`, `model/bandit.py`)**: This is the primary execution engine. It relies heavily on NumPy matrix operations to execute updates across the entire network graph simultaneously. This is the implementation you should focus on when designing new large-scale studies.

### 2. Execution Flow
The typical simulation study utilizes a set of wrapper functions to execute iterations in parallel using Python's `multiprocessing.Pool`:
1. **Network Initialization**: Networks are loaded from `networks/citation_data/` (empirical networks like PUD, Tobacco, Ego) or generated synthetically via `networks/network_generation.py`.
2. **Wrapper Setup**: Functions in `model/vectorized_simulation_functions.py` (e.g., `run_vectorized_simulation_with_params`) act as the bridge between raw parameters and the model execution.
3. **Simulation Loop**: The `VectorizedModel.run_simulation` method loops through steps, allowing agents to choose theories (Epsilon-greedy or Bayes choice), run bandit experiments, and update their beliefs (credences) using network adjacency matrices (`self.adj_matrix.T @ outcomes`).
4. **Data Aggregation**: After the simulation loop completes (via step limit, tolerance threshold, or AUC-ROC threshold), the model concludes and the wrapper function packages the resulting metrics (e.g., truth share, convergence step, trajectory snapshots) into a dictionary, which is then concatenated into pandas DataFrames inside the Jupyter notebooks.

### 3. Notebooks and Where Scripts Live
- **Root Notebooks**: Designed as the high-level orchestrators (`1. Citation Data...`, `2. GColab Simulations...`).
- **`model/convergence_analysis/`**: Contains targeted, detailed Colab notebooks specifically created to study convergence speeds, belief changes, and the impact of specific network conditions.
- **`testing/notebooks/`**: Contains sandbox Jupyter notebooks (`basic_model_testing.ipynb`, `basic_model_testing_v2.ipynb`) useful for rapid prototyping and generating quick visualization plots locally without invoking the full multiprocessing overhead.

## Project Documentation
- status: active

In addition to this README, the project root contains several living documents that track workflow, history, and feature design. New contributors (and AI assistants) should skim these before starting non-trivial work:

- [TODO_WORKFLOW.md](TODO_WORKFLOW.md) — Active task list and goals, organised by priority with owners (Ignacio, Hein, Max). Replaces the previous `WORKPLAN.md`.
- [WORKLOG.md](WORKLOG.md) — Append-only chronological log of significant changes and AI-assisted interventions. Most recent entry first.
- [HOUSEKEEPING.md](HOUSEKEEPING.md) — Routine sanity-check workflow: unit tests, import checks, network integrity, notebook validation, and snapshot smoke test.
- [ADD_SNAPSHOT_PLAN.md](ADD_SNAPSHOT_PLAN.md) — Design document for the snapshot feature in `VectorizedModel` (records truth share and max belief change at fixed intervals during long simulations).

For deeper AI-agent–oriented context (conventions, repository map, intervention log), see the [AI_AGENTS/](AI_AGENTS/) folder.

## Directory Structure
- status: active

```
e_network_inequality/
│
├── 1. Citation Data and Networks Generation.ipynb   # Step 1: fetch data and build networks
├── 2. GColab Simulations.ipynb                      # Step 2: run simulations on Colab
├── 3. Results Data Analysis.ipynb                   # Step 3: analyse and plot results
├── A. Visualizations.ipynb                          # Appendix: network and result visualisations
├── A. GColab Simulations Playground.ipynb           # Appendix: experimental Colab sandbox
│
├── model/                          # All model and simulation code
│   ├── agents.py                   # Legacy OO agent classes (immutable)
│   ├── bandit.py                   # Vectorized multi-armed bandit
│   ├── model.py                    # Legacy OO Model class (immutable)
│   ├── vectorized_model.py         # Fast vectorized simulation (primary)
│   ├── simulation_functions.py     # Wrappers for running Model in parallel
│   ├── vectorized_simulation_functions.py  # Wrappers for VectorizedModel
│   └── convergence_analysis/       # Colab notebooks for convergence studies
│
├── networks/                       # Network generation and manipulation
│   ├── network_generation.py       # Synthetic graph generators (BA, WS, etc.)
│   ├── variation_methods.py        # Network variation utilities (densify, equalize)
│   └── citation_data/              # Pickled empirical network files (.pkl, .json)
│
├── utils/                          # Shared utilities
│   ├── imports.py                  # Central external library imports
│   ├── network_utils.py            # Network statistics and helper functions
│   ├── mc_analysis.py              # Markov Chain analysis utilities
│   └── data_analysis_utils.py      # OLS regression, multicollinearity (VIF/Pearson), Cohen's f², diagnostics
│
├── testing/                        # All tests
│   ├── unit_tests/                 # Automated test suite (run with unittest discover)
│   │   ├── test_agents.py          # Tests for Bandit and BetaAgent
│   │   ├── test_vectorization.py   # Equivalence tests: Model vs VectorizedModel
│   │   ├── test_mc_analysis.py     # Tests for Markov Chain analysis utilities
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
├── AI_AGENTS/                      # Documentation for AI assistants
│   ├── AGENTS.md                   # AI agent instructions and conventions
│   ├── AGENTS_LOG.md               # Log of AI interventions
│   ├── REPOSITORY_MAP.md           # Detailed module dependency map
│   ├── HOUSEKEEPING.md             # Housekeeping protocol
│   ├── MD_CONVENTIONS.md           # Markdown-JSON schema conventions
│   └── ...
│
├── README.md                       # This file
├── TODO_WORKFLOW.md                # Active task list and goals (replaces WORKPLAN.md)
├── WORKLOG.md                      # Append-only chronological change log
├── HOUSEKEEPING.md                 # Routine sanity-check workflow
├── ADD_SNAPSHOT_PLAN.md            # Snapshot feature design document
├── requirements.txt
└── setup.py
```

## Development & Conventions
- status: active

- **Immutable Core Files**: Do not modify `model/agents.py`, `model/model.py`, or `model/simulation_functions.py`. Create new versions (subclasses or new files) instead.
- **Markdown Conventions**: All `.md` files must follow the [Markdown-JSON Hybrid Schema](AI_AGENTS/MD_CONVENTIONS.md).
- **AI Agents**: If you are an AI assistant, primarily rely on `AI_AGENTS/AGENTS.md` and the `AI_AGENTS/` folder for context.
- **Import Convention**: All files use absolute imports from the project root (e.g., `from model.vectorized_model import VectorizedModel`, `from utils.imports import *`). Notebooks add the project root to `sys.path` at startup.
