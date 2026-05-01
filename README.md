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
├── README.md
├── requirements.txt
└── setup.py
```

## Development & Conventions
- status: active

- **Immutable Core Files**: Do not modify `model/agents.py`, `model/model.py`, or `model/simulation_functions.py`. Create new versions (subclasses or new files) instead.
- **Markdown Conventions**: All `.md` files must follow the [Markdown-JSON Hybrid Schema](AI_AGENTS/MD_CONVENTIONS.md).
- **AI Agents**: If you are an AI assistant, primarily rely on `AI_AGENTS/AGENTS.md` and the `AI_AGENTS/` folder for context.
- **Import Convention**: All files use absolute imports from the project root (e.g., `from model.vectorized_model import VectorizedModel`, `from utils.imports import *`). Notebooks add the project root to `sys.path` at startup.
