# Network Epistemology Simulation
- status: active
- owner: user

This project is a simulation framework for agent-based models on various network structures, specifically focusing on network epistemology and theory choice using Bandit problems. It allows for studying how agents update their beliefs in networked environments using Bayesian inference.

## Project Overview
- status: active

The core of this project simulates a population of agents connected via a directed graph. Agents are faced with a "Two-Armed Bandit" problem (Theory 0 or Theory 1). They perform experiments, observe the results of their neighbors (predecessors), and update their beliefs (Alpha/Beta parameters) accordingly.

Key features include:
- **Directed Graphs**: Modeled using `networkx`. Edges represent information flow (e.g., A -> B means A listens to B).
- **Bayesian Agents**: Agents use Beta distributions to model their credence in the theories.
- **Dual Implementations**:
    - **Object-Oriented**: Easy to understand, flexible agents (`src/net_epistemology/core/agents.py`).
    - **Vectorized**: High-performance, matrix-based implementation for large-scale simulations (`src/net_epistemology/core/vectorized_model.py`).
- **Analysis Tools**: Built-in tools for Markov Chain analysis and convergence studies.

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
    *Manual install*: `numpy`, `scipy`, `pandas`, `networkx`, `tqdm`, `matplotlib`, `seaborn`, `dill`.

## Usage
- status: active

### Running Tests
- status: active

To verify the installation and core logic:
```bash
python -m unittest unit_tests.py
```

### Running Simulations
- status: active

Refer to the `notebooks/` directory for examples of how to set up and run simulations.
- `notebooks/basic_testing/basic_model_testing.ipynb`: Good starting point for understanding the object-oriented model.
- `notebooks/basic_testing/vectorized_basic_model_testing.ipynb`: Guide for the vectorized model.

## Directory Structure
- status: active

- **`src/net_epistemology/`**: The core package containing the simulation logic.
    - `core/`: Contains the primary classes for the model (`Model`, `VectorizedModel`) and agents (`BetaAgent`, `BayesAgent`). This is where the Bayesian update logic and network interactions are defined.
    - `simulation/`: Helper functions and wrappers to run large-scale simulations, including parallel execution tools.
    - `data/`: Modules for handling data loading and processing.
    - `utils/`: Utilities for network generation (e.g., scale-free, small-world graphs) and dependency management.
    - `analysis/`: Tools for analyzing simulation results, including Markov Chain analysis and convergence diagnostics.
- **`data/`**: Storage for input and output data.
    - `empirical_networks/`: JSON files representing real-world networks used for empirical validation of the models.
    - `results_data_sets/`: Generated datasets from simulations, often used for plotting and analysis in notebooks.
- **`notebooks/`**: Jupyter notebooks for interactive testing, visualization, and deep-dive analysis (e.g., convergence studies, root influence).
- **`tests/`**: Unit tests ensuring the stability and correctness of the core logic, vectorization, and analysis tools.
- **`AI_AGENTS/`**: Documentation and context files specifically designed for AI assistants (like this one) to understand the project architecture, rules, and specialized tasks.
    - `AGENTS.md`, `MD_CONVENTIONS.md`, etc.

## Development & Conventions
- status: active

- **Markdown Conventions**: All `.md` files must follow the [Markdown-METADATA Hybrid Schema](AI_AGENTS/MD_CONVENTIONS.md).
    - Headers must be immediately followed by a metadata block (bulleted list).
    - There must be a blank line between metadata and content.
- **AI Agents**: If you are an AI assistant, primarily rely on `AI_AGENTS/AGENTS.md` and the `AI_AGENTS/` folder for context.
