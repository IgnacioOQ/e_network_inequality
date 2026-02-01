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

- **`AI_AGENTS/`**: Context files for specialized AI agents.
    - `AGENTS.md`: Main context file for AI agents.
    - `MD_CONVENTIONS.md`: Markdown conventions for the project.
    - `LINEARIZE_AGENT.md`, `MC_AGENT.md`, etc.
- **`src/net_epistemology/`**: Source code.
    - `core/`: Core classes (`Model`, `BetaAgent`, `vectorized_model`).
    - `simulation/`: Helper functions to run simulations.
    - `utils/`: Network generation (`network_generation.py`) and import management.
    - `analysis/`: Tools for analyzing simulation results.
- **`notebooks/`**: Jupyter notebooks for testing, variations, and analysis.
- **`tests/`**: Unit tests.

## Development & Conventions
- status: active

- **Markdown Conventions**: All `.md` files must follow the [Markdown-METADATA Hybrid Schema](AI_AGENTS/MD_CONVENTIONS.md).
    - Headers must be immediately followed by a metadata block (bulleted list).
    - There must be a blank line between metadata and content.
- **AI Agents**: If you are an AI assistant, primarily rely on `AI_AGENTS/AGENTS.md` and the `AI_AGENTS/` folder for context.
