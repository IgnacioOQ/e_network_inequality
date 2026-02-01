# Network Epistemology Simulation
- status: active
- owner: user

This project is a simulation framework for agent-based models on various network structures, specifically focusing on network epistemology and theory choice using Bandit problems. It allows for studying how agents update their beliefs in networked environments using Bayesian inference.

## Project Overview
- status: active

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
- **Object-Oriented**: Logical, easy to extend (`src/net_epistemology/core/agents.py`).
- **Vectorized**: High-performance, matrix-based implementation using NumPy for large-scale simulations (`src/net_epistemology/core/vectorized_model.py`).

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

- **Markdown Conventions**: All `.md` files must follow the [Markdown-JSON Hybrid Schema](AI_AGENTS/MD_CONVENTIONS.md).
    - Headers must be immediately followed by a metadata block (bulleted list).
    - Metadata blocks must be separated from content by a `<!-- content -->` line.
    - This schema allows for bidirectional conversion between Markdown and JSON for programmatic task management.
- **AI Agents**: If you are an AI assistant, primarily rely on `AI_AGENTS/AGENTS.md` and the `AI_AGENTS/` folder for context.
