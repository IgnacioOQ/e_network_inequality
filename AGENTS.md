# AGENTS.md

This file provides context for AI agents (and humans) working on this codebase.

## Project Overview
This project is a simulation framework for agent-based models on various network structures, specifically focusing on network epistemology and theory choice using Bandit problems.

## Setup & Testing
*   **Install Dependencies:** `pip install -r requirements.txt` (or manually install `numpy`, `scipy`, `pandas`, `networkx`, `tqdm`, `matplotlib`, `seaborn`, `dill`).
*   **Run Tests:** `python -m unittest unit_tests.py`

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
*   `unit_tests.py`: Contains unit tests for the agents. Note that tests must account for random initialization.
*   `network_utils.py`: Helper functions for network manipulation.
