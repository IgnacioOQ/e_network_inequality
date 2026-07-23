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

- **Source**: Data is obtained from **OpenAlex** (via `pyalex`), covering three scientific episodes in which a consensus shifted. All three are built by `1. Citation Data and Networks Generation.ipynb`:

| Episode | OpenAlex query | Years | Network file | Size |
|:---|:---|:---|:---|:---|
| Peptic ulcer disease | `"peptic ulcer disease"` | 1900–1978 | `pud_network.pkl` | 90 nodes, 160 edges |
| Tobacco and health | `(tobacco OR smoking OR cigarette) AND (health OR cancer OR lung)` | 1900–1964 | `tobacco_network.pkl` | 289 nodes, 1229 edges |
| Ego depletion | `"ego depletion"` | 1900–2016 | `ego_network.pkl` | 503 nodes, 2933 edges |

  A fourth episode (**perceptron**) exists in the notebook's `Archive` section and as `perceptron_final.pkl`; it is not part of the current studies.

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

**Python 3.9 or newer.** The codebase uses PEP 585 builtin generics in annotations
(e.g. `-> tuple[int, int]` in `model/agents.py`) without `from __future__ import annotations`,
so it raises `TypeError` on 3.8. The project virtualenv (`.venv/`) runs **Python 3.10.7**, which is
what the test suite is exercised against.

### Setup
- status: active

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    This covers the simulation and analysis stack: `numpy`, `scipy`, `pandas`, `networkx`, `tqdm`,
    `matplotlib`, `seaborn`, `dill`, `statsmodels`, `joblib`, `scikit-learn`.

3.  **Extra dependencies for data collection.** `1. Citation Data and Networks Generation.ipynb`
    additionally needs `pyalex` (OpenAlex client) and `python-dotenv`, which are **not** currently
    in `requirements.txt` and are not installed in `.venv/`:
    ```bash
    pip install pyalex python-dotenv
    ```
    It also requires an OpenAlex API key. Copy `.env.example` to `.env` and fill in
    `OPEN_ALEX_API_KEY`. The other notebooks read the pre-built networks from
    `networks/citation_data/` and need none of this.

    Some analysis and visualisation notebooks pull in further optional packages not listed in
    `requirements.txt` — `pyvis` and `graphviz` (`3. Results Data Analysis.ipynb`), `powerlaw`,
    `graphistry` and `colormaps` (`4. Network-Visualizations.ipynb`). Install them on demand.

> **Note:** `setup.py`'s `install_requires` is narrower than `requirements.txt` — it omits
> `statsmodels`, `joblib`, and `scikit-learn`. Prefer `requirements.txt` for a working environment.

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

The entry-point notebooks are at the **project root**. The leading number is the workflow stage;
two notebooks sharing a number are alternatives at that stage, not sequential steps.

| Stage | Notebook | Purpose |
|:--|:---|:---|
| 1 | `1. Citation Data and Networks Generation.ipynb` | Fetch data from OpenAlex and build the three empirical citation networks. Needs an API key (see Setup). |
| 2 | `2. GColab Simulations.ipynb` | Run large-scale simulations on Google Colab — the primary simulation entry point. |
| 2 | `2. GColab Simulations Equality.ipynb` | Colab variant for the equality and clustering simulations. |
| 3 | `3. Results Data Analysis.ipynb` | Load simulation outputs, analyse and plot results. |
| 3 | `3. Local Simulations SA.ipynb` | Local sensitivity-analysis runs, as an alternative to the Colab path. |
| 4 | `4. Network-Visualizations.ipynb` | Network statistics and visualisations. |
| A | `A. GColab Simulations Playground.ipynb` | Appendix: experimental runs and sandbox work on Colab. Standalone, may be incomplete. |

For testing and debugging refer to `testing/notebooks/`. For the targeted convergence and
stopping-condition studies, see `model/convergence_analysis/` (described below).

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
4. **Data Aggregation**: After the simulation loop completes (see stopping conditions below), the model concludes and the wrapper function packages the resulting metrics (e.g., truth share, convergence step, trajectory snapshots) into a dictionary, which is then concatenated into pandas DataFrames inside the Jupyter notebooks.

#### Stopping conditions
- status: active

`VectorizedModel` supports four ways to end a run. The first three are **mutually exclusive** modes
set on the constructor; AUC stopping is a separate flag on `run_simulation`:

| Mode | Constructor flag | Stops when |
|:---|:---|:---|
| Tolerance (default) | `tolerance_stopping=True` | Max absolute credence change in a step falls below `tolerance` |
| Fixed steps | `tstep_stopping=True` | `number_of_steps` is reached; no early exit |
| Choice stability | `choice_stability_stopping=True` | *Every* agent's chosen theory has been unchanged for `choice_stability_window` consecutive steps |
| AUC-ROC | `run_simulation(auc_stopping=True)` | Node-level AUC-ROC reaches `auc_threshold` (default 0.95) |

**Choice-stability stopping** is a decision-stability criterion added to address the
false-convergence failure mode of tolerance stopping (a quiet step can fire `allclose` while the
network is still drifting). It is governed by `choice_stability_window` (default 500),
`choice_stability_min_steps` (a floor guarding against stopping on transient early stability), and
`record_choice_flips`, which logs `(step, truth_share)` to `choice_flip_history` so an entire window
sweep can be derived offline from a single *record-once* run. Design and results:
[STOPPING_CONDITION_ANALYSIS.md](model/convergence_analysis/stopping_condition/STOPPING_CONDITION_ANALYSIS.md).

#### Random seeds in parallel studies
- status: active

Each parallel run gets a different random seed so the study reflects genuine stochastic variance — without this, `multiprocessing.Pool` workers fork-inherit the parent's RNG state and silently produce identical trajectories. `run_vectorized_simulation_with_params` handles this in two modes:

- **Default (no seed in `param_dict`)**: the wrapper draws a fresh seed from OS entropy at the start of every job. Different seed per run, but **not reproducible** — you cannot re-run a specific outlier.
- **Reproducible (recommended for published studies)**: derive child seeds from a master seed via `numpy.random.SeedSequence.spawn(N)` and attach them to each `param_dict` before the `Pool.imap_unordered` call:

  ```python
  from numpy.random import SeedSequence
  ss = SeedSequence(MASTER_SEED)
  child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_simulations)]
  for pd_, cs in zip(param_dicts, child_seeds):
      pd_["seed"] = cs
  ```

  `SeedSequence` guarantees statistically independent streams. Using `seed = i` or `seed = master + i` does not.

In both modes the seed actually used is written into `result_dict["seed"]`, so any single run can be replayed even when the study itself wasn't pre-seeded.

### 3. Notebooks and Where Scripts Live
- **Root Notebooks**: The high-level orchestrators (`1. Citation Data...` → `4. Network-Visualizations...`), listed under [Running Simulations](#running-simulations) above.
- **`model/convergence_analysis/`**: Targeted studies of convergence behaviour, organised into four themed subdirectories. Each pairs Colab notebooks with standalone `.py` drivers and a markdown analysis document:
    - `stopping_condition/` — the largest: seven notebooks, six `.py` drivers (`choice_stability_stopping.py`, `convergence_speed_analysis.py`, `parameter_search.py`, `post_stopping_drift.py`, `stopping_tolerance_sensitivity.py`, `tolerance_vs_alphabeta.py`), plus `STOPPING_CONDITION_ANALYSIS.md` and `CHOICE_STABILITY_STOPPING_PLAN.md`.

      The seven notebooks are four *source* studies plus three *executed snapshots*, distinguished by naming convention: spaced names (`A. Choice Stability Stopping Study.ipynb`) are the editable sources; underscored names (`A_Choice_Stability_Stopping_Study.ipynb`) are committed snapshots of a completed Colab run, carrying full outputs and running to several MB. Three of the four sources have a snapshot; `A. Stopping Condition Study v2.ipynb` does not. Edit the spaced files; treat the underscored ones as read-only results.
    - `phase_dynamics/` — `convergence_studies.py`, `two_phase_dynamics.py`, and a Colab study notebook.
    - `root_node_influence/` — `root_influence_analysis.py`, a Colab notebook, and `ROOTNODE_HYPOTHESIS.md`.
    - `formal_markov/` — Markov-chain formalisation: a Colab notebook plus `HYPOTHESIS.md` and `STOCHASTIC_HYPOTHESIS.md`.

    `00_Colab_Template.ipynb` is the starting template for a new study; `MC_AGENT.md` and `OPEN_QUESTIONS.md` sit at the top of the folder.
- **`testing/notebooks/`**: Sandbox Jupyter notebooks (`basic_model_testing.ipynb`, `basic_model_testing_v2.ipynb`, `vectorized_basic_model_testing.ipynb`, `reproducing_zollman.ipynb`, `variation_methods_test.ipynb`) useful for rapid prototyping and generating quick visualization plots locally without invoking the full multiprocessing overhead.
- **`results/`**: Analysis notebooks (`simulation_analysis.ipynb`, `playground_Hein.ipynb`) and the `zollman_2007.csv` reference dataset. Bulk simulation outputs are not committed — large runs write to Google Drive.
- **`scripts/archive/`**: Historical one-shot scripts, retained for provenance only. Not runnable against the current tree.

## Project Documentation
- status: active

In addition to this README, the project root contains several living documents that track workflow, history, and feature design. New contributors (and AI assistants) should skim these before starting non-trivial work:

- [TODO_WORKFLOW.md](TODO_WORKFLOW.md) — Active task list and goals, organised by priority with owners (Ignacio, Hein, Max). Replaces the previous `WORKPLAN.md`.
- [WORKLOG.md](WORKLOG.md) — Append-only chronological log of significant changes and AI-assisted interventions. Most recent entry first.
- [HOUSEKEEPING.md](HOUSEKEEPING.md) — Routine sanity-check workflow: unit tests, import checks, network integrity, notebook validation, and snapshot smoke test.
- [docs/COLAB_MCP_WORKFLOW.md](docs/COLAB_MCP_WORKFLOW.md) — Protocol for driving this project's notebooks on Google Colab from an MCP client, and snapshotting executed state back into git.

The `AI_AGENTS/` folder holds AI-agent–oriented context and historical design documents:

- [AI_AGENTS/REPOSITORY_MAP.md](AI_AGENTS/REPOSITORY_MAP.md) — Module dependency map. **Note:** last verified 2026-03-15 and now partly stale — its directory listing still uses pre-rename notebook names. Treat this README's Directory Structure as authoritative.
- [AI_AGENTS/MD_CONVENTIONS.md](AI_AGENTS/MD_CONVENTIONS.md) — Markdown-JSON hybrid schema all `.md` files follow.
- [AI_AGENTS/WORKLOG.md](AI_AGENTS/WORKLOG.md) — Earlier AI-agent reference and intervention log, superseded by the root `WORKLOG.md`.
- [AI_AGENTS/LINEARIZE_AGENT.md](AI_AGENTS/LINEARIZE_AGENT.md) — Brief for the vectorization effort that produced `VectorizedModel`.
- [AI_AGENTS/ADD_SNAPSHOT_PLAN.md](AI_AGENTS/ADD_SNAPSHOT_PLAN.md) — Design document for the snapshot feature in `VectorizedModel` (records truth share and max belief change at fixed intervals during long simulations).

## Directory Structure
- status: active

```
e_network_inequality/
│
├── 1. Citation Data and Networks Generation.ipynb   # Step 1: fetch OpenAlex data, build networks
├── 2. GColab Simulations.ipynb                      # Step 2: run simulations on Colab
├── 2. GColab Simulations Equality.ipynb             # Step 2 (variant): equality + clustering runs
├── 3. Results Data Analysis.ipynb                   # Step 3: analyse and plot results
├── 3. Local Simulations SA.ipynb                    # Step 3 (variant): local sensitivity analysis
├── 4. Network-Visualizations.ipynb                  # Step 4: network stats and visualisations
├── A. GColab Simulations Playground.ipynb           # Appendix: experimental Colab sandbox
│
├── model/                          # All model and simulation code
│   ├── __init__.py
│   ├── agents.py                   # Legacy OO agent classes: Bandit, BetaAgent, BayesAgent (immutable)
│   ├── model.py                    # Legacy OO Model class (immutable)
│   ├── simulation_functions.py     # Wrappers for running Model in parallel (immutable)
│   ├── bandit.py                   # VectorizedBandit — vectorized multi-armed bandit
│   ├── vectorized_model.py         # Fast vectorized simulation (primary engine)
│   ├── vectorized_simulation_functions.py  # Wrappers for VectorizedModel
│   └── convergence_analysis/       # Targeted convergence studies (notebooks + .py drivers + .md analyses)
│       ├── 00_Colab_Template.ipynb # Starting template for a new study
│       ├── MC_AGENT.md             # Markov-chain analysis brief
│       ├── OPEN_QUESTIONS.md
│       ├── stopping_condition/     # Stopping-criterion studies (7 notebooks = 4 sources
│       │                           #   + 3 executed snapshots; 6 drivers, 2 analyses)
│       ├── phase_dynamics/         # Two-phase convergence dynamics
│       ├── root_node_influence/    # Influence of root/source nodes
│       └── formal_markov/          # Formal Markov-chain treatment
│
├── networks/                       # Network generation and manipulation
│   ├── __init__.py
│   ├── network_generation.py       # Synthetic graph generators (BA, WS, etc.)
│   ├── variation_methods.py        # Network variation utilities (densify, equalize)
│   └── citation_data/              # Pickled empirical networks and raw works (.pkl, .json)
│                                   #   pud_/tobacco_/ego_network.pkl are the three live networks
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── imports.py                  # Central external library re-export hub
│   ├── network_utils.py            # Network statistics and helper functions
│   ├── network_plot_utils.py       # Network plotting helpers
│   ├── mc_analysis.py              # Markov Chain analysis utilities
│   ├── data_analysis_utils.py      # OLS regression, multicollinearity (VIF/Pearson), Cohen's f², diagnostics
│   └── sa_network_variation_directed.py  # Sensitivity analysis over directed network variations
│
├── testing/                        # All tests (no __init__.py — run via unittest discover)
│   ├── unit_tests/                 # Automated test suite — 32 tests
│   │   ├── test_agents.py          # Tests for Bandit and BetaAgent
│   │   ├── test_vectorization.py   # Equivalence tests: Model vs VectorizedModel
│   │   ├── test_mc_analysis.py     # Tests for Markov Chain analysis utilities
│   │   ├── test_stopping_conditions.py    # Tolerance / step / AUC / choice-stability stopping
│   │   ├── basic_model_testing_script.py           # Manual script, not collected by discover
│   │   └── vectorized_basic_model_testing_script.py
│   └── notebooks/                  # Interactive testing notebooks
│       ├── basic_model_testing.ipynb
│       ├── basic_model_testing_v2.ipynb            # Snapshot-enabled variant
│       ├── vectorized_basic_model_testing.ipynb
│       ├── reproducing_zollman.ipynb
│       └── variation_methods_test.ipynb
│
├── results/                        # Analysis notebooks + reference data
│   ├── simulation_analysis.ipynb
│   ├── playground_Hein.ipynb
│   └── zollman_2007.csv            # Zollman (2007) reference values
│
├── figures/                        # Exported figures (currently empty; contents are gitignored)
├── docs/
│   └── COLAB_MCP_WORKFLOW.md       # Driving notebooks on Colab from an MCP client
├── scripts/
│   └── archive/                    # Historical one-shot scripts — not runnable, kept for provenance
│       └── fix_notebook_paths.py
│
├── AI_AGENTS/                      # AI-agent context and historical design documents
│   ├── REPOSITORY_MAP.md           # Module dependency map (partly stale — see above)
│   ├── MD_CONVENTIONS.md           # Markdown-JSON schema conventions
│   ├── WORKLOG.md                  # Earlier AI intervention log (superseded by root WORKLOG.md)
│   ├── LINEARIZE_AGENT.md          # Brief for the vectorization effort
│   └── ADD_SNAPSHOT_PLAN.md        # Snapshot feature design document
│
├── README.md                       # This file
├── TODO_WORKFLOW.md                # Active task list and goals (replaces WORKPLAN.md)
├── WORKLOG.md                      # Append-only chronological change log
├── HOUSEKEEPING.md                 # Routine sanity-check workflow
├── .env.example                    # Template for OPEN_ALEX_API_KEY
├── requirements.txt
└── setup.py
```

Not shown: `.venv/`, `__pycache__/`, `.claude/`, `.pytest_cache/`, `.ruff_cache/`, and your local `.env` — all gitignored.

## Development & Conventions
- status: active

- **Immutable Core Files**: Do not modify `model/agents.py`, `model/model.py`, or `model/simulation_functions.py`. They are the baseline source of truth that `test_vectorization.py` checks `VectorizedModel` against. Create new versions (subclasses or new files) instead.
- **Markdown Conventions**: All `.md` files must follow the [Markdown-JSON Hybrid Schema](AI_AGENTS/MD_CONVENTIONS.md).
- **AI Agents**: If you are an AI assistant, start with this README, then [HOUSEKEEPING.md](HOUSEKEEPING.md) (what to run before and after changes) and [TODO_WORKFLOW.md](TODO_WORKFLOW.md) (what is in flight). [AI_AGENTS/](AI_AGENTS/) holds supplementary and historical context.
- **Import Convention**: All files use absolute imports from the project root (e.g., `from model.vectorized_model import VectorizedModel`, `from utils.imports import *`), except within-package relative imports inside `model/` (e.g. `from .bandit import VectorizedBandit`). Notebooks add the project root to `sys.path` at startup.
- **Verify before committing**: Run the [Housekeeping Workflow](HOUSEKEEPING.md) — unit tests, import check, network integrity, notebook validation, and the snapshot smoke test.
