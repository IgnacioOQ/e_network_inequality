# Inequality and the Reliability of Science — Companion Code
- status: active
- type: reference
- description: Simulation code, empirically grounded citation networks, and statistical analysis for the paper "Inequality and the Reliability of Science" (Duijf, Noichl & Ojea Quintana).

This repository is the companion code for the paper **"Inequality and the Reliability of Science"**
by Hein Duijf, Max Noichl and Ignacio Ojea Quintana (author order alphabetical; all authors
contributed equally).

**The claim under test — the equality effect:** scientific communities are more vulnerable to
getting trapped in a false consensus when certain results or scientists are too influential.
Everything else being equal, more equally connected communities should be more reliable. The paper
evaluates this as a *counterfactual* — had the community been more equally connected, would it have
been more reliable? — and that requires more than empirical networks alone: it requires
counterfactual networks that differ in inequality but are otherwise maximally similar to the real
one. Supplying those is the job of the **network variation method**
([`networks/variation_methods.py`](networks/variation_methods.py)), the methodological core of this
repository.

Three strands run through the code: **(1)** building empirically grounded communication networks
from bibliometric data (peptic ulcer disease, tobacco and health, ego depletion); **(2)** generating
inequality-varying counterfactual variants of each; **(3)** running Bayesian bandit simulations
across both and testing statistically whether equality predicts reliability.

The manuscript itself is not tracked here — see [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md)
for what is deliberately excluded and why.

## What the code does
- status: active

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

> ### Raw data is tracked — nothing to do before you pull
>
> The raw OpenAlex dumps (`networks/citation_data/*_works.pkl`, ~150 MB) are **tracked**, alongside
> the derived networks. They were briefly untracked on 2026-08-26 and re-tracked the same day.
>
> Two reasons they stay tracked. They are an **April-2026 snapshot**: OpenAlex is a living
> database, so re-running `1. Citation Data and Networks Generation.ipynb` today returns different
> records and does not reproduce the published networks. And untracking them makes git delete a
> collaborator's local copy on their next pull — that is how git applies a recorded deletion, and no
> `.gitignore` entry prevents it.
>
> If you pulled during that window and lost your copies, restore them exactly — no OpenAlex call,
> byte-identical:
>
> ```bash
> git checkout pre-cleanup-2026-08-26 -- networks/citation_data/
> ```

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

> **Known gap — `NetworkInequality`.** `utils/network_plot_utils.py` imports
> `NetworkInequality.edgebundling`, a small package that lives outside this repository (it was
> written for a separate Hugging Face app) and is on no package index. Importing that module, and
> therefore running `4. Network-Visualizations.ipynb` end to end, currently fails for anyone but the
> authors. The simulation and analysis path (notebooks 1, 2a–2d, 3) is unaffected. Resolving this —
> by vendoring the edge-bundling helper or dropping the dependency — is tracked in
> [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md).

## Usage
- status: active

### Running Tests
- status: active

To verify the installation and core logic (run from project root):
```bash
.venv/bin/python -m pytest testing/unit_tests -q
```

`unittest discover -s testing/unit_tests` also works for all but
`test_equality_study_aggregation.py`, which uses `pytest` fixtures.

For a deeper sanity check (unit tests + import check + network integrity + notebook validation + snapshot smoke test), follow the [Housekeeping Workflow](HOUSEKEEPING.md). Recommended after any non-trivial change to the model or before committing.

### Running Simulations
- status: active

The entry-point notebooks are at the **project root**. The leading number is the workflow stage;
two notebooks sharing a number are alternatives at that stage, not sequential steps.

| Stage | Notebook | Purpose |
|:--|:---|:---|
| 1 | `1. Citation Data and Networks Generation.ipynb` | Fetch data from OpenAlex and build the three empirical citation networks. Needs an API key (see Setup). |
| 2a | `2a. GColab Simulations Equality - Literature.ipynb` | Colab study, literature-standard parameters. |
| 2b | `2b. GColab Simulations Equality - Harder.ipynb` | Colab study, harder inquiry regime. |
| 2c | `2c. GColab Simulations Equality - Phase Transition.ipynb` | Colab study, phase-transition regime. |
| 2d | `2d. GColab Simulations Equality - Aggregation.ipynb` | Aggregates the 2a/2b/2c partial runs into the summary CSVs in `results/`. Resumable across sessions. |
| 3 | `3. Results Data Analysis.ipynb` | Load the summary CSVs, run the regressions, produce the paper's §6.2 figures. |
| 4 | `4. Network-Visualizations.ipynb` | Network statistics and visualisations (Figure 1 family). |

Stages 2a–2c are the three parameter conditions of the simulation study reported in §6.1 of the
paper; they are **alternatives at the same stage, not sequential steps**, and each is a multi-day
Google Colab job whose partial outputs 2d accumulates. The paper's headline runs use problem
easiness 0.001, 1,000 experiments per step, a stability window of 100 and a horizon of 100,000
steps, with 1,000 network variants per empirical network at rewiring probabilities sampled
uniformly in [0, 10%].

For the convergence work that justifies those stopping parameters, see
`model/convergence_analysis/` (described below). For the mapping from each committed figure and CSV
back to the notebook that produced it, see [results/MANIFEST.md](results/MANIFEST.md).

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
- **`model/convergence_analysis/`**: The methodological groundwork behind the stopping condition used
  in the paper, in four themed subdirectories of standalone `.py` drivers and written analyses. The
  exploratory Colab notebooks were removed in the 2026-08-26 cleanup; recover them from the
  `pre-cleanup-2026-08-26` tag if a referee asks.
    - `stopping_condition/` — six drivers (`choice_stability_stopping.py`,
      `convergence_speed_analysis.py`, `parameter_search.py`, `post_stopping_drift.py`,
      `stopping_tolerance_sensitivity.py`, `tolerance_vs_alphabeta.py`) plus
      `STOPPING_CONDITION_ANALYSIS.md` and `CHOICE_STABILITY_STOPPING_PLAN.md`. **This is the
      justification for the stability window of 100 and the horizon of 100,000 used in §6.1.**
    - `phase_dynamics/` — `convergence_studies.py`, `two_phase_dynamics.py`.
    - `root_node_influence/` — `root_influence_analysis.py`, `ROOTNODE_HYPOTHESIS.md`,
      `BANDITSxNETWORKS_ANALYSIS_BRIEF.md`.
    - `formal_markov/` — Markov-chain formalisation: `HYPOTHESIS.md`, `STOCHASTIC_HYPOTHESIS.md`.

    `MC_AGENT.md` and `OPEN_QUESTIONS.md` sit at the top of the folder.
- **`results/`**: The committed paper figures, the two simulation summary CSVs, the
  `zollman_2007.csv` reference dataset, and [`MANIFEST.md`](results/MANIFEST.md), which maps each
  artifact to its source and its place in the paper. Bulk simulation state is not committed — large
  runs write to Google Drive and are aggregated by notebook 2d.

## Project Documentation
- status: active

This repository keeps a deliberately small governance bundle:

- [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md) — What is excluded from the public repository
  and why, what the 2026-08-26 cleanup removed, and the steps remaining before submission. **Read
  this before restoring anything that looks missing.**
- [HOUSEKEEPING.md](HOUSEKEEPING.md) — Routine sanity-check workflow: unit tests, import checks,
  network integrity, notebook validation.
- [results/MANIFEST.md](results/MANIFEST.md) — Figure and data traceability from the paper back to
  the code, including an honest record of which figures are *not* reproducible from this repo.
- [CITATION.cff](CITATION.cff) — Machine-readable citation metadata.
- [LICENSE](LICENSE) — MIT.

## Directory Structure
- status: active

```
e_network_inequality/
│
├── 1. Citation Data and Networks Generation.ipynb   # Stage 1: fetch OpenAlex data, build networks
├── 2a. GColab Simulations Equality - Literature.ipynb        # Stage 2: literature parameters
├── 2b. GColab Simulations Equality - Harder.ipynb            # Stage 2: harder regime
├── 2c. GColab Simulations Equality - Phase Transition.ipynb  # Stage 2: phase-transition regime
├── 2d. GColab Simulations Equality - Aggregation.ipynb       # Stage 2: aggregate partial runs
├── 3. Results Data Analysis.ipynb                   # Stage 3: regressions and paper figures
├── 4. Network-Visualizations.ipynb                  # Stage 4: network stats and visualisations
│
├── model/                          # All model and simulation code
│   ├── agents.py                   # Legacy OO agent classes: Bandit, BetaAgent, BayesAgent (immutable)
│   ├── model.py                    # Legacy OO Model class (immutable)
│   ├── simulation_functions.py     # Wrappers for running Model in parallel (immutable)
│   ├── bandit.py                   # VectorizedBandit — vectorized multi-armed bandit
│   ├── vectorized_model.py         # Fast vectorized simulation (primary engine)
│   ├── vectorized_simulation_functions.py  # Wrappers for VectorizedModel
│   ├── equality_study.py           # The paper's simulation study: variants, runs, aggregation
│   └── convergence_analysis/       # Stopping-condition groundwork (.py drivers + .md analyses)
│       ├── stopping_condition/     # Justifies the §6.1 stopping parameters
│       ├── phase_dynamics/         # Two-phase convergence dynamics
│       ├── root_node_influence/    # Influence of root/source nodes
│       └── formal_markov/          # Formal Markov-chain treatment
│
├── networks/                       # Network generation and manipulation
│   ├── network_generation.py       # Synthetic graph generators (BA, WS, etc.)
│   ├── variation_methods.py        # The network variation method (paper §5)
│   └── citation_data/              # The three empirical networks + PUD/perceptron artifacts
│                                   #   pud_/tobacco_/ego_network.pkl are the three live networks
│                                   #   *_works.pkl (raw OpenAlex, ~150 MB) are gitignored
│
├── utils/                          # Shared utilities
│   ├── imports.py                  # Central external library re-export hub
│   ├── network_utils.py            # Network statistics and helper functions
│   ├── network_plot_utils.py       # Network plotting helpers
│   ├── mc_analysis.py              # Markov Chain analysis utilities
│   └── data_analysis_utils.py      # OLS regression, multicollinearity (VIF/Pearson), Cohen's f²
│
├── testing/unit_tests/             # Automated test suite (run via unittest discover)
│   ├── test_agents.py              # Tests for Bandit and BetaAgent
│   ├── test_vectorization.py       # Equivalence tests: Model vs VectorizedModel
│   ├── test_mc_analysis.py         # Tests for Markov Chain analysis utilities
│   ├── test_stopping_conditions.py # Tolerance / step / AUC / choice-stability stopping
│   └── test_equality_study_aggregation.py  # Aggregation contract for notebooks 2a–2d
│
├── results/                        # Paper figures, summary CSVs, and MANIFEST.md
├── figures/                        # Local figure exports (gitignored contents)
│
├── README.md                       # This file
├── PUBLICATION_CHECKLIST.md        # What is excluded and why; pre-submission steps
├── HOUSEKEEPING.md                 # Routine sanity-check workflow
├── CITATION.cff                    # Machine-readable citation metadata
├── LICENSE                         # MIT
├── .env.example                    # Template for OPEN_ALEX_API_KEY
├── requirements.txt
└── pyproject.toml
```

Not shown: `.venv/`, `__pycache__/`, `.claude/`, `.pytest_cache/`, `.ruff_cache/`, and your local `.env` — all gitignored.

## Development & Conventions
- status: active

- **Immutable Core Files**: Do not modify `model/agents.py`, `model/model.py`, or `model/simulation_functions.py`. They are the baseline source of truth that `test_vectorization.py` checks `VectorizedModel` against. Create new versions (subclasses or new files) instead.
- **Scope discipline**: This repository is scoped to the paper. Before adding a file, ask whether a
  reader reproducing the paper's results needs it — see [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md).
- **No CI, no merge gates, no linters.** Research code; the correctness unit is the figure. The
  honest substitute for a build is Restart-and-Run-All over the notebooks, plus the unit tests.
- **Import Convention**: All files use absolute imports from the project root (e.g., `from model.vectorized_model import VectorizedModel`, `from utils.imports import *`), except within-package relative imports inside `model/` (e.g. `from .bandit import VectorizedBandit`). Notebooks add the project root to `sys.path` at startup.
- **Verify before committing**: Run the [Housekeeping Workflow](HOUSEKEEPING.md) — unit tests, import check, network integrity, notebook validation, and the snapshot smoke test.

## Citation and License
- status: active

If you use this code or the derived networks, please cite the paper. Machine-readable metadata is in
[CITATION.cff](CITATION.cff); GitHub renders it in the sidebar.

> Duijf, H., Noichl, M., & Ojea Quintana, I. (2026). *Inequality and the Reliability of Science*.

The bibliometric data underlying the networks comes from [OpenAlex](https://openalex.org/) and is
subject to OpenAlex's terms. The code is released under the [MIT License](LICENSE),
© 2026 Hein Duijf, Max Noichl, and Ignacio Ojea Quintana.

**Archived version:** a DOI-bearing snapshot has not yet been minted — see
[PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md).
