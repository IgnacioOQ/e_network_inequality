# Repository Map: `e_network_inequality`
- status: active
- type: context
- owner: AI
- last_checked: 2026-03-15
<!-- content -->

This document details the complete structure of the repository, highlighting explicit module boundaries and import dependencies between internal files and external libraries.

## 1. Directory Structure
- status: active
<!-- content -->

```text
e_network_inequality/
│
├── 1. Citation Data and Networks Generation.ipynb   # Entry point 1: build networks
├── 2. Colab Simulations.ipynb                       # Entry point 2: run simulations
├── 3. Simulations Data Analysis.ipynb               # Entry point 3: analyse results
├── A. Visualizations.ipynb                          # Entry point 4: visualisations
│
├── model/                          # All model and simulation code
│   ├── __init__.py
│   ├── agents.py                   # Legacy OO agent classes (IMMUTABLE)
│   ├── bandit.py                   # Vectorized multi-armed bandit
│   ├── model.py                    # Legacy OO Model class (IMMUTABLE)
│   ├── vectorized_model.py         # Fast vectorized simulation (primary)
│   ├── simulation_functions.py     # Wrappers for running Model in parallel (IMMUTABLE)
│   ├── vectorized_simulation_functions.py  # Wrappers for VectorizedModel
│   └── convergence_analysis/       # Colab notebooks for convergence studies
│
├── networks/                       # Network generation and manipulation
│   ├── __init__.py
│   ├── network_generation.py       # Synthetic graph generators (BA, WS, etc.)
│   ├── variation_methods.py        # Network variation utilities (densify, equalize)
│   └── citation_data/              # Pickled empirical network files (.pkl, .json)
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── imports.py                  # Central external library imports (used by all modules)
│   ├── network_utils.py            # Network statistics and helper functions
│   └── mc_analysis.py              # Markov Chain analysis utilities
│
├── testing/                        # All tests
│   ├── unit_tests/                 # Automated test suite
│   │   ├── test_agents.py          # Tests for Bandit and BetaAgent
│   │   ├── test_vectorization.py   # Equivalence: Model vs VectorizedModel
│   │   ├── test_mc_analysis.py     # Tests for Markov Chain analysis
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
│   ├── AGENTS.md
│   ├── AGENTS_LOG.md
│   ├── HOUSEKEEPING.md
│   ├── LINEARIZE_AGENT.md
│   ├── MC_AGENT.md
│   ├── MD_CONVENTIONS.md
│   └── REPOSITORY_MAP.md           # This file
│
├── README.md
├── requirements.txt
└── setup.py
```

## 2. Import Convention
- status: active
<!-- content -->

All modules use **absolute imports from the project root**. Notebooks add the project root to `sys.path` at startup via:

```python
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))  # from testing/notebooks/
```

**Key rules:**
- `from utils.imports import *` — pulls all shared external libraries
- `from model.X import Y` — imports from model package
- `from networks.X import Y` — imports from networks package
- `from utils.network_utils import ...` — note: `network_utils` is in `utils/`, NOT `networks/`
- `from networks.variation_methods import ...` — `variation_methods` is in `networks/`

## 3. File Imports and Dependencies
- status: active
<!-- content -->

### `model/`
- status: active
<!-- content -->

| File | Imports |
| :--- | :--- |
| `agents.py` | `from utils.imports import beta, np, rd` |
| `bandit.py` | `from utils.imports import beta, np, nx, rd, tqdm` |
| `model.py` | `from utils.imports import np, nx, rd, tqdm` · `from .agents import Bandit, BayesAgent, BetaAgent` |
| `vectorized_model.py` | `from utils.imports import beta, np, nx, rd, tqdm` · `from .bandit import VectorizedBandit` |
| `simulation_functions.py` | `from utils.imports import *` · `from model.agents import ...` · `from model.model import Model` · `from networks.network_generation import *` · `from utils.network_utils import *` |
| `vectorized_simulation_functions.py` | `from utils.imports import *` · `from model.vectorized_model import VectorizedModel` · `from networks.network_generation import *` · `from utils.network_utils import *` |

### `networks/`
- status: active
<!-- content -->

| File | Imports |
| :--- | :--- |
| `network_generation.py` | `from utils.imports import *` |
| `variation_methods.py` | `from utils.imports import *` · `from functools import partial` |

### `utils/`
- status: active
<!-- content -->

| File | Imports |
| :--- | :--- |
| `imports.py` | External libraries only (`numpy`, `scipy`, `networkx`, `pandas`, `tqdm`, `dill`, `matplotlib`, `seaborn`, `statsmodels`, `joblib`, std lib) |
| `network_utils.py` | `from utils.imports import *` |
| `mc_analysis.py` | `from model.vectorized_model import VectorizedModel` · external libraries directly |
