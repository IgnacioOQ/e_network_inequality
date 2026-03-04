# Repository Map: `e_network_inequality`
- status: active
- type: context
- owner: AI
<!-- content -->

This document details the complete structure of the repository, highlighting explicit module boundaries and import dependencies between internal files and external libraries.

## 1. Directory Structure
- status: active
<!-- content -->

```text
e_network_inequality/
├── AI_AGENTS/                   # Context and convention files for AI agents
│   ├── AGENTS.md                
│   ├── AGENTS_LOG.md            
│   ├── HOUSEKEEPING.md          
│   ├── LINEARIZE_AGENT.md       
│   ├── MC_AGENT.md              
│   ├── MD_CONVENTIONS.md
│   └── REPOSITORY_MAP.md        # This file
│
├── data/                        # Contains simulation output data and empirical network JSONs
│   ├── empirical_networks/
│   └── results_data_sets/
│
├── notebooks/                   # Jupyter notebooks for interactive analysis
│   ├── basic_testing/           # e.g., basic_model_testing.ipynb
│   ├── convergence_analysis/    # e.g., convergence_studies.py
│   └── simulation_variations/   
│
├── src/net_epistemology/        # Main source package
│   ├── __init__.py
│   ├── analysis/                # Markov chain and convergence analysis
│   │   ├── __init__.py
│   │   └── mc_analysis.py       
│   ├── core/                    # Core simulation instances and agents
│   │   ├── __init__.py
│   │   ├── agents.py            # Legacy object-oriented agents
│   │   ├── bandit.py            # Vectorized multi-armed bandit (numpy arrays)
│   │   ├── model.py             # Legacy sequential model definitions
│   │   └── vectorized_model.py  # Fast vectorized model simulation
│   ├── simulation/              # Runners and wrappers
│   │   ├── __init__.py
│   │   ├── simulation_functions.py
│   │   └── vectorized_simulation_functions.py
│   └── utils/                   # Helpers, graph algorithms, and global imports
│       ├── __init__.py
│       ├── imports.py           # Global entry point for external libraries
│       ├── network_generation.py
│       ├── network_utils.py
│       └── variation_methods.py
│
├── tests/                       # Unit tests and regression checks
│   ├── test_mc_analysis.py
│   ├── test_stopping_conditions.py
│   ├── test_vectorization.py
│   └── unit_tests.py
│
├── README.md
├── requirements.txt
└── setup.py
```

## 2. File Imports and Dependencies
- status: active
<!-- content -->

### Core Package Modules (`src/net_epistemology/core/`)
- status: active
<!-- content -->

*   **`__init__.py`**
    *   *Internal Exports*: `Model`, `Bandit`, `BetaAgent`, `BayesAgent`, `VectorizedModel`, `VectorizedBandit`
*   **`model.py`**
    *   *Internal Imports*: `from .agents import Bandit, BayesAgent, BetaAgent`
    *   *External via utils*: `from ..utils.imports import np, nx, rd, tqdm`
*   **`vectorized_model.py`**
    *   *Internal Imports*: `from .bandit import VectorizedBandit`
    *   *External via utils*: `from ..utils.imports import beta, np, nx, rd, tqdm`
*   **`agents.py`**
    *   *External via utils*: `from ..utils.imports import beta, np, rd`
*   **`bandit.py`** *(formerly `vectorized_agents.py`)*
    *   *External via utils*: `from ..utils.imports import np, rd`
    *   *Standard Lib*: `from typing import Tuple`, `from numpy.typing import ArrayLike, NDArray`

### Simulation Package Modules (`src/net_epistemology/simulation/`)
- status: active
<!-- content -->

*   **`simulation_functions.py`**
    *   *Internal Imports*: 
        *   `from ..core.agents import BayesAgent, BetaAgent`
        *   `from ..core.model import Model`
        *   `from ..utils.network_generation import *`
        *   `from ..utils.network_utils import *`
    *   *External via utils*: `from ..utils.imports import *`
*   **`vectorized_simulation_functions.py`**
    *   *Internal Imports*:
        *   `from ..core.vectorized_model import VectorizedModel`
        *   `from ..utils.network_generation import *`
        *   `from ..utils.network_utils import *`
    *   *External via utils*: `from ..utils.imports import *`

### Analysis Modules (`src/net_epistemology/analysis/`)
- status: active
<!-- content -->

*   **`__init__.py`**
    *   *Internal Exports*: `MarkovChainAnalyzer`
*   **`mc_analysis.py`**
    *   *External Libraries*: `import numpy as np`, `import numpy.random as rd`, `import networkx as nx`, `from scipy import stats`
    *   *Standard Lib*: `import hashlib`, `from typing import Optional, List, Tuple, ...`, `from dataclasses import dataclass, field`

### Utils Modules (`src/net_epistemology/utils/`)
- status: active
<!-- content -->

*   **`imports.py`** *(Central dependency file to prevent cyclic imports and ensure library consistency)*
    *   *Standard Lib*: `copy`, `hashlib`, `os`, `pickle`, `random`, `unittest`, `uuid`, `multiprocessing`
    *   *External Libraries*: `dill`, `matplotlib.pyplot as plt`, `networkx as nx`, `numpy as np`, `numpy.random as rd`, `pandas as pd`, `scipy.stats` (specifically `beta`), `seaborn as sns`, `tqdm`
*   **`network_generation.py`**
    *   *Internal Imports*: `from .imports import *`
*   **`network_utils.py`**
    *   *Internal Imports*: `from .imports import *` (though it uses `from imports import *` locally inside the folder as denoted by the search tool, possibly needing an update)
*   **`variation_methods.py`**
    *   *Internal Imports*: `from .imports import *`
    *   *Standard Lib*: `from functools import partial`

### Main Package Root (`src/net_epistemology/`)
- status: active
<!-- content -->

*   **`__init__.py`**
    *   *Internal Imports*: Collects and exposes the core classes (`Model`, `BetaAgent`, `VectorizedModel`, `VectorizedBandit`, etc.) from the `core/` submodule.
