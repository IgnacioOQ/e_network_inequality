---
type: how-to
label: [skill, notebook, colab, simulation]
status: active
owner: user
---

# How to Create and Structure Colab Notebooks

<!-- content -->
This document explains how to set up Jupyter notebooks for Google Colab within the `e_network_inequality` project. Because Colab environments are ephemeral, the notebook must clone the repository, install dependencies, load the project into the path, and correctly mount Google Drive before any simulations run. 

Use the accompanying `00_Colab_Template.ipynb` as your boilerplate when starting new studies.

## 0. Notebook Format Requirements
All notebooks must be saved with **nbformat ≥ 4.5** so that every cell carries a stable `id` field. Cell IDs make notebooks scriptable — individual cells can be addressed by AI tooling, migration scripts, and comment threads — and they stabilise diffs in version control by keeping a cell's identity invariant under reordering or insertions.

Modern Jupyter, VS Code, and Colab assign IDs automatically when saving a 4.5+ notebook, so no manual work is required for new cells once the format is in place. If you start a notebook by copying `00_Colab_Template.ipynb`, IDs are already wired up and the editor will continue assigning them on save.

If you ever encounter an older notebook (e.g. one created in pre-2021 Colab), bump its `nbformat_minor` to `5` and let the editor re-save it; existing cells will receive freshly-generated IDs.

## 1. Initial Setup and Environment Sourcing
The first cells of any notebook must fetch the repository and install required packages (like `dill`). This ensures that the codebase used in Colab matches the `main` branch.

```python
import shutil, os
if os.path.exists('e_network_inequality'):
    shutil.rmtree('e_network_inequality')

!git clone -b main https://github.com/IgnacioOQ/e_network_inequality
```
Then install `dill` and `cd` into the repo:
```python
!pip install dill
%cd e_network_inequality
```

## 2. Path Insertion and Imports
Because the project uses absolute imports, the current working directory (`e_network_inequality`) must be inserted into `sys.path`.

```python
import sys, os
sys.path.insert(0, os.getcwd())

from utils.imports import *
from model.model import Model
from model.vectorized_model import VectorizedModel
from model.vectorized_simulation_functions import *
# ... (see template for full imports)
```

## 3. Google Drive Mounting
Simulations often output large datasets and plots. These must be saved directly to Google Drive.
```python
from google.colab import drive
drive.mount('/content/drive')

dumping_path = '/content/drive/My Drive/Colab Projects/Data Driven ABMs/Data Sets/'
```

## 4. Loading Networks
Always load the standard networks (`pud_network`, `tobacco_network`, `ego_network`) via `pickle` and reindex them to ensure nodes are zero-indexed integers (which is necessary for the vectorized model).

```python
import pickle
import networkx as nx

with open('./networks/citation_data/pud_network.pkl', 'rb') as f:
    G_pud = pickle.load(f)
mapping = {node: idx for idx, node in enumerate(G_pud.nodes())}
G_pud_indexed = nx.relabel_nodes(G_pud, mapping)
```

## 5. Running Simulations
Use `multiprocessing.Pool` combined with `functools.partial` and `tqdm` to run `run_vectorized_simulation_with_params` in parallel.

```python
from multiprocessing import Pool, cpu_count
num_cores = cpu_count()

wrapper = partial(
    run_vectorized_simulation_with_params,
    tolerance=1e-5,
    number_of_steps=100_000,
    # ...
)

with Pool(num_cores) as pool:
    results = list(tqdm(pool.imap_unordered(wrapper, param_dicts), total=N_RUNS))
```

## 6. Disconnecting Runtime
Since simulations might be long-running, always append a cell at the end of the notebook to automatically disconnect the Google Colab runtime to preserve compute units.

```python
from datetime import datetime
import pytz
from IPython.display import Javascript

nyc_time = datetime.now(pytz.timezone('America/New_York'))
print(f"✅ Disconnected from runtime at: {nyc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
display(Javascript('google.colab.kernel.disconnect()'))
```
