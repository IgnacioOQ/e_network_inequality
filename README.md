# Inequality and the Reliability of Science — Companion Code

Companion code for **"Inequality and the Reliability of Science"** by Hein Duijf, Max Noichl and
Ignacio Ojea Quintana (author order alphabetical; all authors contributed equally).

**The claim under test — the equality effect:** scientific communities are more vulnerable to a
false consensus when certain results or scientists are too influential; all else equal, more
equally connected communities should be more reliable. The paper evaluates this as a
*counterfactual* — had the community been more equally connected, would it have been more reliable?
— which needs more than empirical networks: it needs counterfactual networks that differ in
inequality but are otherwise maximally similar to the real one. Supplying those is the job of the
network variation method ([`networks/variation_methods.py`](networks/variation_methods.py)), the
methodological core of this repository.

Three strands run through the code: **(1)** building empirically grounded communication networks
from bibliometric data (peptic ulcer disease, tobacco and health, ego depletion); **(2)** generating
inequality-varying counterfactual variants of each; **(3)** running Bayesian bandit simulations
across both and testing statistically whether equality predicts reliability.

The manuscript itself is not tracked here — see [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md)
for what is deliberately excluded and why.

## The model

Agents hold Beta(α, β) credences over two competing theories and sit in a directed graph whose
edges are observational channels (A listens to B). Each step: agents run a two-armed bandit
experiment on their currently preferred theory, observe their predecessors' outcomes, and update
by Bayesian inference.

Two implementations exist:

- **Object-oriented** (`model/model.py`, `model/agents.py`) — readable baseline, held immutable as
  the source of truth that `test_vectorization.py` checks against.
- **Vectorized** (`model/vectorized_model.py`, `model/bandit.py`) — the primary engine; NumPy
  matrix updates across the whole graph at once. Build new studies on this one.

Studies run iterations in parallel through `multiprocessing.Pool`, with
`model/vectorized_simulation_functions.py` bridging parameters to model execution and
`model/equality_study.py` holding the paper's study itself (variants, runs, aggregation).

### Stopping conditions

`VectorizedModel` supports four ways to end a run. The first three are mutually exclusive modes set
on the constructor; AUC stopping is a separate flag on `run_simulation`:

| Mode | Flag | Stops when |
|:---|:---|:---|
| Tolerance (default) | `tolerance_stopping=True` | Max absolute credence change in a step falls below `tolerance` |
| Fixed steps | `tstep_stopping=True` | `number_of_steps` is reached; no early exit |
| Choice stability | `choice_stability_stopping=True` | *Every* agent's chosen theory is unchanged for `choice_stability_window` consecutive steps |
| AUC-ROC | `run_simulation(auc_stopping=True)` | Node-level AUC-ROC reaches `auc_threshold` (default 0.95) |

Choice-stability stopping addresses the false-convergence failure mode of tolerance stopping: a
quiet step can fire `allclose` while the network is still drifting. It is governed by
`choice_stability_window` (default 500), `choice_stability_min_steps` (a floor against stopping on
transient early stability), and `record_choice_flips`, which logs `(step, truth_share)` to
`choice_flip_history` so a whole window sweep can be derived offline from one record-once run.

### Random seeds in parallel studies

Without an explicit seed, `multiprocessing.Pool` workers fork-inherit the parent's RNG state and
silently produce identical trajectories. `run_vectorized_simulation_with_params` handles this in
two modes:

- **Default (no `seed` in `param_dict`)** — a fresh seed is drawn from OS entropy per job. Different
  seed each run, but not reproducible: you cannot re-run a specific outlier.
- **Reproducible (recommended for published studies)** — derive child seeds from a master seed and
  attach them before the `Pool.imap_unordered` call:

  ```python
  from numpy.random import SeedSequence
  ss = SeedSequence(MASTER_SEED)
  child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_simulations)]
  for pd_, cs in zip(param_dicts, child_seeds):
      pd_["seed"] = cs
  ```

  `SeedSequence` guarantees statistically independent streams; `seed = i` or `seed = master + i`
  does not.

Either way the seed actually used is written to `result_dict["seed"]`, so any single run can be
replayed even when the study was not pre-seeded.

## Empirical networks

Networks are derived from **OpenAlex** (via `pyalex`), covering three episodes in which a scientific
consensus shifted. All three are built by `1. Citation Data and Networks Generation.ipynb`.

| Episode | OpenAlex query | Years | Network file | Size |
|:---|:---|:---|:---|:---|
| Peptic ulcer disease | `"peptic ulcer disease"` | 1900–1978 | `pud_network.pkl` | 90 nodes, 160 edges |
| Tobacco and health | `(tobacco OR smoking OR cigarette) AND (health OR cancer OR lung)` | 1900–1964 | `tobacco_network.pkl` | 289 nodes, 1229 edges |
| Ego depletion | `"ego depletion"` | 1900–2016 | `ego_network.pkl` | 503 nodes, 2933 edges |

Nodes are authors; edges are citations between them, directed from cited to citing. The raw data is
cleaned by pruning "twins" (authors who always co-author, treated as one epistemic unit), taking the
largest weakly connected component, and removing self-loops.

`networks/citation_data/` holds, for each episode, the raw OpenAlex dump (`*_works.pkl`) and the
derived network (`*_network.pkl`). Both are **tracked**: the dumps are an April-2026 snapshot, and
because OpenAlex is a living database, re-running notebook 1 today returns different records and
does not reproduce the published networks.

## Installation

Python **3.10 or newer**. The codebase uses PEP 585 builtin generics without
`from __future__ import annotations`, so older interpreters raise `TypeError`.

With [uv](https://docs.astral.sh/uv/) (the canonical path — `pyproject.toml` and `uv.lock` are the
source of truth):

```bash
uv sync                 # simulation, analysis and test dependencies
uv sync --extra viz     # additionally: notebook 4's plotting stack
```

With pip, from the generated lockfile export:

```bash
pip install -r requirements.txt
```

Notebook 1 also needs an OpenAlex API key: copy `.env.example` to `.env` and fill in
`OPEN_ALEX_API_KEY`. The other notebooks read the pre-built networks and need no key.

**Known gap — `NetworkInequality`.** `utils/network_plot_utils.py` imports
`NetworkInequality.edgebundling`, a small package that lives outside this repository (written for a
separate Hugging Face app) and is on no package index. Running `4. Network-Visualizations.ipynb`
end to end therefore fails for anyone but the authors; notebooks 1, 2a–2d and 3 are unaffected.
Resolving this is tracked in [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md).

## Usage

### Tests

From the project root:

```bash
.venv/bin/python -m pytest unit_tests -q
```

`unit_tests/test_repo_integrity.py` checks the repository rather than the model: that every module
imports from the project root, that the star-import hubs still export, that each citation network
unpickles at its published size, and that the notebooks are valid JSON. It ends with an end-to-end
simulation smoke test, marked `slow` — skip it with `pytest unit_tests -m "not slow"`.

`unittest discover -s unit_tests` also works, but collects only the `unittest`-style files:
`test_equality_study_aggregation.py` and `test_repo_integrity.py` use pytest fixtures and markers.

### Notebooks

The entry points are at the project root. The leading number is the workflow stage; notebooks
sharing a number are alternatives at that stage, not sequential steps.

| Stage | Notebook | Purpose |
|:--|:---|:---|
| 1 | `1. Citation Data and Networks Generation.ipynb` | Fetch OpenAlex data and build the three empirical networks. Needs an API key. |
| 2a | `2a. GColab Simulations Equality - Literature.ipynb` | Colab study, literature-standard parameters. |
| 2b | `2b. GColab Simulations Equality - Harder.ipynb` | Colab study, harder inquiry regime. |
| 2c | `2c. GColab Simulations Equality - Phase Transition.ipynb` | Colab study, phase-transition regime. |
| 2d | `2d. GColab Simulations Equality - Aggregation.ipynb` | Aggregates the 2a/2b/2c partial runs into the summary CSVs in `results/`. Resumable across sessions. |
| 3 | `3. Results Data Analysis.ipynb` | Load the summary CSVs, run the regressions, produce the §6.2 figures. |
| 4 | `4. Network-Visualizations.ipynb` | Network statistics and visualisations (Figure 1 family). |

Stages 2a–2c are the three parameter conditions of the study reported in §6.1; each is a multi-day
Google Colab job whose partial outputs 2d accumulates. The headline runs use problem easiness 0.001,
1,000 experiments per step, a stability window of 100 and a horizon of 100,000 steps, with 1,000
network variants per empirical network at rewiring probabilities sampled uniformly in [0, 10%].

For the mapping from each committed figure and CSV back to the notebook that produced it, see
[results/MANIFEST.md](results/MANIFEST.md).

## Repository layout

```
e_network_inequality/
│
├── 1. Citation Data and Networks Generation.ipynb   # Stage 1: fetch OpenAlex data, build networks
├── 2a-2c. GColab Simulations Equality - *.ipynb     # Stage 2: the three parameter conditions
├── 2d. GColab Simulations Equality - Aggregation.ipynb
├── 3. Results Data Analysis.ipynb                   # Stage 3: regressions and paper figures
├── 4. Network-Visualizations.ipynb                  # Stage 4: network stats and visualisations
│
├── model/
│   ├── agents.py, model.py, simulation_functions.py   # Legacy OO baseline (immutable)
│   ├── bandit.py, vectorized_model.py                 # Primary vectorized engine
│   ├── vectorized_simulation_functions.py             # Parallel-run wrappers
│   └── equality_study.py                              # The paper's study: variants, runs, aggregation
│
├── networks/
│   ├── network_generation.py       # Synthetic graph generators (BA, WS, etc.)
│   ├── variation_methods.py        # The network variation method (paper §5)
│   └── citation_data/              # *_works.pkl (raw OpenAlex) + *_network.pkl (derived)
│
├── utils/
│   ├── imports.py                  # Central external-library re-export hub
│   ├── network_utils.py            # Network statistics and helpers
│   ├── network_plot_utils.py       # Network plotting helpers
│   ├── mc_analysis.py              # Markov chain analysis utilities
│   └── data_analysis_utils.py      # OLS regression, VIF/Pearson, Cohen's f²
│
├── unit_tests/                     # Automated test suite
│
├── results/                        # Summary CSVs, zollman_2007.csv, MANIFEST.md
│   └── figures/                    # Committed paper figures
│
├── README.md, PUBLICATION_CHECKLIST.md
├── CITATION.cff, LICENSE, .env.example
└── pyproject.toml, uv.lock, requirements.txt
```

## Conventions

- **Immutable core.** Do not modify `model/agents.py`, `model/model.py` or
  `model/simulation_functions.py` — they are the baseline that `test_vectorization.py` checks
  `VectorizedModel` against. Subclass or add new files instead.
- **Scope discipline.** Before adding a file, ask whether a reader reproducing the paper's results
  needs it — see [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md).
- **No CI, no merge gates, no linters.** Research code; the correctness unit is the figure. The
  honest substitute for a build is Restart-and-Run-All over the notebooks, plus the unit tests.
- **Imports** are absolute from the project root (`from model.vectorized_model import ...`), except
  within-package relative imports inside `model/`. Notebooks add the project root to `sys.path` at
  startup.

Supporting documents: [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md) (what is excluded and
why, plus remaining pre-submission steps), [results/MANIFEST.md](results/MANIFEST.md) (figure and
data traceability, including which figures are *not* reproducible from this repo),
[CITATION.cff](CITATION.cff), [LICENSE](LICENSE).

## Citation and license

If you use this code or the derived networks, please cite the paper; machine-readable metadata is in
[CITATION.cff](CITATION.cff).

> Duijf, H., Noichl, M., & Ojea Quintana, I. (2026). *Inequality and the Reliability of Science*.

The bibliometric data comes from [OpenAlex](https://openalex.org/) and is subject to OpenAlex's
terms. The code is released under the [MIT License](LICENSE), © 2026 Hein Duijf, Max Noichl and
Ignacio Ojea Quintana.

A DOI-bearing archived snapshot has not yet been minted — see
[PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md).
