# Inequality and the Reliability of Science — Companion Code

Companion code and data for **"Inequality and the Reliability of Science"** by Hein Duijf, Max Noichl
and Ignacio Ojea Quintana (author order alphabetical; all authors contributed equally).

This repository contains everything needed to reproduce the paper's simulation results, tables and
figures: the bibliometric networks, the network variation method, the simulation model, the
per-variant result summaries, and the statistical analysis. The manuscript itself is not tracked
here.

## Contents

1. [Overview](#overview)
2. [Reproducing the results](#reproducing-the-results)
3. [Installation](#installation)
4. [Empirical networks](#empirical-networks)
5. [The model](#the-model)
6. [The equality study](#the-equality-study)
7. [Statistical analysis and figures](#statistical-analysis-and-figures)
8. [Validation](#validation)
9. [Repository layout](#repository-layout)
10. [Conventions](#conventions)
11. [Citation, license and data sources](#citation-license-and-data-sources)
12. [References](#references)

## Overview

**The claim under test — the equality effect.** Scientific communities are more vulnerable to a
false consensus when certain results or scientists are too influential; all else equal, more
equally connected communities should be more reliable. The paper evaluates this as a
*counterfactual*: had the community been more equally connected, would it have been more reliable?
Answering that needs more than empirical networks. It needs counterfactual networks that differ in
inequality but are otherwise as similar as possible to the real one. Supplying those is the job of
the network variation method ([`networks/variation_methods.py`](networks/variation_methods.py)),
the methodological core of this repository.

The code has three parts:

1. **Empirical networks.** Author citation networks are built from bibliometric data for three
   episodes in which a scientific consensus shifted: peptic ulcer disease, tobacco and health, and
   ego depletion.
2. **Counterfactual variants.** For each network, the *equalize* method generates variants that
   lower degree inequality while holding density fixed and clustering approximately fixed.
3. **Simulation and analysis.** A Bayesian bandit model of collective inquiry is run on every
   variant, and regressions test whether degree inequality predicts reliability.

## Reproducing the results

The repository supports three levels of reproduction, from cheapest to most expensive. Each starts
from files that are committed, so no level depends on re-running the one below it.

| Level | What it reproduces | Run | Cost |
|:---|:---|:---|:---|
| Analysis | The regression tables and regression figures, from the committed variant summaries | Notebook 3 | Minutes on a laptop |
| Simulation | The variant summaries, from the committed networks | Notebook 2 with `SMOKE_TEST = False`, then notebook 3 | About 56 hours on 60 cores (see [The equality study](#the-equality-study)) |
| Data | The networks, from OpenAlex | Notebook 1 | Needs an API key; **does not reproduce the published networks** (see [Empirical networks](#empirical-networks)) |

The entry points are four notebooks at the project root, numbered by workflow stage:

| Stage | Notebook | Purpose |
|:--|:---|:---|
| 1 | `1. Citation Data and Networks Generation.ipynb` | Fetch OpenAlex data and build the three empirical networks. |
| 2 | `2. Simulations.ipynb` | Run the equality study: one self-contained, resumable study per difficulty condition, then combine the four summaries. |
| 3 | `3. Results Data Analysis.ipynb` | Regress reliability on degree Gini, controlling for clustering: the combined LaTeX table, the regression figures, and per-condition detail. |
| 4 | `4. Network-Visualizations.ipynb` | Draw the network figures: the tobacco network beside an equalized variant, and cycle, wheel and complete graphs beside the peptic ulcer disease network. |

Notebooks 2 and 3 run either locally (`RUNNING_LOCALLY = True`, reading and writing
`results/equality_study/`) or on Google Colab (`RUNNING_LOCALLY = False`: clone the repository, read
and write Google Drive). Run locally, notebook 3 renders figure text with LaTeX and so needs a TeX
installation.

## Installation

Python **3.10 or newer** is required. The codebase uses PEP 585 builtin generics without
`from __future__ import annotations`, so older interpreters raise `TypeError`.

With [uv](https://docs.astral.sh/uv/) (the canonical path; `pyproject.toml` and `uv.lock` are the
source of truth for dependency versions):

```bash
uv sync
```

This creates a local `.venv` with the simulation, analysis, notebook-kernel and test dependencies,
which covers notebooks 1–3 and the test suite. In your editor's notebook kernel picker, select
`.venv/bin/python`. To activate the same environment in a terminal on macOS or Linux:

```bash
source .venv/bin/activate
```

With pip, from the exported lockfile:

```bash
pip install -r requirements.txt
```

**Notebook 1** also needs an OpenAlex API key: copy `.env.example` to `.env` and fill in
`OPEN_ALEX_API_KEY`. The other notebooks read the pre-built networks and need no key.

**Notebook 4** needs two packages that are not in `pyproject.toml` or `uv.lock`:
[Datashader](https://datashader.org/) for edge bundling and
[pylabeladjust](https://github.com/MNoichl/pylabeladjust) for node-overlap removal. Install them
into the environment separately:

```bash
uv pip install datashader pylabeladjust
```

The alternative overlap method (`OVERLAP_METHOD = "vpsc"`) instead requires Graphviz's `neato` on
`PATH`.

## Empirical networks

Networks are derived from [OpenAlex](https://openalex.org/) (queried with `pyalex`) and cover three
episodes in which a scientific consensus shifted. All three are built by notebook 1.

| Episode | OpenAlex full-text query | Years | Network file | Nodes | Edges |
|:---|:---|:---|:---|---:|---:|
| Peptic ulcer disease | `"peptic ulcer disease"` | 1900–1978 | `pud_network.pkl` | 90 | 160 |
| Tobacco and health | `(tobacco OR smoking OR cigarette) AND (health OR cancer OR lung)` | 1900–1964 | `tobacco_network.pkl` | 289 | 1,229 |
| Ego depletion | `"ego depletion"` | 1900–2016 | `ego_network.pkl` | 503 | 2,933 |

Nodes are authors. Edges are citations between authors, directed **from the cited author to the
citing author**, that is, in the direction information flows. The records are restricted to journal
articles with a reference list, and the resulting network is cleaned in three steps: "twins"
(authors who always co-author, treated as one epistemic unit) are pruned, the largest weakly
connected component is kept, and self-loops are removed.

`networks/citation_data/` holds, for each episode, the raw OpenAlex records (`*_works.pkl`) and the
derived network (`*_network.pkl`, a pickled `networkx.DiGraph`). Both are tracked on purpose. The
raw records are a snapshot taken in April 2026, and because OpenAlex is a living database,
re-running notebook 1 today returns different records and does not reproduce the published
networks. The committed files are the authoritative data for the paper.

## The model

The model follows the bandit models of network epistemology introduced by Zollman (2007, 2010).
Agents choose between two competing theories and sit in a directed graph whose edges are
observational channels. In each step, every agent runs a two-armed bandit experiment on the theory
it currently prefers, observes the outcomes of its predecessors in the graph, and updates by
Bayesian inference. In the paper's study, agents hold Beta(α, β) credences about each theory's
success rate (`agent_type="beta"`); a single-credence variant (`agent_type="bayes"`) is also
available and is used in the Zollman (2007) replication.

Two implementations exist:

- **Object-oriented** (`model/model.py`, `model/agents.py`): the readable baseline, kept unchanged
  as the reference that `unit_tests/test_vectorization.py` checks against.
- **Vectorized** (`model/vectorized_model.py`, `model/bandit.py`): the primary engine, which updates
  the whole graph at once with NumPy matrix operations. All results in the paper come from this
  implementation.

Studies run their replicates in parallel through `multiprocessing.Pool`.
`model/vectorized_simulation_functions.py` maps a parameter dictionary to one model run, and
`model/equality_study.py` holds the paper's study itself (variants, runs, checkpointing and
aggregation).

### Stopping conditions

`VectorizedModel` supports four ways to end a run. The first three are mutually exclusive modes set
on the constructor; AUC stopping is a separate flag on `run_simulation`.

| Mode | Flag | Stops when |
|:---|:---|:---|
| Tolerance (default) | `tolerance_stopping=True` | No credence changes by more than `tolerance` in a step |
| Fixed steps | `tstep_stopping=True` | `number_of_steps` is reached; no early exit |
| Choice stability | `choice_stability_stopping=True` | *Every* agent's chosen theory is unchanged for `choice_stability_window` consecutive steps |
| AUC-ROC | `run_simulation(auc_stopping=True)` | Node-level AUC-ROC reaches `auc_threshold` (default 0.95) |

The paper's study uses choice-stability stopping. It addresses a false-convergence failure of
tolerance stopping, in which a single quiet step ends the run while the network is still drifting.
It is governed by `choice_stability_window` (default 500), `choice_stability_min_steps` (a floor
against stopping on transient early stability, default 0), and `record_choice_flips`, which logs
`(step, truth_share)` to `choice_flip_history` so that a whole sweep over window sizes can be
derived offline from one recorded run.

### Random seeds in parallel studies

Without an explicit seed, `multiprocessing.Pool` workers inherit the parent's random-number state
on fork and silently produce identical trajectories. `run_vectorized_simulation_with_params`
handles this in two modes:

- **Default (no `seed` in `param_dict`).** A fresh seed is drawn from OS entropy for each job. Runs
  differ from each other, but the study as a whole is not reproducible from a master seed.
- **Reproducible (used for the paper's study).** Child seeds are derived from a master seed and
  attached before the jobs are dispatched:

  ```python
  from numpy.random import SeedSequence
  ss = SeedSequence(MASTER_SEED)
  child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_simulations)]
  for pd_, cs in zip(param_dicts, child_seeds):
      pd_["seed"] = cs
  ```

  `SeedSequence` guarantees statistically independent streams; `seed = i` or `seed = master + i`
  does not.

In both modes the seed actually used is written to `result_dict["seed"]`, so any single run can be
replayed.

## The equality study

The study in notebook 2 (machinery in `model/equality_study.py`) is nested. For each empirical
network it builds `N_VARIANTS` counterfactual variants with the *equalize* method, and simulates
each variant `N_RUNS` times under different seeds. Each variant edits a fraction of the network's
edges drawn uniformly from [0, 10%]. Replicates of a variant share every parameter and differ only
in their simulation seed.

Problem difficulty, the gap ε between the two theories' success rates, is the one parameter that
varies across the four conditions. Everything else is held fixed.

| Condition | ε | Master seed | Wall-clock time (60 cores) |
|:---|:---|:---|---:|
| `easy` | 0.1 | 20260722 | 1.1 h |
| `moderate` | 0.01 | 20260723 | 1.5 h |
| `hard` | 0.001 | 20260724 | 5.3 h |
| `super_hard` | 0.0001 | 20260725 | 47.7 h |

Settings shared by the four full runs:

| Parameter | Value |
|:---|:---|
| Variants per network | 1,000 |
| Runs per variant | 1,000 |
| Experiments per step (`n_experiments`) | 1,000 |
| Stopping rule | Choice stability, window 100, no minimum number of steps |
| Maximum steps | 100,000 |
| Variation method | *equalize*, at most 10% of edges edited |

This gives 3 networks × 4 conditions × 1,000 variants = 12,000 variant summaries, each averaging
1,000 runs.

Three switches at the top of notebook 2 control a run:

- `SMOKE_TEST`. As committed, the notebook has `SMOKE_TEST = True`, which runs a tiny grid (3
  variants, 8 runs each) into a separate `smoke/` directory to check the plumbing. **Set it to
  `False` to run the full study.** Smoke outputs measure nothing and must not be analysed.
- `ACCUMULATE` (default `True`). Every variant is checkpointed as it finishes, so an interrupted
  study resumes where it stopped. `False` deletes the existing run and starts clean.
- `MAX_CORES` (default 60). The ceiling on worker processes.

The notebook can also be converted to a script with `jupyter nbconvert --to script` and run with
`ipython`; every executing cell is guarded by `if __name__ == "__main__":` for that purpose.

### Outputs

Each study writes to `results/equality_study/<condition>/<run_tag>/`, where `<run_tag>` is `full`
or `smoke`:

- `equality_study_config.json`: the parameters and master seed the study ran with.
- `variant_summary.csv`, with one file per network as `variant_summary_<network>.csv`: one row per
  variant, holding its network statistics (degree Gini coefficient, clustering coefficient, average
  degree, …) and the mean and standard deviation of each outcome over its runs.
- Diagnostics: `parameter_coverage.csv`, `variance_check.csv` (confirms that the replicate seeds of
  each variant are distinct and that outcomes vary across them), `progress_report.csv`,
  `runtime_projection.csv`, `cost_all_arms.csv`, `failed_arms.json`, and one scatter plot per
  network.

The outcomes recorded for each run are the share of agents holding the correct theory when the run
stops, the step at which it stops, and the proportion of agents reached by the truth.

The per-run simulation shards are not committed; the summaries are. The last cell of notebook 2
stacks the four `full` summaries into `results/equality_study/variant_summary_combined.csv`
(12,000 rows), which is the file notebook 3 reads.

## Statistical analysis and figures

Notebook 3 fits one ordinary least squares model per network and difficulty condition (12 models).
The dependent variable is reliability, measured as a variant's mean share of correct agents when
its runs stop. The predictors are the variant's degree Gini coefficient and its approximate average
clustering coefficient, the latter as a control. For each model the notebook reports standardized
coefficients, Cohen's *f*² effect sizes, and collinearity checks (pairwise Pearson correlations and
variance inflation factors).

The combined LaTeX table reports all 12 models. Its coefficient stars use Bonferroni-adjusted
*p*-values across the 24 coefficient tests in the table (*p*<sub>adj</sub> = min(1, 24*p*)); the
*F*-statistic stars are unadjusted.

Figures are written to `results/figures/`:

| File | Written by | Content |
|:---|:---|:---|
| `combined_conditions_regression_facets.{png,svg}` | Notebook 3 | One panel per network; variant means and fitted lines for the four conditions, with pointwise 95% confidence intervals for the fitted mean, on a shared reliability scale |
| `combined_conditions_regression_facets_zoomed.{png,svg}` | Notebook 3 | The same fits with a separate, progressively tighter vertical scale per panel |
| `tobacco_network_comparison.png` | Notebook 4 | The tobacco network beside an equalized variant |

Running the notebooks also produces files that are not committed: notebook 3 saves grayscale,
protanopia and deuteranopia previews of each regression figure, and notebook 4 saves
`simple_and_pud_networks` and a PDF version of each of its figures.

## Validation

### Unit tests

From the project root:

```bash
.venv/bin/python -m pytest unit_tests -q
```

| File | Checks |
|:---|:---|
| `test_agents.py` | The bandit and the Beta agent of the object-oriented baseline |
| `test_vectorization.py` | The vectorized model against the object-oriented baseline |
| `test_stopping_conditions.py` | The stopping modes of `VectorizedModel` |
| `test_equality_study_aggregation.py` | Aggregation over a partially completed study |
| `test_equality_study_variant_cache.py` | That caching built variants does not change the study |
| `test_repo_integrity.py` | The repository itself: every module imports from the project root, each citation network unpickles at its published size, and the notebooks are valid JSON |

Two end-to-end tests that run real simulations are marked `slow`; skip them with
`pytest unit_tests -m "not slow"`.

`unittest discover -s unit_tests` also works, but collects only the `unittest`-style files. The two
`test_equality_study_*.py` files and `test_repo_integrity.py` use pytest fixtures and markers.

### Replication of Zollman (2007, 2010)

`unit_tests/zollman/reproducing_zollman.ipynb` runs the vectorized model on cycle, wheel and
complete networks of 5–10 agents, with 1,000 simulations per network shape and size, and compares
the results with the published ones. The two CSV files beside it hold the run-level results.

- **Zollman (2007)** uses `agent_type="bayes"`. The qualitative result is reproduced: the complete
  network is the least reliable and the cycle the most.
- **Zollman (2010)** uses `agent_type="beta"` with choice-stability stopping and a window of 1,000
  steps, because the default tolerance stop fires within a few steps for these agents.

## Repository layout

```
e_network_inequality/
│
├── 1. Citation Data and Networks Generation.ipynb   # Stage 1: fetch OpenAlex data, build networks
├── 2. Simulations.ipynb                             # Stage 2: the equality study, four conditions
├── 3. Results Data Analysis.ipynb                   # Stage 3: regressions, table and figures
├── 4. Network-Visualizations.ipynb                  # Stage 4: network figures
├── network_viz_to_crib.ipynb                        # Auxiliary talk figures; see note below
│
├── model/
│   ├── agents.py, model.py, simulation_functions.py   # Object-oriented baseline (kept unchanged)
│   ├── bandit.py, vectorized_model.py                 # Primary vectorized engine
│   ├── vectorized_simulation_functions.py             # One parameter dictionary -> one model run
│   └── equality_study.py                              # The paper's study: variants, runs, aggregation
│
├── networks/
│   ├── variation_methods.py        # The network variation method (paper §5)
│   └── citation_data/              # *_works.pkl (raw OpenAlex) + *_network.pkl (derived)
│
├── utils/
│   ├── imports.py                  # Central re-export hub for external libraries
│   ├── network_utils.py            # Network statistics (degree Gini, clustering, ...)
│   ├── equality_plots.py           # Figure rendering for notebook 3
│   └── network_plot_utils.py       # Bundled citation-graph plots; see note below
│
├── unit_tests/                     # Test suite
│   └── zollman/                    # Replication of Zollman (2007, 2010)
│
├── results/
│   ├── equality_study/             # <condition>/<run_tag>/ outputs + variant_summary_combined.csv
│   └── figures/                    # Figures written by notebooks 3 and 4
│
├── README.md
├── CITATION.cff, LICENSE, .env.example
└── pyproject.toml, uv.lock, requirements.txt, .python-version
```

**Files outside the reproduction pipeline.** `network_viz_to_crib.ipynb` draws network figures for
a talk. It reads inputs that are not part of this repository and cannot be run from a fresh clone;
nothing in the paper depends on it. It is also the only user of `utils/network_plot_utils.py`, which
imports `NetworkInequality.edgebundling`, a package that is on no package index. The `viz` extra in
`pyproject.toml` (`uv sync --extra viz`) installs the other dependencies of these two files and is
not needed for notebooks 1–4.

## Conventions

- **Unchanged baseline.** `model/agents.py`, `model/model.py` and `model/simulation_functions.py`
  are the reference that `test_vectorization.py` checks `VectorizedModel` against. Extend the model
  by subclassing or adding files, not by editing these.
- **Imports** are absolute from the project root (`from model.vectorized_model import ...`), except
  for relative imports within `model/`. Notebooks add the project root to `sys.path` at startup.
- **Dependencies.** `pyproject.toml` and `uv.lock` are canonical. `requirements.txt` is generated
  from them with
  `uv export --format requirements-txt --all-extras --no-hashes --no-emit-project`.
- **No continuous integration.** This is research code. The checks are the unit tests and a
  Restart-and-Run-All over the notebooks.

## Citation, license and data sources

If you use this code or the derived networks, please cite the paper. Machine-readable metadata is in
[CITATION.cff](CITATION.cff).

> Duijf, H., Noichl, M., & Ojea Quintana, I. (2026). *Inequality and the Reliability of Science*.
> Unpublished manuscript.

```bibtex
@unpublished{inequality_reliability_2026,
  author = {Duijf, Hein and Noichl, Max and Ojea Quintana, Ignacio},
  title  = {Inequality and the Reliability of Science},
  year   = {2026},
  note   = {Unpublished manuscript}
}
```

The code is released under the [MIT License](LICENSE), © 2026 Hein Duijf, Max Noichl and
Ignacio Ojea Quintana. The bibliometric data comes from [OpenAlex](https://openalex.org/), which
releases its data under CC0.

An archived snapshot with a DOI has not yet been minted.

## References

- Zollman, K. J. S. (2007). The communication structure of epistemic communities. *Philosophy of
  Science*, 74(5), 574–587.
- Zollman, K. J. S. (2010). The epistemic benefit of transient diversity. *Erkenntnis*, 72(1),
  17–35.
- Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works,
  authors, venues, institutions, and concepts. *arXiv:2205.01833*.
