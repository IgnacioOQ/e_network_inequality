# Results Manifest

- status: active
- type: reference
- description: Maps every committed artifact in `results/` to the code that produced it and the place it appears in the paper. Gaps are recorded explicitly rather than glossed over.

Paper: *Inequality and the Reliability of Science* (Duijf, Noichl & Ojea Quintana).
Because the manuscript tree is outside this repository (see [PUBLICATION_CHECKLIST.md](../PUBLICATION_CHECKLIST.md)),
this file is the only surviving link between the published figures and the code.

## Figures

| File | Source | Paper location | Reproducible? |
|:---|:---|:---|:---|
| `simple_and_pud_networks.png` | not in repo — see *Gaps* | Figure 1 (idealized networks vs. PUD network) | **No** |
| `pud_network_black.png` | not in repo — see *Gaps* | Figure 1 panel / §2.1 | **No** |
| `tobacco_network_black.png` | not in repo — see *Gaps* | §4 network illustrations | **No** |
| `tobacco_variant_network_black.png` | not in repo — see *Gaps* | §5 network variants | **No** |
| `pud_regression_plot.png` | `3. Results Data Analysis.ipynb` | §6.2 regression results (PUD) | Yes, but see *filename collision* |
| `tobacco_regression_plot.png` | `3. Results Data Analysis.ipynb` | §6.2 regression results (tobacco) | Yes, but see *filename collision* |
| `ego_regression_plot.png` | `3. Results Data Analysis.ipynb` | §6.2 regression results (ego depletion) | Yes, but see *filename collision* |
| `hard_problems_regression_grid.png` | `3. Results Data Analysis.ipynb`, cell 28 | §6.2 overview, option 1 (literature parameters) | Yes |
| `super_hard_problems_regression_grid.png` | `3. Results Data Analysis.ipynb`, cell 43 | §6.2 overview, option 2 (harder) | Yes |
| `varying_easiness_regression_grid.png` | `3. Results Data Analysis.ipynb`, cell 57 | §6.2 overview, option 3 (phase transition) | Yes |

**Filenames in notebook 3 are constructed, not literal.** Figures are written through a
`savefig(fig_path, ...)` helper where `fig_path = Path(paper_fig_dir) / filename`, so grepping
the notebook for a `.png` name finds nothing — the name appears only as a `filename="..."` argument.
This table is the navigation path from paper to code. Notebook 3 also writes to `results/` only
when `RUNNING_LOCALLY` is true; on Colab it targets `STUDY_ROOT / 'figures'`.

**⚠ Filename collision across the three conditions.** The notebook runs the same three per-network
regression plots once per experimental condition, each time under the *same* filename:

| Cells | Condition | Per-network figures written | Grid figure written |
|:---|:---|:---|:---|
| 25–28 | option 1 — literature | `pud_` / `tobacco_` / `ego_regression_plot.png` | `hard_problems_regression_grid.png` |
| 40–43 | option 2 — harder | *the same three filenames, overwritten* | `super_hard_problems_regression_grid.png` |
| 54–57 | option 3 — phase transition | *the same three filenames, overwritten again* | `varying_easiness_regression_grid.png` |

A full top-to-bottom run therefore leaves `results/*_regression_plot.png` holding **option 3**
output, and the option 1 and option 2 versions of those figures are destroyed. The three grid
figures are unaffected — each has a distinct name.

**Which condition the committed `*_regression_plot.png` files represent is not determinable from
the repository.** They were last written on 2026-08-26 12:27, before the grid figures (17:26), so
they are not from the same run. Anyone using them in the paper should re-run the intended condition
and confirm, or give the three per-condition variants distinct filenames
(e.g. `pud_regression_plot__option1.png`) so that a single run produces all nine.

## Data

| File | Source | Role |
|:---|:---|:---|
| `option1_literature_summary.csv` | `2a. GColab Simulations Equality - Literature.ipynb` → aggregated by `2d.` | §6 literature-parameter condition; loaded by notebook 3 cell 16 via `load_summary()` |
| `option2_harder_summary.csv` | `2b. GColab Simulations Equality - Harder.ipynb` → aggregated by `2d.` | §6 harder condition; loaded by notebook 3 cell 33 |
| `option3_phase_transition_summary.csv` | `2c. GColab Simulations Equality - Phase Transition.ipynb` → aggregated by `2d.` | §6 phase-transition condition; loaded by notebook 3 cell 47 via `load_summary()` |
| `zollman_2007.csv` | transcribed from Zollman (2007) | replication baseline referenced in §2.2 |

These CSVs are committed deliberately: the runs behind them are multi-day Colab jobs, and the
CSVs are the evidentiary basis for the published regression figures. Intermediate `.pkl` / `.npy`
simulation state is not committed — regenerate it.

All three conditions now have a committed aggregate; each holds the same 19-column contract that
notebook 2d writes and notebook 3 reads. One inconsistency worth tidying: options 1 and 3 are read
through the `load_summary()` helper (which falls back across Drive and the repo's `results/`), while
option 2 is read with a hard-coded `pd.read_csv("results/option2_harder_summary.csv")` — so the
option 2 cell works locally but not on Colab.

## Gaps

The four black-and-white network figures were produced outside this repository and their
generating code was not recovered (the paper's own footnote in §1 records this). They are
committed as artifacts, not as reproducible outputs. `4. Network-Visualizations.ipynb` produces
*related* colour/PDF network visualizations (`basic_network_types.pdf`,
`degree_distributions_all_networks_loglog.pdf`, `pud_network_original.pdf`,
`pud_network_variants.pdf`) and writes them to a `NetworkInequality/Figures/` path that is external
to this repository — that notebook is the closest available starting point for regenerating them.
