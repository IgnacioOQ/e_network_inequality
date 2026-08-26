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
| `pud_regression_plot.png` | `3. Results Data Analysis.ipynb` | §6.2 regression results (PUD) | Yes |
| `tobacco_regression_plot.png` | `3. Results Data Analysis.ipynb` | §6.2 regression results (tobacco) | Yes |
| `ego_regression_plot.png` | `3. Results Data Analysis.ipynb` | §6.2 regression results (ego depletion) | Yes |

**Filenames in notebook 3 are constructed, not literal.** Figures are written through a
`savefig(fig_path, ...)` helper where `fig_path = Path(paper_fig_dir) / filename`, so grepping
the notebook for a `.png` name finds nothing. This table is the navigation path from paper to code.
Notebook 3 also writes to `results/` only when `RUNNING_LOCALLY` is true; on Colab it targets
`STUDY_ROOT / 'figures'`.

## Data

| File | Source | Role |
|:---|:---|:---|
| `option1_literature_summary.csv` | `2a. GColab Simulations Equality - Literature.ipynb` → aggregated by `2d.` | §6 literature-parameter condition; input to notebook 3 |
| `option3_phase_transition_summary.csv` | `2c. GColab Simulations Equality - Phase Transition.ipynb` → aggregated by `2d.` | §6 phase-transition condition; input to notebook 3 |
| `zollman_2007.csv` | transcribed from Zollman (2007) | replication baseline referenced in §2.2 |

These CSVs are committed deliberately: the runs behind them are multi-day Colab jobs, and the
CSVs are the evidentiary basis for the published regression figures. Intermediate `.pkl` / `.npy`
simulation state is not committed — regenerate it.

The `2b. … Harder.ipynb` condition has no summary CSV committed; its aggregate was not part of the
figures in the current draft.

## Gaps

The four black-and-white network figures were produced outside this repository and their
generating code was not recovered (the paper's own footnote in §1 records this). They are
committed as artifacts, not as reproducible outputs. `4. Network-Visualizations.ipynb` produces
*related* colour/PDF network visualizations (`basic_network_types.pdf`,
`degree_distributions_all_networks_loglog.pdf`, `pud_network_original.pdf`,
`pud_network_variants.pdf`) and writes them to a `NetworkInequality/Figures/` path that is external
to this repository — that notebook is the closest available starting point for regenerating them.
