# Publication Checklist

- status: active
- type: reference
- description: What is deliberately excluded from this public repository and why, plus the steps remaining before the paper is submitted. Read this before "restoring" anything that looks missing.
- last_checked: 2026-08-26

This repository is the public companion code for *Inequality and the Reliability of Science*
(Duijf, Noichl & Ojea Quintana). It was reduced to its paper-essential contents on **2026-08-26**.
The full pre-cleanup state is preserved on the annotated tag **`pre-cleanup-2026-08-26`**
(pushed to `origin`); recover anything from there rather than reconstructing it.

## The publication boundary

| Excluded | Why | Reversible? |
|:---|:---|:---|
| `*.pdf`, `manuscript/`, `slides/` | Publisher owns the accepted and published versions; drafts are correspondence, not the scientific record | Link the DOI or a preprint instead |
| `networks/citation_data/*_works.pkl` (~150 MB) | Raw OpenAlex API responses; regenerated end-to-end by `1. Citation Data and Networks Generation.ipynb`. The derived networks are tracked instead | Yes — rerun notebook 1 |
| `.claude/`, `.vscode/`, `AI_AGENTS/`, `docs/` | Agent and editor config, local absolute paths, internal workflow notes | n/a |
| `.env` | OpenAlex polite-pool email / API credentials. `.env.example` shows the shape | n/a |

**These gaps are intentional.** A later reader (or agent) that reads them as an oversight and
helpfully re-adds the excluded material undoes the boundary.

## What was removed in the 2026-08-26 cleanup

- `AI_AGENTS/`, `docs/`, `scripts/archive/` — agent scaffolding and internal process notes.
- Playground and superseded notebooks: `playground_hein.ipynb`,
  `A. GColab Simulations Playground.ipynb`, `2. GColab Simulations.ipynb`,
  `2. GColab Simulations Equality.ipynb`, `3. Local Simulations SA.ipynb`,
  `results/playground_Hein.ipynb`, `results/simulation_analysis.ipynb`.
- `testing/notebooks/` — exploratory model-checking notebooks. The unit tests in
  `testing/unit_tests/` remain and are the maintained correctness check.
- `networks/archive/`, `networks/citation_data/tobacco_extended_*`,
  `perceptron_final_dill.pkl` — legacy or unreferenced by any study in the paper.
- `model/convergence_analysis/**/*.ipynb` and the duplicate `.pdf` brief — the exploratory Colab
  notebooks behind the stopping condition. The `.py` modules and the written analyses
  (`STOPPING_CONDITION_ANALYSIS.md`, the hypothesis documents) were kept: they are the
  justification for the stability window of 100 and the horizon of 100,000 used in §6.1.
- `setup.py` — superseded by `pyproject.toml`.

## Remaining steps before submission

- [ ] **Add a data-and-code-availability statement to the paper.** The current draft never cites
      this repository. One paragraph naming the repo URL, the archived DOI, and the OpenAlex data
      source is what most journals and reproducibility guidelines now expect.
- [ ] **Archive a tagged release to Zenodo** to mint a DOI, then add the DOI badge to `README.md`
      and the `doi:` and `identifiers:` fields to `CITATION.cff`. A GitHub tag alone is mutable;
      a DOI is the persistent identifier the citation apparatus indexes.
- [ ] **Fill in the ORCIDs** in `CITATION.cff` (three commented placeholders).
- [ ] **Set the GitHub repository description and topics** — both are currently empty.
- [ ] **Confirm co-author agreement on MIT licensing.** `LICENSE` names all three authors in the
      copyright line.
- [ ] **Resolve the `NetworkInequality` dependency.** `utils/network_plot_utils.py` imports
      `NetworkInequality.edgebundling`, which is not in this repository and not on any package
      index, so notebook 4 cannot be run by a reader. Vendor the helper in or drop the import.
      (`utils/sa_network_variation_directed.py`, which had the same problem and no remaining
      consumer, was removed in the cleanup.)
- [ ] **Run the Restart-and-Run-All sweep** over the surviving notebooks (see `HOUSEKEEPING.md`),
      confirming they execute against a clean `.venv` from `requirements.txt`.
- [ ] **Decide on the `2b. … Harder` condition** — its summary CSV is not committed and it is not
      referenced by the current figures. Either commit the aggregate or note it as exploratory.

## Not needed

Per the academic-repo standard this repository follows: no CI, no merge gates, no pre-commit
hooks, no linters. The honest substitute for a test suite is Restart-and-Run-All plus the unit
tests in `testing/unit_tests/`.
