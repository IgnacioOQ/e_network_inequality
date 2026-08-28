# Publication Checklist

This repository is the public companion code for *Inequality and the Reliability of Science*
(Duijf, Noichl & Ojea Quintana). It was reduced to its paper-essential contents on **2026-08-26**.
The full pre-cleanup state is preserved on the annotated tag **`pre-cleanup-2026-08-26`**
(pushed to `origin`); recover anything from there rather than reconstructing it.

## The publication boundary

| Excluded | Why | Reversible? |
|:---|:---|:---|
| `*.pdf`, `manuscript/`, `slides/` | Publisher owns the accepted and published versions; drafts are correspondence, not the scientific record | Link the DOI or a preprint instead |
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
  `unit_tests/` remain and are the maintained correctness check.
- `networks/archive/` — legacy, unreferenced by any study in the paper.
  (`networks/citation_data/tobacco_extended_*` was removed here and later restored: it follows the
  same `*_works.pkl` / `*_network.pkl` shape as the three published episodes.)
- `model/convergence_analysis/**/*.ipynb` and the duplicate `.pdf` brief — the exploratory Colab
  notebooks behind the stopping condition. (The rest of that folder was untracked on 2026-08-28;
  see below.)
- `setup.py` — superseded by `pyproject.toml`.
- `3. Results Data Analysis.ipynb` cells 58–171 — the author-labelled `# Older stuff` /
  `# IGNORE THE REST` / `# Archive` tail: 114 cells and 6.9 MB of stale outputs, including
  redefinitions of functions the paper cells already define and a read from a `data/` directory
  that is not in the repository.
- `4. Network-Visualizations.ipynb` — one code cell containing a natural-language prompt
  (`make all of these only consider outdegrees…`) that had been executed and committed as a
  `SyntaxError`, plus committed tracebacks and pip-install logs. That notebook still needs a
  substantive pass; see the `NetworkInequality` item below.
- Absolute home-directory paths in committed cell outputs (three collaborators' machines) were
  replaced with `/Users/<user>`.

## Repository history — a settled decision, not an oversight

**The raw OpenAlex dumps remain in git history, and that is deliberate (decided 2026-08-26).**
`networks/citation_data/*_works.pkl` were untracked, so they appear in no checkout of any commit
from `d00ee0b` onward. They are still reachable in history, which is why GitHub reports the
repository at ~586 MB while a checkout is ~20 MB.

Purging them would mean `git filter-repo` plus force-replacing the remote. That was considered and
**declined**, for two reasons:

1. It rewrites *every* ref. The repository's 24 other branches would have to go with it — they
   point at the old objects and would otherwise keep the data reachable. Those branches were
   explicitly left alone.
2. Both co-authors are actively pushing and would each have to re-clone.

Do not "clean this up" without raising it first. The cost of the rewrite is coordination, not
disk, and the current state is correct for every practical purpose: nobody who clones gets the
150 MB in their working tree.

### Raw data is tracked on purpose

`networks/citation_data/*_works.pkl` (~150 MB) are **tracked**. They were untracked on 2026-08-26
and re-tracked the same day, once two things became clear:

1. **They are not regenerable.** OpenAlex is a living database. Re-running notebook 1 today returns
   a different set of records and does not reproduce the April-2026 networks the paper reports. The
   dumps are the only surviving snapshot of the inputs behind the published results — precisely the
   artifact a reproducibility reviewer asks for.
2. **Untracking them deletes collaborators' copies.** Git applies the untracking commit as a
   deletion, removing the file from the working tree of everyone who pulls. Verified against a
   throwaway clone: a byte-identical copy is deleted silently; a locally modified one makes git
   refuse the checkout. No `.gitignore` entry prevents either outcome. One local working copy was
   destroyed this way before it was caught.

Re-tracking cost nothing in repository size: the blobs were already in history (that purge was
declined, see above), so re-adding them introduced no new objects. The only difference is ~150 MB
in the working tree.

**Do not untrack these again** without a plan for both points.

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
- [ ] **Decide whether options 2 and 3 need per-network regression figures.** Their `filename=`
      arguments are commented out (cells 40–42, 54–56), so only option 1 writes
      `pud_`/`tobacco_`/`ego_regression_plot.png`; options 2 and 3 appear only as grid figures. If
      you re-enable them, give them condition-specific filenames — all three blocks currently target
      the same three names.
- [ ] **Run the Restart-and-Run-All sweep** over the surviving notebooks, confirming they execute
      against a clean environment built by `uv sync`.
- [ ] **Fix the stale hard-coded paths in `4. Network-Visualizations.ipynb`.** It loads
      `NetworkInequality/empirical_networks/ego_network.pkl` and writes to
      `NetworkInequality/Figures/`, neither of which exists in the repository layout. Part of the
      same pass as the `NetworkInequality` dependency above.

## The 2026-08-28 pass

- **`model/convergence_analysis/` untracked and gitignored.** The stopping-condition groundwork is
  kept locally but is not part of the published companion code. It remains reachable in history up
  to `d8b71d3`; it is *not* deleted from anyone's disk by this repository, but pulling the
  untracking commit will remove it from a collaborator's working tree — the same mechanism
  documented for the raw dumps above. Tell co-authors to back the folder up before they pull.
- **`networks/citation_data/` reduced to the `*_works.pkl` / `*_network.pkl` pairs.** Removed:
  `perceptron_final.{json,pkl}`, `perceptron_final_dill.pkl`, `pud_alternative.pkl`,
  `pud_final.{json,pkl}`. All were written only by archive cells of notebook 1 or read only by
  `model/convergence_analysis/`; no published study loads them.
- **Paper figures moved to `results/figures/`.** `FIGURE_DIR` in `3. Results Data Analysis.ipynb`
  and the paths in `results/MANIFEST.md` were updated to match.
- **Dependencies consolidated on uv.** `pyproject.toml` is the source of truth (with a `viz` extra
  for notebook 4's stack); `requirements.txt` is now generated by
  `uv export --format requirements-txt --all-extras --no-hashes --no-emit-project`. The previous
  gaps — `pyalex`, `python-dotenv`, `pytz`, `ipython`, `powerlaw`, `graphistry`, `colormaps` — are
  covered, and the incorrect `dotenv` package name was replaced with `python-dotenv`.
- **Test suite moved to `unit_tests/` at the root**, and the `testing/` wrapper deleted. The two
  `sys.path` hacks in `test_stopping_conditions.py` and `test_equality_study_aggregation.py`
  counted directory levels and were adjusted for the shallower path.
- **`HOUSEKEEPING.md` deleted.** The routine sanity-check workflow was internal process material,
  not something a reader of the paper needs. Its four checks were not lost with it: they are now
  `unit_tests/test_repo_integrity.py`, so they run with the suite instead of waiting for someone to
  remember a checklist. Recover the document from history if the wider workflow is wanted back.
- **README pruned** and the `status:` / `type:` / `description:` metadata headers stripped from the
  tracked markdown documents.

## Not needed

No CI, no merge gates, no pre-commit hooks, no linters. The honest substitute for a test suite is
Restart-and-Run-All plus the unit tests in `unit_tests/`.

The repository also carries no routine-maintenance workflow document. `HOUSEKEEPING.md` was deleted
on 2026-08-28 as internal process material: it described how the authors keep the repository tidy,
which is not something a reader reproducing the paper needs. Its four ad-hoc checks — core-module
imports, empirical-network loadability, notebook JSON validity, and a `VectorizedModel` snapshot
smoke test — were converted into `unit_tests/test_repo_integrity.py` rather than dropped.
