# OBSERVATIONS.md
- status: active
- last_checked: 2026-04-15
<!-- content -->
Technical findings, design decisions, and open concerns organized by topic.
See [WORKPLAN.md](WORKPLAN.md) for the task list.

---

## Variation Methods
- status: active
<!-- content -->
**File:** [`networks/variation_methods.py`](networks/variation_methods.py)

Builds a variant of an input network by adding/rewiring `n_edges` edges while preserving the
**degree distribution** and **clustering coefficient**. Three phases:

1. **Setup** — Optionally remove the same number of edges as will be added (fixed density). Set
   target degree distribution: original for densify, uniform for equalization.
2. **Main loop** — Add edges one at a time. Either (a) add a triangle-completing edge if
   clustering fell below target (interim clustering branch), or (b) sample an edge toward the
   target degree distribution (degree branch).
3. **Post clustering** — Degree-preserving edge swaps to close any remaining clustering gap.

Both the interim clustering branch and post clustering loop are optional.

### Per-network Status

| Network | Densify | Equalize | Notes |
|:---|:---|:---|:---|
| PUD | ✅ | ✅ | Both interim and post clustering work |
| Tobacco | ✅ | ✅ | Densify: both; Equalize: post only (interim → corr. 0.21; post → −0.025) |
| Ego depletion | ✅ | ⏯️ | Densify: post only; Equalize in progress |

Tobacco equalize Gini ranges: interim \[0.62—0.76\], post \[0.55—0.76\].

### Degree Branch: Independent vs Conditional

Controlled by `p_conditional` ∈ [0, 1]:

- **Independent** (`1 − p_conditional`): Draw sources and targets independently, filter
  self-loops and duplicates, pick a surviving pair. Simple but builds up a rejection bias.
- **Conditional** (`p_conditional`): Draw one source, exclude its existing connections, draw a
  target from the remainder. First attempt almost always succeeds.

They produce **opposite correlations with n_edges** (both use static original-degree weights):

- Independent → *negative* correlation: as edges fill up, hub-to-hub proposals get rejected
  more, surviving edges skew toward peripheral nodes, flattening the distribution (lower Gini).
- Conditional → *positive* correlation: only the drawn source's slots are excluded, other hubs
  remain available, preferential attachment runs unimpeded (higher Gini).

### Clustering Loops

**Interim branch:** When clustering dips below target, pick a random node and add an edge between
two of its neighbours — guaranteed to complete a triangle. Skipped if no valid pair exists.

**Post loop:** After all edges are added, do degree-preserving swaps accepted only if they move
clustering toward the target. Biased toward triangle edges when clustering is too high, non-triangle
edges when too low.

**Concerns:**
- Interim *de*clustering was tried and gave worse results; not worth revisiting.
- Post loop may distort structure in subtle ways — worth monitoring.
- There may be an intrinsic trade-off: past some point, lowering degree Gini necessarily lowers
  clustering too.

---

## Empirical Networks — Visual Inspection
- status: todo
<!-- content -->
**Files:** [`networks/citation_data/`](networks/citation_data/)

Three citation networks used as simulation topologies (all processed: twins pruned, LCC extracted,
self-loops removed):

- **PUD** — Peptic Ulcer Disease (OpenAlex, 1900–1978)
- **Tobacco** — Tobacco research
- **Ego depletion** — Ego depletion research

Visual inspection of all three is still pending to check for suspicious features before simulations.

---

## Simulation Stopping Conditions
- status: active
<!-- content -->
**File:** [`model/vectorized_model.py`](model/vectorized_model.py)

Current plan (Max + Ignacio): **fixed-step stopping condition**.

- Literature standard: 10,000 steps. Ours converge more slowly; likely need up to 1,000,000.
- Hard requirement: must replicate Zollman (2007) as a correctness check.
- Optional: run a minimum, then check every 10k steps whether anything changed (uncertain feasibility).

Utrecht University cloud services available for large runs.

---

## Paper / Research Framing
- status: active
<!-- content -->
Three concerns raised by FEW reviewers:

1. **Why do results differ from theoretical networks?** — What structural property (degree
   heterogeneity, clustering, diameter, hubs) causes the Zollman effect and equality effect to
   disappear on empirical networks?
2. **What do we learn about dynamics of inquiry?** — The discussion should clearly state the
   practical takeaway about how scientific inquiry proceeds.
3. **"How actually" model** — Clarify what phenomenon the model explains and how empirical
   network topologies advance the mechanistic explanation over theoretical ones.
