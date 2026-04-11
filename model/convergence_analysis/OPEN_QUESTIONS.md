# Open Questions and Conceptual Tensions
- status: active
- type: research
- owner: user
- last_checked: 2026-04-11
<!-- content -->

> **Scope reminder:** work investigating these questions must not modify any file outside `model/convergence_analysis/`. Scripts and outputs (plots, datasets) belong in this folder. See `MC_AGENT.md` for the full constraints.

This document records open conceptual tensions and future research directions arising from the network epistemology simulation and its Markov Chain analysis. It is a living document — add new entries as questions emerge.

## 1. The Lower-Bound Anomaly
- status: active
<!-- content -->

**Observation:** At $10^6$ steps on the PUD network, the actual truth share (0.7724) slightly *exceeds* the root-reachability prediction (0.7628), yielding a positive gap of +0.0096.

**Tension:** The Root Node Hypothesis treats root-reachability as an upper bound on truth convergence. A positive gap violates this framing.

**Candidate mechanisms:**
- Self-reinforcing credence cycles in weakly-connected subgraphs where no root is truthful, but agents collectively accumulate enough evidence.
- Agents with favorable initial priors (high $\alpha_1$, low $\beta_1$) who happen to test Theory 1 disproportionately early and lock in before observing any neighbor.
- Isolated components (after pruning) where the bandit problem alone drives convergence, independent of the network structure.

**Open question:** Is this anomaly structural (a consequence of the PUD network topology) or generic? Does it persist across synthetic networks (ER, BA, WS)?

---

## 2. Stability vs. Convergence Speed
- status: active
<!-- content -->

**Observation:** The chain is guaranteed to absorb ($\alpha$, $\beta$ only increase), but the lag phase is large: the predicted-vs-actual gap is still $-28\%$ at 1,000 steps on a 312-node network.

**Tension:** Guaranteed absorption does not imply fast convergence. The mixing time appears to scale poorly with network size or depth.

**Open questions:**
- Does mixing time scale with graph diameter? Mean path length from root to leaves?
- Is the bottleneck the number of "relay hops" a signal must traverse from a root to a peripheral node?
- Can we bound mixing time analytically as a function of $\varepsilon$, $n_{\text{experiments}}$, and network depth?

**Implementation gap:** `ConvergenceDiagnostics.estimated_mixing_time` in `utils/mc_analysis.py` is computed empirically via parallel chains but not connected to any graph-structural predictor.

---

## 3. Two-Phase Dynamics: Exploration → Exploitation
- status: active
<!-- content -->

**Observation:** The $\varepsilon$-greedy rule creates a structural transition from a high-variance exploration phase (early steps, agents still testing both arms) to a low-variance exploitation phase (late steps, agents locked in). The lag in root-to-descendant belief propagation appears concentrated in the early phase.

**Tension:** The Markov Chain framing treats all steps identically, but the *effective* transition kernel changes dramatically between phases.

**Open questions:**
- Can we characterize the transition step $t^*$ analytically (e.g., when the expected credence change drops below a threshold)?
- Does $t^*$ depend primarily on $\varepsilon$, prior strength $(\alpha_0, \beta_0)$, or the bandit gap $\Delta = p_1 - p_0$?
- Is there a two-timescale decomposition: fast dynamics (within-agent belief updates) and slow dynamics (network-level consensus propagation)?

---

## 4. Absorption Probability Factorization
- status: active
<!-- content -->

**Informal claim:**
$$P(\text{network absorbs to truth}) \approx P(\text{roots reach truth}) \times P(\text{signal propagates} \mid \text{roots reach truth})$$

The second factor appears to converge to 1 empirically (gap $\to 0$ as $t \to \infty$), but no formal proof exists.

**Open questions:**
- Under what conditions on $\varepsilon$, prior strength, and network topology does the second factor equal 1 exactly?
- Does the factorization break down for graphs with cycles (where "root" is not well-defined)?
- Can the absorption probability be computed analytically for small networks?

**Related implementation:** `MarkovChainAnalyzer.estimate_absorption_probabilities()` in `utils/mc_analysis.py` estimates this via Monte Carlo but does not decompose it into the two factors.

---

## 5. Formal Stability Conditions
- status: active
<!-- content -->

**Question:** What is the minimum exploration rate $\varepsilon$ that guarantees a root node converges to truth with probability 1?

**Context:** A root node is a standalone bandit problem. Classical results (e.g., Lai & Robbins 1985) give asymptotic regret bounds for $\varepsilon$-greedy, but the specific condition for *almost-sure* convergence to truth in this Beta-updating setup is not pinned down.

**Conjecture:** Any $\varepsilon > 0$ suffices for almost-sure convergence of an isolated agent, since the Beta posteriors will eventually concentrate around the true parameter. But the *rate* depends on $\varepsilon$.

**Open questions:**
- Does this generalize to non-root agents in cycles? A cycle with two agents could in principle lock each other into the wrong belief.
- Is there a phase transition in $\varepsilon$ below which consensus to falsehood becomes likely?

---

## 6. Network Topology Effects
- status: active
<!-- content -->

**Context:** The PUD network is a DAG (after pruning twins and extracting the LCC). For networks with cycles, root influence no longer cleanly determines outcomes.

**Open questions:**
- How does the Root Node Hypothesis degrade for Erdős-Rényi, Watts-Strogatz, or Barabási-Albert graphs with cycles?
- Is there a natural generalization of "root influence" for cyclic graphs (e.g., nodes with low in-degree, or high betweenness centrality)?
- Does the presence of hubs (high out-degree nodes in BA graphs) create a "broadcast" dynamic that accelerates or impedes consensus?

**Related work:** `model/convergence_analysis/Colab_Root_Influence_Analysis.ipynb` tests the hypothesis on ER, WS, and BA synthetic networks. Results have not been fully analyzed against these questions.

---

## 7. Spectral Gap as Mixing-Time Proxy
- status: active
<!-- content -->

**Context:** `utils/mc_analysis.py` defines `ConvergenceDiagnostics.estimated_spectral_gap` but never populates it. `model/convergence_analysis/Colab_Formal_Analysis.ipynb` performs spectral analysis of the listening matrix but does not connect it to empirical mixing time.

**Hypothesis:** The spectral gap of the row-normalized adjacency matrix (listening matrix $L$) should predict mixing time: a larger gap implies faster convergence.

**Open questions:**
- Does the second eigenvalue of $L$ predict the empirical mixing time measured in `estimate_mixing_time()`?
- For DAGs, what is the appropriate spectral object (the matrix is not symmetric and may not be diagonalizable)?
- Can we use the spectral structure to identify "bottleneck" edges whose removal would most slow convergence?

**Implementation gap:** Connect `Colab_Formal_Analysis.ipynb`'s spectral computation to `MarkovChainAnalyzer.compute_convergence_diagnostics()` and populate `estimated_spectral_gap`.
