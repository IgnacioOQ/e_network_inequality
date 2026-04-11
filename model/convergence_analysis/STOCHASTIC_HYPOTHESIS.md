# Stochastic Truth-Convergence Hypothesis
- status: active
- type: theory
- owner: user
- last_updated: 2026-04-11
<!-- content -->

> **Scope reminder:** scripts testing this hypothesis must not modify any file outside
> `model/convergence_analysis/`. See `MC_AGENT.md` for the full constraints.

---

## 1. The Idea in Natural Language

Before any simulation runs, we know two things about the network: its topology (who listens to whom) and the bandit parameters (how much better Theory 1 is, how many experiments each agent runs, how often they explore). The question is: can we write down, for every node in the network, a probability that it will ultimately converge to the truth?

**The intuition has two parts.**

*Part A — Independent discovery.* Every agent runs its own experiments at every step. Even a node whose ancestors are all committed to the wrong theory will occasionally explore the good arm (Theory 1) because of the epsilon-greedy rule. Over a long simulation, it accumulates evidence for the good theory on its own. The probability that an isolated agent — one with no network connections at all — converges to truth is called the **solo probability** $p^*$. It depends only on the bandit gap, the exploration rate, the prior, and the number of experiments per step, not on the network.

*Part B — Inherited truth.* Truth is contagious in a directed network. If an ancestor of node $i$ has converged to the good theory, it accumulates enormous evidence for Theory 1 — hundreds of thousands of successful trials. Every step, node $i$ observes this ancestor's experiments alongside its own. Even if node $i$ started with a prior favouring the wrong theory, the ancestor's overwhelming evidence will eventually pull $i$'s posterior to the correct side. In the long run, having *any* ancestor that believes the truth is sufficient to guarantee that $i$ believes it too.

Putting the two parts together: node $i$ converges to truth if either (a) it discovers truth independently, or (b) at least one of its ancestors does. These events overlap — if an ancestor also discovered truth independently, both mechanisms fire simultaneously — so we need to count them without double-counting. The result is a system of equations, one per node, that can be solved in topological order (roots first, leaves last) to give each node a probability before the simulation starts.

**The network-level prediction** is then just the average of these per-node probabilities. The variance of the truth share across repeated simulation runs is a consequence of how correlated the nodes are — nodes that share many common ancestors tend to succeed or fail together, increasing the variance relative to what independent nodes would produce.

---

## 2. Formal Setup

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| $G = (V, E)$ | Directed graph; edge $(u, v)$ means $v$ listens to $u$ |
| $\text{Pa}(i)$ | Direct parents of node $i$ (nodes $i$ listens to) |
| $N = \lvert V \rvert$ | Number of nodes |
| $T_i$ | Event: node $i$ converges to truth |
| $P_i = P(T_i)$ | Pre-simulation truth probability of node $i$ |
| $p^*$ | Solo truth probability (isolated agent, no network) |
| $p^r$ | Resistance probability (node whose parents all converged to falsehood) |
| $\Delta$ | Bandit gap: $p_1 - p_0 = \text{uncertainty}$ |
| $\varepsilon$ | Exploration rate |
| $n$ | Experiments per agent per step |

A node $r$ is a **root** if $\text{Pa}(r) = \emptyset$ (in-degree zero).

### 2.2 The Solo Probability $p^*$

An isolated Beta agent (no network, no neighbors) runs as a standalone multi-armed bandit. It converges to truth with probability:

$$p^* = P(\text{isolated agent converges to Theory 1})$$

$p^*$ is not 1. With epsilon-greedy exploration and a finite number of experiments per step, there is a non-trivial probability of accumulating early spurious evidence for Theory 0 and locking in before the good-arm signal dominates. $p^*$ is an increasing function of $\varepsilon$ (more exploration), $\Delta$ (larger gap), and $n$ (more experiments per step).

$p^*$ is not analytically tractable in closed form for Beta-updating epsilon-greedy, but it is empirically estimable: run many isolated-agent simulations and measure the fraction that converge to Theory 1.

### 2.3 The Resistance Probability $p^r$

Suppose all parents of node $i$ have converged to falsehood. They accumulate large alpha values for Theory 0 and large beta values for Theory 1, producing overwhelming evidence *against* the good theory. Node $i$ observes this misleading evidence alongside its own experiments.

In this regime, node $i$ can still reach truth, but only through its own $\varepsilon$-fraction exploratory trials on Theory 1. The probability is:

$$p^r = P(\text{node } i \text{ converges to truth} \mid \text{all parents converge to falsehood})$$

By symmetry of the Beta-updating rule and the bandit structure, $p^r$ does not depend on $i$ (only on $\varepsilon$, $\Delta$, $n$, and the prior). We expect $p^r \ll p^*$: false parents actively suppress the good signal. As the number of parents grows, $p^r \to 0$ (the misleading evidence scales with the number of false parents).

---

## 3. The Topological Recursion

### 3.1 Core Equation

Process nodes in **topological order** (roots first, leaves last). Define $P_i = P(T_i)$.

**Root nodes:**
$$P_i = p^* \quad \text{for all } i \text{ with } \text{Pa}(i) = \emptyset$$

**Non-root nodes:**

Let $A_i = \bigcup_{j \in \text{Pa}(i)} T_j$ be the event that *at least one parent* converges to truth.

Under two assumptions (discussed in Section 3.3):
1. If $A_i$ occurs, node $i$ converges to truth almost surely: $P(T_i \mid A_i) = 1$.
2. If $A_i$ does not occur (all parents false), node $i$ converges to truth with probability $p^r$.

Then:
$$P_i = P(T_i \mid A_i)\,P(A_i) + P(T_i \mid A_i^c)\,P(A_i^c)$$

$$\boxed{P_i = P(A_i) + p^r \cdot (1 - P(A_i))}$$

Under the assumption that parent convergences are *mutually independent* (see Section 3.3):
$$P(A_i) = 1 - \prod_{j \in \text{Pa}(i)} (1 - P_j)$$

Substituting:
$$\boxed{P_i = 1 - (1 - p^r)\prod_{j \in \text{Pa}(i)}(1 - P_j)}$$

This recursion can be evaluated in $O(N + |E|)$ time once $p^*$ and $p^r$ are known.

### 3.2 Log-Space Linearization

Define the **log-falsehood probability**:
$$q_i = \log(1 - P_i) \leq 0$$

The recursion becomes:
$$q_i = \log(1 - p^r) + \sum_{j \in \text{Pa}(i)} q_j$$

Let $c = \log(1 - p^r) < 0$ and $c^* = \log(1 - p^*) < 0$. Then:

- Root node: $q_r = c^*$
- Non-root node: $q_i = c + \sum_{j \in \text{Pa}(i)} q_j$

This is a **linear recursion** over the DAG. Expanding it fully, $q_i$ is a weighted sum over all paths from roots to $i$:

$$q_i = c^* \cdot |\mathcal{R}_i| + c \cdot \text{PathWeight}(i)$$

where $|\mathcal{R}_i|$ is the number of distinct root ancestors of $i$ and $\text{PathWeight}(i)$ counts the total number of (root, $i$)-paths weighted by path length. This quantity is directly computable from powers of the adjacency matrix.

**Implication:** nodes with many distinct root ancestors, or nodes reachable via many long paths, are better predicted by the network topology. The log-space recursion makes the path structure explicit.

### 3.3 Assumptions and Their Validity

**Assumption 1 — Truthful parent sufficiency:** $P(T_i \mid A_i) = 1$. This holds asymptotically: a truthful parent accumulates unbounded evidence for Theory 1 over time, and by the Strong Law of Large Numbers, this eventually overwhelms any prior or peer pressure from false siblings. The assumption fails at finite time (hence the lag phase in the Root Node Hypothesis data).

**Assumption 2 — Parental independence:** The convergences of $j_1, j_2 \in \text{Pa}(i)$ are treated as independent. This is exact if $j_1$ and $j_2$ share no common ancestor. It is violated when multiple parents share common ancestors (positive correlation: shared root good → all converge together). The independence assumption gives a **lower bound** on $P(A_i)$ (it underestimates the probability of at least one parent being truthful when parents are positively correlated). The direction of the bias thus depends on whether the positive correlation from shared ancestry or the negative correlation from shared misleading ancestors dominates; for DAGs with few common-ancestor paths, the approximation is likely good.

---

## 4. Network-Level Predictions

### 4.1 Mean Truth Share

The expected fraction of agents believing truth at convergence:

$$\mu = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \mathbf{1}[T_i]\right] = \frac{1}{N}\sum_{i=1}^N P_i$$

This is exact regardless of correlations — the expectation is linear.

### 4.2 Variance of Truth Share

Let $X = N^{-1}\sum_i \mathbf{1}[T_i]$ be the truth share in a single simulation run.

$$\text{Var}(X) = \frac{1}{N^2} \sum_{i,j} \text{Cov}(\mathbf{1}[T_i], \mathbf{1}[T_j])$$

where $\text{Cov}(\mathbf{1}[T_i], \mathbf{1}[T_j]) = P(T_i \cap T_j) - P_i P_j$.

The joint probability $P(T_i \cap T_j)$ is hard to compute exactly, but has a natural approximation:

$$P(T_i \cap T_j) \approx 1 - (1 - P_i)(1 - P_j) - P(\overline{T_i} \cap \overline{T_j} \mid \text{common ancestors})$$

Nodes sharing many common ancestors have **higher positive covariance** (they tend to succeed or fail together), increasing $\text{Var}(X)$ above the independent-nodes baseline of $N^{-1}\bar{P}(1-\bar{P})$.

The **network correlation matrix** $C_{ij} = P(T_i \cap T_j)$ can be estimated via Monte Carlo simulation (run many replicates and compute empirical joint frequencies). This is the key empirical target.

---

## 5. Connection to Existing Results

### 5.1 Root Node Hypothesis as a Special Case

Set $p^r = 0$ (no resistance). Then $(1 - P_i) = \prod_{j \in \text{Pa}(i)} (1 - P_j)$.

For a node whose only ultimate ancestors are roots, this gives:
- $P_i = 1$ if and only if at least one root ancestor converges to truth.
- $P_i = 0$ if all root ancestors converge to falsehood.

This is exactly the Root Node Hypothesis: the truth share equals the fraction of the network reachable from truthful roots. **The Root Node Hypothesis is the $p^r \to 0$ limit of the Stochastic Hypothesis.**

### 5.2 The Lower-Bound Anomaly (Q1)

In the empirical data, at $10^6$ steps the actual truth share (0.7724) slightly *exceeds* the root-reachability prediction (0.7628). Under the Stochastic Hypothesis, this gap is:

$$\text{Gap} = \frac{1}{N}\sum_{i} p^r \cdot (1 - P(A_i))$$

The sum runs only over nodes whose parent set $A_i^c$ occurs (all parents false). The gap is a direct measurement of $p^r$ weighted by the probability of the resistance regime. The approximately $+1\%$ gap thus gives a rough estimate of $p^r$ for the PUD network.

### 5.3 Two-Phase Dynamics (Q3)

The two-phase dynamics script (`two_phase_dynamics.py`) identified $t^* \approx 550$ as the half-life of active agents. In the Stochastic Hypothesis language:
- **Exploration phase** ($t < t^*$): $p^*$ and $p^r$ are still being "revealed" — agents have not yet committed. The recursion does not yet apply.
- **Exploitation phase** ($t > t^*$): Beliefs have stabilized enough that the recursion's absorbing-state logic applies. The truth-propagation process is approximately over; remaining changes are slow Bayesian accumulation, not structural transitions.

$t^*$ thus marks the onset of the regime in which the Stochastic Hypothesis is predictive.

---

## 6. Spectral Connection (Tentative)

The log-space recursion $q_i = c + \sum_{j \in \text{Pa}(i)} q_j$ is equivalent to a matrix equation:

$$\mathbf{q} = c^* \mathbf{e}_{\text{roots}} + \mathbf{L}^\top \mathbf{q} + c \, \mathbf{1}$$

where $\mathbf{L}$ is the adjacency matrix of the DAG. Since $G$ is a DAG, $\mathbf{L}$ is nilpotent (all eigenvalues zero). The formal solution is:

$$\mathbf{q} = (I - \mathbf{L}^\top)^{-1}(c^* \mathbf{e}_{\text{roots}} + c \, \mathbf{1})$$

This is well-defined because $\mathbf{L}$ is nilpotent, so $(I - \mathbf{L}^\top)^{-1} = \sum_{k=0}^{d} (\mathbf{L}^\top)^k$ where $d$ is the depth of the DAG. The $k$-th power $(\mathbf{L}^\top)^k$ counts paths of length $k$ from parents to descendants.

**Spectral gap and mixing time (Q7):** For networks with cycles, $\mathbf{L}$ is no longer nilpotent. The spectral gap $1 - |\lambda_2(\mathbf{L})|$ of the row-normalized version controls how fast information diffuses. A small spectral gap (close-to-1 second eigenvalue) means slow mixing — the truth signal from a root takes many steps to reach distant nodes. This connects Q7 (spectral gap) to Q2 (mixing time) via the Stochastic Hypothesis: the mixing time controls *when* the recursion's predictions become accurate, not the predictions themselves.

---

## 7. Empirical Test Protocol

To validate the Stochastic Hypothesis:

**Step 1 — Estimate $p^*$.**
Run many simulations of a single isolated Beta agent (no network, no neighbors) for a long time. Measure the fraction that converge to Theory 1. This gives $\hat{p}^*$.

**Step 2 — Estimate $p^r$.**
From simulation runs on the full network, identify nodes whose parents all converged to falsehood. Measure the fraction of such nodes that nevertheless converged to truth. This gives $\hat{p}^r$.

**Step 3 — Compute predicted $P_i$ values.**
Run the topological recursion with $\hat{p}^*$ and $\hat{p}^r$ on the network topology. This produces a predicted probability $\hat{P}_i$ for every node.

**Step 4 — Compare to simulation.**
Run many full-network simulations. For each node, compute the empirical fraction of runs in which it converged to truth. Compare to $\hat{P}_i$. Compute:
- Predicted vs. actual mean truth share ($\hat{\mu}$ vs. $\bar{X}$)
- Predicted vs. actual variance ($\widehat{\text{Var}}(X)$ vs. $\text{Var}_{\text{empirical}}(X)$)
- Node-level calibration: scatter plot of $\hat{P}_i$ vs. empirical $P_i$

**Step 5 — Test network generalization.**
Repeat Steps 3–4 on synthetic networks (Erdős-Rényi, Barabási-Albert, Watts-Strogatz) to test whether $p^*$ and $p^r$ transfer across topologies or are topology-specific.

---

## 8. Open Questions

1. **Is $p^r$ topology-independent?** The resistance probability may depend on the number of false parents (more false parents → less resistance) and the strength of their accumulated evidence (more steps → stronger suppression). If so, $p^r$ is not a single scalar but a function $p^r(k, t)$ where $k$ is the number of false parents.

2. **Does the independence assumption systematically bias the recursion?** Nodes in highly clustered regions (many shared ancestors) will have correlated convergences, violating Assumption 2. Characterize the bias direction and magnitude.

3. **Can $p^*$ be computed analytically?** For a Beta agent with a fixed $\varepsilon > 0$ and bandit gap $\Delta$, is there a closed-form expression for the absorption probability? This is related to the Gambler's Ruin problem but for a non-linear updating rule.

4. **What is the distribution of truth share across runs?** The mean $\mu$ and variance $\text{Var}(X)$ are the first two moments. Is $X$ approximately normal for large $N$? If so, a Central Limit Theorem with the network correlation structure would give a complete distributional prediction.

5. **Cycles.** For graphs with cycles, root nodes are not well-defined and the recursion does not apply. What replaces the topological recursion? A fixed-point equation: $P_i = 1 - (1-p^r)\prod_{j \in \text{Pa}(i)}(1-P_j)$, now solved simultaneously rather than iteratively. Does a unique fixed point always exist? Under what conditions?
