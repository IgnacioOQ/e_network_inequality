# Stochastic Truth-Convergence Hypothesis
- status: active
- type: theory
- owner: user
- last_updated: 2026-04-11
<!-- content -->

> **Scope reminder:** scripts testing this hypothesis must not modify any file outside
> `model/convergence_analysis/`. See `MC_AGENT.md` for the full constraints.

---

## 0. Simulation Model

This section is a self-contained description of the simulation so that the hypothesis can be studied independently of the codebase.

### 0.1 Setting

$N$ agents are arranged on a **directed graph** $G = (V, E)$. An edge $(u, v) \in E$ means agent $v$ **observes** (listens to) agent $u$: at each step, $v$ receives the experimental outcomes produced by $u$. Edges encode information flow, not physical connection.

There are two competing theories:

| Theory | Success probability |
|--------|-------------------|
| Theory 0 (bad / false) | $p_0 = 0.5$ |
| Theory 1 (good / true) | $p_1 = 0.5 + \Delta$ |

where $\Delta > 0$ is the **bandit gap** (called `uncertainty` in the code). The ground truth is that Theory 1 is better, but agents do not know this — they must learn it through experimentation.

### 0.2 Agent State

Each agent $i$ maintains a **Beta distribution** over each theory's success probability. The state is four numbers:

$$(\alpha_i^{(0)},\ \beta_i^{(0)},\ \alpha_i^{(1)},\ \beta_i^{(1)})$$

where $\alpha_i^{(k)}$ counts accumulated successes and $\beta_i^{(k)}$ counts accumulated failures for theory $k$. The **credence** (mean belief) for theory $k$ is:

$$c_i^{(k)} = \frac{\alpha_i^{(k)}}{\alpha_i^{(k)} + \beta_i^{(k)}}$$

**Initialization:** all agents start with uniform priors:

$$\alpha_i^{(0)} = \beta_i^{(0)} = \alpha_i^{(1)} = \beta_i^{(1)} = 1 \implies c_i^{(0)} = c_i^{(1)} = 0.5 \quad \forall i$$

### 0.3 Per-Step Dynamics

At each discrete time step $t$, every agent executes three phases simultaneously:

**Phase 1 — Choice (epsilon-greedy):**

$$k_i(t) = \begin{cases} \text{Uniform}\{0, 1\} & \text{with probability } \varepsilon \quad \text{(explore)} \\ \operatorname{argmax}_k\, c_i^{(k)}(t) & \text{with probability } 1 - \varepsilon \quad \text{(exploit)} \end{cases}$$

Ties in argmax are broken randomly. $\varepsilon \in (0, 1)$ is the **exploration rate** (fixed throughout the simulation).

**Phase 2 — Experiment:**

Agent $i$ runs $n$ independent Bernoulli trials on its chosen theory $k_i(t)$:

$$S_i(t) \sim \operatorname{Binomial}(n,\ p_{k_i(t)}), \qquad F_i(t) = n - S_i(t)$$

The individual results matrix is:
$$s_i^{(k)}(t) = S_i(t) \cdot \mathbf{1}[k = k_i(t)], \qquad f_i^{(k)}(t) = F_i(t) \cdot \mathbf{1}[k = k_i(t)]$$

(Agent $i$ produces evidence only for the theory it tested; the other theory's entry is zero.)

**Phase 3 — Observation and Bayesian Update:**

Agent $i$ observes its own results and those of every agent it listens to ($j \in \text{Pa}(i)$). Total evidence received for theory $k$:

$$\hat{S}_i^{(k)}(t) = s_i^{(k)}(t) + \sum_{j \in \text{Pa}(i)} s_j^{(k)}(t)$$
$$\hat{F}_i^{(k)}(t) = f_i^{(k)}(t) + \sum_{j \in \text{Pa}(i)} f_j^{(k)}(t)$$

Bayesian update (conjugate Beta–Binomial):

$$\alpha_i^{(k)}(t+1) = \alpha_i^{(k)}(t) + \hat{S}_i^{(k)}(t)$$
$$\beta_i^{(k)}(t+1) = \beta_i^{(k)}(t) + \hat{F}_i^{(k)}(t)$$

Credences are updated as means: $c_i^{(k)}(t+1) = \alpha_i^{(k)}(t+1)\,/\,(\alpha_i^{(k)}(t+1) + \beta_i^{(k)}(t+1))$.

Note: $\alpha$ and $\beta$ are **monotonically non-decreasing** — evidence is never forgotten. This guarantees that the chain is absorbing: once an agent's credences have strongly concentrated on one theory, the other theory requires an ever-larger evidence shock to dislodge.

### 0.4 Convergence Criterion

Agent $i$ is said to have **converged to truth** if $c_i^{(1)} > c_i^{(0)}$ at the end of the simulation (credence for Theory 1 exceeds credence for Theory 0).

The **simulation stops** when the maximum per-agent credence change in a step falls below a tolerance:

$$\max_{i,k}\left|c_i^{(k)}(t) - c_i^{(k)}(t-1)\right| < \tau$$

Default: $\tau = 10^{-6}$, max steps $= 10^6$.

### 0.5 Reference Parameters

The values used in the empirical verification runs on the PUD citation network:

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Theory 0 success probability | $p_0$ | $0.5$ |
| Theory 1 success probability | $p_1$ | $0.5 + \Delta$ |
| Bandit gap (uncertainty) | $\Delta$ | $0.1$ $\Rightarrow$ $p_1 = 0.6$ |
| Experiments per step | $n$ | $10$ |
| Exploration rate | $\varepsilon$ | $0.1$ |
| Prior | $\alpha^{(k)}_0 = \beta^{(k)}_0$ | $1$ (uniform Beta) |
| Convergence tolerance | $\tau$ | $10^{-6}$ |
| Network | PUD citation network | $N = 87$ nodes, 160 edges, 16 root nodes, 25 simple cycles (lengths 2–6) |

### 0.6 Root Node Dynamics (Relevant to $p^*$)

A **root node** ($\text{Pa}(i) = \emptyset$) observes no neighbors. Its dynamics reduce to a pure two-armed bandit with Beta-updating and epsilon-greedy exploration. At each step it:

1. Exploits $\operatorname{argmax}(c^{(0)}, c^{(1)})$ with probability $1 - \varepsilon$, or explores uniformly with probability $\varepsilon$.
2. Runs $n$ Bernoulli trials on the chosen arm.
3. Updates its own $\alpha^{(k)}, \beta^{(k)}$ directly (no neighbor evidence).

Since $p_1 > p_0$, a root that tests Theory 1 gets systematically more successes per trial. But in early steps, if random outcomes favour Theory 0, the agent may lock into exploitation of Theory 0. The probability $p^*$ that a root agent eventually converges to Theory 1 (before simulation stops) depends on $\Delta$, $\varepsilon$, $n$, and $\tau$. It is **not analytically tractable** in closed form for Beta-updating epsilon-greedy but is straightforwardly estimated by Monte Carlo: run many isolated-agent simulations and record the fraction that end with $c^{(1)} > c^{(0)}$.

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

### 2.3 The Dampening Effect of False Parents

Suppose some parents of node $i$ have converged to falsehood. They are running experiments on Theory 0, accumulating successes at rate $0.5$ per experiment. Node $i$ observes these results and its Bayesian parameters update as:

$$\alpha_i^{(0)} \mathrel{+}= \text{(neighbour successes for T0)}, \quad \beta_i^{(0)} \mathrel{+}= \text{(neighbour failures for T0)}$$

This raises $i$'s credence for Theory 0: $c_i^{(0)} = \alpha_i^{(0)} / (\alpha_i^{(0)} + \beta_i^{(0)})$. A higher $c_i^{(0)}$ means node $i$ exploits Theory 0 more often (epsilon-greedy), which in turn generates *its own* evidence for Theory 0, further entrenching the false belief.

**However, dampening is about timing, not the final state.** Theory 1 has a strictly higher success rate ($0.5 + \Delta > 0.5$). Even when node $i$ is mostly exploiting Theory 0, its occasional $\varepsilon$-fraction explorations of Theory 1 accumulate evidence for T1 at rate $(0.5 + \Delta)$ per exploration. Simultaneously, $c_i^{(0)}$ converges to $0.5$ (the true success rate of T0) as evidence accumulates. In the infinite-time limit:

$$c_i^{(1)} \to 0.5 + \Delta, \qquad c_i^{(0)} \to 0.5$$

So eventually $c_i^{(1)} > c_i^{(0)}$, node $i$ switches to exploiting T1, and truth convergence follows. **False parents cannot permanently prevent truth convergence — they only delay it.**

The delay can be very long. With $k$ false parents each running $n$ experiments per step, the rate of false T0 evidence arriving at node $i$ is proportional to $k \cdot n$, while the rate of T1 evidence from $i$'s own exploration is only $\varepsilon \cdot n$. The ratio is $k / \varepsilon$ — for $k = 1$ false parent and $\varepsilon = 0.1$, false evidence arrives 10 times faster than true evidence. The crossover time at which $c_i^{(1)} > c_i^{(0)}$ scales roughly as $O(k / (\varepsilon \Delta^2))$, growing with the number of false parents and shrinking with the bandit gap.

**The resistance probability $p^r(t)$** is therefore not a static constant but a function of time:

$$p^r(t) = P(\text{node } i \text{ has converged to truth by step } t \mid \text{all parents converge to falsehood})$$

$p^r(t)$ starts near $p^*$ (no accumulated false evidence at $t = 0$), decreases as false-parent evidence builds up, and then recovers slowly as T1 evidence eventually overcomes T0 evidence. For practical simulation horizons (e.g., $10^6$ steps), $p^r \approx 0$ for nodes with one or more active false parents. The $\approx 1\%$ lower-bound anomaly in the Root Node Hypothesis data represents the small but non-zero residual $p^r$ at $10^6$ steps.

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

## 4. Cycles and the Condensation Generalization

### 4.1 The PUD Network Is Not a DAG

The empirical PUD network ($N = 87$, 160 edges) contains **25 simple cycles** with lengths 2–6, alongside 16 root nodes (in-degree 0). The topological recursion of Section 3 does not apply directly: there is no topological ordering when cycles exist, and the formula $P_i = f(\{P_j\}_{j \in \text{Pa}(i)})$ for a node inside a cycle references its own probability through the cycle.

### 4.2 The Condensation Graph

The standard remedy is to replace the original graph $G$ with its **condensation** $\tilde{G}$:

1. **Find all Strongly Connected Components (SCCs).** An SCC is a maximal set of nodes that can all reach each other. In a DAG every node is its own SCC. In a cyclic graph, each cycle forms (part of) a non-trivial SCC.
2. **Collapse each SCC into a single super-node.** The condensation $\tilde{G}$ has one node per SCC and an edge $S_a \to S_b$ if any node in $S_a$ has an edge to any node in $S_b$. The condensation of any directed graph is always a DAG — this is a theorem.
3. **Apply the topological recursion on $\tilde{G}$.** Each super-node has a single truth probability $P_{S}$, computed from its parents in $\tilde{G}$.

### 4.3 Truth Probability of an SCC

For a trivial SCC (single node, no cycle), $P_S = P_i$ as before.

For a non-trivial SCC (two or more mutually reachable nodes), the internal dynamics must be solved separately. Define $P_S$ as the probability that the SCC as a whole converges to truth — meaning all (or most) nodes inside reach the good theory.

The internal dynamics of an SCC are governed by a fixed-point equation. For each node $i$ inside SCC $S$, define $P_i^S$ as its individual truth probability *given* the truth probabilities of $S$'s parent SCCs. Within the SCC, nodes mutually observe each other, so:

$$P_i^S = 1 - (1 - p^r(t)) \prod_{j \in \text{Pa}(i),\, j \in S} (1 - P_j^S) \cdot \prod_{j \in \text{Pa}(i),\, j \notin S} (1 - P_j)$$

The second product (over parents outside $S$) is known from the condensation recursion; the first product (over parents inside $S$) creates the circular dependency. The solution is the fixed point of this system.

**The symmetric case (all SCC nodes equivalent):** For a uniform cycle of $k$ nodes where each has the same parent structure and same $p^r$, by symmetry $P_i^S = P^S$ for all $i \in S$, and:

$$P^S = 1 - (1 - p^r)(1 - P^S)^{k-1}$$

This is a polynomial fixed-point equation in $P^S$. For $p^r = 0$: $P^S = 0$ or $P^S = 1$ are both solutions; the realized outcome depends on which basin of attraction the system enters. For $p^r > 0$: the only stable fixed point is $P^S = 1$ in infinite time, consistent with the dampening analysis — but the transient time grows with the cycle size.

### 4.4 Mutual Reinforcement in Cycles

A cycle introduces **mutual reinforcement** of whatever belief the cycle converges to. If a length-2 cycle (nodes $A$ and $B$ observe each other) both converge to falsehood, each step:
- $A$'s false T0 experiments reinforce $B$'s false belief
- $B$'s false T0 experiments reinforce $A$'s false belief

The effective false-evidence rate arriving at $A$ is doubled compared to a node with one unidirectional false parent. The crossover time at which T1 evidence overtakes T0 evidence is correspondingly longer. For a length-$k$ cycle of false believers, the crossover time grows approximately as $O(k^2 / (\varepsilon \Delta^2))$ — quadratically in cycle size, because each node is both a source and consumer of false evidence.

This means that empirically, nodes inside cycles are **more likely to be stuck** at finite simulation time than isolated false-parent situations, even though the theoretical infinite-time outcome is the same.

### 4.5 Practical Implication

For the PUD network with 25 short cycles (lengths 2–6):
- Compute SCCs (most will be trivial; a few will be 2–6 node cycles).
- For non-trivial SCCs, solve the internal fixed-point equation numerically.
- Apply the condensation DAG recursion on the resulting super-node probabilities.
- Compare predicted $P_i$ values to empirical convergence frequencies across simulation runs.

---

## 5. Network-Level Predictions

### 5.1 Mean Truth Share

The expected fraction of agents believing truth at convergence:

$$\mu = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \mathbf{1}[T_i]\right] = \frac{1}{N}\sum_{i=1}^N P_i$$

This is exact regardless of correlations — the expectation is linear.

### 5.2 Variance of Truth Share

Let $X = N^{-1}\sum_i \mathbf{1}[T_i]$ be the truth share in a single simulation run.

$$\text{Var}(X) = \frac{1}{N^2} \sum_{i,j} \text{Cov}(\mathbf{1}[T_i], \mathbf{1}[T_j])$$

where $\text{Cov}(\mathbf{1}[T_i], \mathbf{1}[T_j]) = P(T_i \cap T_j) - P_i P_j$.

The joint probability $P(T_i \cap T_j)$ is hard to compute exactly, but has a natural approximation:

$$P(T_i \cap T_j) \approx 1 - (1 - P_i)(1 - P_j) - P(\overline{T_i} \cap \overline{T_j} \mid \text{common ancestors})$$

Nodes sharing many common ancestors have **higher positive covariance** (they tend to succeed or fail together), increasing $\text{Var}(X)$ above the independent-nodes baseline of $N^{-1}\bar{P}(1-\bar{P})$.

The **network correlation matrix** $C_{ij} = P(T_i \cap T_j)$ can be estimated via Monte Carlo simulation (run many replicates and compute empirical joint frequencies). This is the key empirical target.

---

## 6. Connection to Existing Results

### 6.1 Root Node Hypothesis as a Special Case

Set $p^r = 0$ (no resistance). Then $(1 - P_i) = \prod_{j \in \text{Pa}(i)} (1 - P_j)$.

For a node whose only ultimate ancestors are roots, this gives:
- $P_i = 1$ if and only if at least one root ancestor converges to truth.
- $P_i = 0$ if all root ancestors converge to falsehood.

This is exactly the Root Node Hypothesis: the truth share equals the fraction of the network reachable from truthful roots. **The Root Node Hypothesis is the $p^r \to 0$ limit of the Stochastic Hypothesis.**

### 6.2 The Lower-Bound Anomaly (Q1)

In the empirical data, at $10^6$ steps the actual truth share (0.7724) slightly *exceeds* the root-reachability prediction (0.7628). Under the Stochastic Hypothesis, this gap is:

$$\text{Gap} = \frac{1}{N}\sum_{i} p^r \cdot (1 - P(A_i))$$

The sum runs only over nodes whose parent set $A_i^c$ occurs (all parents false). The gap is a direct measurement of $p^r$ weighted by the probability of the resistance regime. The approximately $+1\%$ gap thus gives a rough estimate of $p^r$ for the PUD network.

### 6.3 Two-Phase Dynamics (Q3)

The two-phase dynamics script (`two_phase_dynamics.py`) identified $t^* \approx 550$ as the half-life of active agents. In the Stochastic Hypothesis language:
- **Exploration phase** ($t < t^*$): $p^*$ and $p^r$ are still being "revealed" — agents have not yet committed. The recursion does not yet apply.
- **Exploitation phase** ($t > t^*$): Beliefs have stabilized enough that the recursion's absorbing-state logic applies. The truth-propagation process is approximately over; remaining changes are slow Bayesian accumulation, not structural transitions.

$t^*$ thus marks the onset of the regime in which the Stochastic Hypothesis is predictive.

---

## 7. Spectral Connection (Tentative)

For the condensation graph $\tilde{G}$ (a DAG by construction — see Section 4), the log-space recursion applied to super-nodes is equivalent to a matrix equation:

$$\mathbf{q} = c^* \mathbf{e}_{\text{root-SCCs}} + \tilde{\mathbf{L}}^\top \mathbf{q} + c \, \mathbf{1}$$

where $\tilde{\mathbf{L}}$ is the adjacency matrix of the condensation DAG. Since $\tilde{G}$ is a DAG, $\tilde{\mathbf{L}}$ is nilpotent (all eigenvalues zero). The formal solution is:

$$\mathbf{q} = (I - \tilde{\mathbf{L}}^\top)^{-1}(c^* \mathbf{e}_{\text{root-SCCs}} + c \, \mathbf{1})$$

This is well-defined because $\tilde{\mathbf{L}}$ is nilpotent, so $(I - \tilde{\mathbf{L}}^\top)^{-1} = \sum_{k=0}^{d} (\tilde{\mathbf{L}}^\top)^k$ where $d$ is the depth of the condensation. The $k$-th power counts paths of length $k$ from parent SCCs to descendant SCCs.

**Spectral gap and mixing time (Q7):** For networks with cycles, $\mathbf{L}$ is no longer nilpotent. The spectral gap $1 - |\lambda_2(\mathbf{L})|$ of the row-normalized version controls how fast information diffuses. A small spectral gap (close-to-1 second eigenvalue) means slow mixing — the truth signal from a root takes many steps to reach distant nodes. This connects Q7 (spectral gap) to Q2 (mixing time) via the Stochastic Hypothesis: the mixing time controls *when* the recursion's predictions become accurate, not the predictions themselves.

---

## 8. Empirical Test Protocol

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

## 9. Open Questions

1. **Is $p^r$ topology-independent?** The resistance probability may depend on the number of false parents (more false parents → less resistance) and the strength of their accumulated evidence (more steps → stronger suppression). If so, $p^r$ is not a single scalar but a function $p^r(k, t)$ where $k$ is the number of false parents.

2. **Does the independence assumption systematically bias the recursion?** Nodes in highly clustered regions (many shared ancestors) will have correlated convergences, violating Assumption 2. Characterize the bias direction and magnitude.

3. **Can $p^*$ be computed analytically?** For a Beta agent with a fixed $\varepsilon > 0$ and bandit gap $\Delta$, is there a closed-form expression for the absorption probability? This is related to the Gambler's Ruin problem but for a non-linear updating rule.

4. **What is the distribution of truth share across runs?** The mean $\mu$ and variance $\text{Var}(X)$ are the first two moments. Is $X$ approximately normal for large $N$? If so, a Central Limit Theorem with the network correlation structure would give a complete distributional prediction.

5. **Cycles.** For graphs with cycles, root nodes are not well-defined and the recursion does not apply. What replaces the topological recursion? A fixed-point equation: $P_i = 1 - (1-p^r)\prod_{j \in \text{Pa}(i)}(1-P_j)$, now solved simultaneously rather than iteratively. Does a unique fixed point always exist? Under what conditions?
