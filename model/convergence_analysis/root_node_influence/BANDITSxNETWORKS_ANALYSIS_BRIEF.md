# Network Epistemology — Collaborator Brief

A self-contained description of the model, the network setting, and the Root Node Hypothesis. The aim is to give you (a) enough to reconstruct any simulation result from scratch, and (b) a clean target for analytic work on convergence and absorption probabilities.

---

## 1. The Model

### 1.1 The two-armed bandit (the world)

There are two competing theories about a Bernoulli process:

- **Theory 0** ("bad"): true success rate $p_0 = 0.5$.
- **Theory 1** ("good"): true success rate $p_1 = 0.5 + \varepsilon$, with $\varepsilon \in (0, 0.5]$.

We call $\varepsilon$ the **bandit gap** (the code calls it `uncertainty`). Typical values in our sims: $\varepsilon \in \{0.001, 0.01, 0.05, 0.1\}$. Small $\varepsilon$ is the interesting regime — it's hard to tell the arms apart.

"Truth" = Theory 1. An agent "believes the truth" when its credence on $T_1$ exceeds its credence on $T_0$.

### 1.2 Agents and their belief state

We have $N$ agents arranged on a directed graph $G = (V, E)$ with $|V| = N$. Each agent maintains a Bayesian belief about **each** theory independently. We use Beta-Bernoulli conjugacy.

For each agent $i$ and each theory $T \in \{0, 1\}$, the belief is a Beta distribution:

$$
\pi_i^T \;=\; \mathrm{Beta}\bigl(\alpha_i^T,\; \beta_i^T\bigr).
$$

The **credence** is the posterior mean:

$$
c_i^T \;=\; \frac{\alpha_i^T}{\alpha_i^T + \beta_i^T}.
$$

(Optional toggle: instead of the mean, sample $c_i^T \sim \mathrm{Beta}(\alpha_i^T, \beta_i^T)$ each step. Default is the mean.)

### 1.3 Initial conditions

At $t = 0$, for every agent $i$ and every theory $T$, the prior parameters are drawn independently:

$$
\alpha_i^T,\; \beta_i^T \;\overset{\text{i.i.d.}}{\sim}\; \mathrm{Uniform}(0, 4).
$$

So each agent's initial credence on $T_1$ is roughly uniform over $(0, 1)$, with a distribution that puts mass mainly near $0.5$ but with non-trivial weight near the boundaries.

### 1.4 Choice rule

At each step, every agent picks **one** theory to test. We use **$\varepsilon$-greedy**:

$$
T_i(t) \;=\;
\begin{cases}
\arg\max_T c_i^T(t) & \text{with prob.\ } 1 - \varepsilon_{\text{explore}},\\[2pt]
\text{uniform in }\{0,1\} & \text{with prob.\ } \varepsilon_{\text{explore}}.
\end{cases}
$$

**Important:** the default in our sims is $\varepsilon_{\text{explore}} = 0$ (pure exploit). This is not a typo — we are deliberately studying the "lock-in" regime. (We use the symbol $\varepsilon$ for the bandit gap; the exploration rate is a separate parameter, which we sometimes denote $\varepsilon_e$ to disambiguate.)

### 1.5 The experiment

Agent $i$ runs $n$ independent Bernoulli trials of its chosen theory $T_i(t)$:

$$
s_i(t) \;\sim\; \mathrm{Binomial}\!\bigl(n,\; p_{T_i(t)}\bigr), \qquad f_i(t) = n - s_i(t).
$$

The parameter $n$ (`n_experiments` in code) is constant across agents and time. Typical values: $n \in \{1, 5, 10, 50\}$.

### 1.6 Information sharing

This is where the network enters. Let $A \in \{0,1\}^{N \times N}$ be the adjacency matrix of $G$, with $A_{uv} = 1$ iff $u \to v$. **The semantics is: $u \to v$ means "$v$ observes $u$".** So $v$'s predecessors $\{u : A_{uv} = 1\}$ are the agents whose results $v$ sees.

At step $t$, after experiments are run, agent $v$ has access to its own outcome **plus** all predecessors' outcomes — but only the outcomes corresponding to the theory each agent actually tested. Crucially:

- If agent $u$ tested $T_0$ at step $t$, then $v$ sees $(s_u, f_u)$ as evidence about $T_0$ only.
- $v$ gets **no information about $T_1$** from $u$ at that step.

Define theory-indexed individual-outcome tensors $S^T_i(t), F^T_i(t)$ so that $(S^T_i, F^T_i) = (s_i, f_i)$ if $T_i(t) = T$ and $(0, 0)$ otherwise. The aggregate evidence received by agent $v$ for theory $T$ at step $t$ is

$$
\widetilde S^T_v(t) \;=\; S^T_v(t) + \sum_{u} A_{uv}\, S^T_u(t), 
\qquad
\widetilde F^T_v(t) \;=\; F^T_v(t) + \sum_{u} A_{uv}\, F^T_u(t).
$$

(In matrix form: $\widetilde S^T = (I + A^\top) S^T$.)

### 1.7 Update rule

Beta-Bernoulli conjugate update, applied independently per theory:

$$
\alpha_v^T(t+1) \;=\; \alpha_v^T(t) + \widetilde S^T_v(t), 
\qquad
\beta_v^T(t+1) \;=\; \beta_v^T(t) + \widetilde F^T_v(t).
$$

Then credences refresh: $c_v^T(t+1) = \alpha_v^T(t+1) / (\alpha_v^T(t+1) + \beta_v^T(t+1))$.

**Structural fact (used a lot below):** $\alpha_v^T, \beta_v^T$ are monotonically non-decreasing in $t$. The state $(\alpha, \beta)$ lives in $\mathbb{Z}_{\geq 0}^{N \times 2 \times 2}$ shifted by the initial real priors; trajectories never revisit a state. The chain on this state space is **absorbing**: as $t \to \infty$, $\alpha + \beta \to \infty$ for any theory that gets tested infinitely often, so the credence on such a theory concentrates at the true rate (SLLN). An untested theory's credence stays at its initial value.

### 1.8 Convergence and "truth share"

We stop when consecutive credence vectors agree within tolerance:

$$
\|c(t) - c(t-1)\|_\infty < \text{tol}, \quad \text{tol} = 5 \times 10^{-3}.
$$

(Or at a fixed max step count, often $10^6$.) Once stopped, the **truth share** is

$$
\mathrm{TS} \;=\; \frac{1}{N} \,\bigl|\{i : c_i^1 > c_i^0\}\bigr|.
$$

This is the main outcome variable.

### 1.9 Parameters at a glance

| Symbol | Meaning | Typical |
|:--|:--|:--|
| $N$ | network size | 50–500 |
| $\varepsilon$ | bandit gap $p_1 - p_0$ | $0.001$–$0.1$ |
| $\varepsilon_e$ | exploration rate | $0$ (default) |
| $n$ | experiments per agent per step | $1$–$50$ |
| $\alpha_0, \beta_0$ | initial Beta params | $\sim U(0,4)$ |
| $t_{\max}$ | max steps | $10^6$ |
| tol | stopping tolerance on credences | $5 \times 10^{-3}$ |

---

## 2. Networks

### 2.1 What we have

We generate or import three families:

1. **Synthetic generators** (in `networks/network_generation.py`):
   - **Directed Barabási–Albert** — preferential attachment on out-degree. Edges go from "cited" to "citing", which gives a heavy-tailed in-degree distribution and a handful of well-cited hubs.
   - **Directed Watts–Strogatz** — ring lattice with rewiring.
   - **Directed Erdős–Rényi** — i.i.d. edges.

2. **Empirical citation networks** (in `networks/citation_data/`):
   - **PUD** — Peptic Ulcer Disease research, 1900–1978, from OpenAlex. After pruning "twin" authors (always co-author, treated as one epistemic unit) and extracting the largest weakly-connected component, the network has $N = 312$ nodes and **72 roots**. It is a **DAG** (no cycles).
   - Tobacco, Ego (similar pipelines).

Edge convention is the same across families: $u \to v$ means $v$ observes $u$ (i.e., $u$ is cited by $v$).

### 2.2 The case we want to focus on

We are focused on **directed networks with root nodes** — nodes with in-degree zero. In the PUD network roughly 23% of nodes are roots.

A root has a special epistemic status: it observes only its own experiments. So **a root is exactly an isolated bandit agent**, embedded in a larger graph. All the network structure does is propagate root beliefs to descendants. This is the structural intuition behind the hypothesis below.

In particular, when $G$ is a **DAG** (the PUD case), every node is reached by some root, and the partition by "which roots can reach me" is well-defined.

---

## 3. The Root Node Hypothesis (as a lower bound)

### 3.1 Statement

Let $R = \{r \in V : \deg^{-}(r) = 0\}$ be the set of roots. Let $R_{\text{true}}(t) \subseteq R$ be the subset of roots that, at time $t$, believe the truth ($c_r^1(t) > c_r^0(t)$). Let $\mathrm{Desc}(r)$ denote the set of nodes reachable from $r$ along directed edges (including $r$ itself). Define the **root-reachable truth share** at time $t$:

$$
\mathrm{LB}(t) \;:=\; \frac{1}{N}\,\Bigl|\bigcup_{r \in R_{\text{true}}(t)} \mathrm{Desc}(r)\Bigr|.
$$

**Hypothesis (corrected framing — lower bound).** As $t \to \infty$,

$$
\mathrm{TS}(t) \;\geq\; \mathrm{LB}(t) \quad\text{in the limit, i.e.}\quad \liminf_{t\to\infty}\bigl[\mathrm{TS}(t) - \mathrm{LB}(t)\bigr] \;\geq\; 0.
$$

That is: **every descendant of a truthful root eventually converges to truth, but additional nodes may also converge to truth via mechanisms independent of root influence.** Root-reachability provides a baseline truth share, not the full picture.

The original ROOTNODE_HYPOTHESIS.md phrases this as the *entire* epistemic state being "determined by" root beliefs. That is too strong — descendant convergence is one source of truth, but not the only one. We are explicitly correcting that here.

### 3.2 Why it's plausible (heuristic), in two halves

**Half 1 — root-driven truth (the lower bound itself).**

1. **Roots are isolated bandits.** A root never receives evidence about a theory it does not personally test. Once it commits to one arm (in the $\varepsilon_e = 0$ regime), the other arm's belief is frozen at its prior mean. So a root's fate is determined by (i) initial priors, (ii) the random sequence of Bernoulli trials it runs on whichever arm it picks first.

2. **Downstream signal swamping.** For any descendant $v$ of a truthful root $r$, evidence on $T_1$ accumulates at $v$ linearly in $t$: per step, $r$ contributes $n$ Bernoulli trials at rate $p_1 = 0.5+\varepsilon$, and the same goes for every truthful intermediate. By SLLN, $c_v^1(t) \to 0.5+\varepsilon$ a.s.; meanwhile $c_v^0$ either stays at its prior or concentrates at $0.5$. Either way $c_v^1 > c_v^0$ eventually.

This is what gives us "TS $\geq$ LB" — every descendant of a truthful root is counted in TS.

**Half 2 — non-root truth (the excess).**

A node $v$ not downstream of any truthful root can still end up at $c_v^1 > c_v^0$. Plausible mechanisms:

- **Favorable-prior lock-in.** $v$'s initial prior already has $c_v^1(0) > c_v^0(0)$ and the early evidence (own + neighbors') is consistent enough with $T_1$ that the agent stays committed to the correct arm, even though no truthful root is upstream to keep feeding it $T_1$ evidence.
- **Cycle-supported consensus.** In a strongly-connected component with no truthful root, agents can collectively reinforce $T_1$ beliefs by circulating their own $T_1$ experiments. (Not possible in strict DAGs.)
- **Isolated components.** Nodes in a weakly-connected component that contains no root in $R_{\text{true}}$ but where individual bandit dynamics happen to land on $T_1$.

We don't expect these to dominate, but they're real, and they're why the relation is $\geq$ rather than $=$.

### 3.3 Empirical status (PUD, $N = 312$, 72 roots, $\varepsilon = 0.05$, $n$ = default)

Running one realization for $10^6$ steps:

| Steps      | $\mathrm{LB}(t)$ (root-reachable share) | Actual $\mathrm{TS}(t)$ | $\mathrm{TS}-\mathrm{LB}$ |
|:--|:--|:--|:--|
| $10^3$     | 0.7244 | 0.4455 | $-0.2788$ |
| $10^4$     | 0.7436 | 0.5994 | $-0.1442$ |
| $10^5$     | 0.7821 | 0.7821 | $\;\;0.0000$ |
| $10^6$     | 0.7628 | 0.7724 | $+0.0096$ |

Interpretation under the corrected framing:

- The **negative gap at early $t$** is the *lag phase*: the lower bound hasn't been achieved yet because signal hasn't propagated from roots to all their descendants. TS is *below* LB transiently — this is allowed because LB is the asymptotic lower bound, not a per-step one.
- The **positive gap at $10^6$** is the *excess from non-root mechanisms*. It is not an anomaly; it is exactly what the $\geq$ framing predicts.
- The exact equality at $10^5$ is a snapshot coincidence on this single realization — across replicates we'd expect a small positive distribution.

### 3.4 Math we have / want

Let's pin down the natural mathematical objects.

**(a) Single-root convergence probability.** Fix an isolated agent with priors $(\alpha^0_0, \beta^0_0, \alpha^1_0, \beta^1_0) \in [0,4]^4$ uniform. In the $\varepsilon_e = 0$ regime, the agent commits at $t=0$ to whichever theory has the higher prior mean, say $T^*$, and from then on:

- runs Bernoulli($p_{T^*}$) trials,
- updates only $(\alpha^{T^*}, \beta^{T^*})$,
- leaves the other theory's parameters frozen.

So $c^{T^*}(t) \to p_{T^*}$ a.s. The agent stays committed to $T^*$ iff $c^{T^*}(t) > c^{1-T^*}_0$ for all $t$. Since the unchosen theory's credence is the **frozen prior mean** $m := \alpha^{1-T^*}_0 / (\alpha^{1-T^*}_0 + \beta^{1-T^*}_0)$, the question becomes: does the random walk $c^{T^*}(t)$ ever cross below $m$?

- If $T^* = 1$ (the agent initially picks the truth): $c^1(t) \to 0.5 + \varepsilon$. The walk may dip below $m$ early, in which case it switches arms. **Open question:** compute $\mathbb{P}(\text{never switches} \mid T^* = 1)$ as a function of $(\alpha_0^1, \beta_0^1, m, \varepsilon, n)$.
- If $T^* = 0$: $c^0(t) \to 0.5$. Switching is more likely (the limit is exactly $0.5$, which is at the typical scale of $m$).

A cleaner sub-question: under the uniform prior on $(0,4)^4$, what is the unconditional probability $q := \mathbb{P}(\text{root converges to truth})$? We have empirical estimates per network; we'd love a closed form or at least an integral expression.

**(b) Descendant propagation lemma (the lower-bound half).** Given a root $r$ has converged to $c_r^1 > c_r^0$ (so $\alpha_r^1, \beta_r^1$ are growing toward the empirical rates of $T_1$), prove that every descendant $v$ of $r$ eventually has $c_v^1 > c_v^0$ almost surely. Empirically this looks like $1$ at any distance, but no formal proof.

A useful decomposition (revised from `OPEN_QUESTIONS.md` §4 under the lower-bound framing):

$$
\mathrm{TS}(\infty) \;=\; \underbrace{\mathrm{LB}(\infty)}_{\text{root-driven}} \;+\; \underbrace{X(\infty)}_{\text{non-root excess}}, \qquad X(\infty) \geq 0.
$$

The lower-bound hypothesis is equivalent to "$\mathrm{LB}(\infty) \leq \mathrm{TS}(\infty)$ a.s." Lemma (b) above gives us the LB term. We then want to characterize $\mathbb{E}[X(\infty)]$ separately.

**(c) The lag phase.** $\mathrm{TS}(t) - \mathrm{LB}(t)$ starts strongly negative (signal hasn't propagated) and crosses zero somewhere between $10^4$ and $10^5$ steps in the PUD example. The lag should scale with some structural property of the graph — diameter? mean root-to-leaf path length? Spectral gap of the row-normalized adjacency matrix? We have no analytic bound.

**(d) The non-root excess $X(\infty)$.** Under the corrected framing this is not an anomaly but a quantity to model. Candidate mechanisms (any non-root truth-believer must come from one of these):

- A handful of non-root nodes have favorable priors and happen to test $T_1$ enough times in the first few steps to lock in correctly before any neighbor's signal arrives or before unfavorable evidence from non-truthful upstream nodes accumulates.
- Self-reinforcing cycles among strongly-connected subgraphs with no truthful root. (Not possible in strict DAGs — but the PUD network is a DAG, so this is ruled out for PUD specifically.)
- Nodes in components or subgraphs without any truthful root where individual bandit dynamics happen to land on $T_1$ despite upstream pressure toward $T_0$.

For PUD ($N=312$, DAG) the $+0.96\%$ is $\sim 3$ extra agents. We'd like to characterize the distribution of $X(\infty)$ across independent realizations, and predict $\mathbb{E}[X(\infty)]$ from graph structure + parameters.

### 3.5 Open analytic questions we'd love your help with

In rough order of how excited we are:

1. **Single-root truth probability.** Compute $q = \mathbb{P}(\text{root converges to truth})$ for uniform $(0,4)^4$ priors, as a function of $(\varepsilon, n)$, with $\varepsilon_e = 0$. Closed form or clean integral both welcome. This pins down $\mathbb{E}[|R_{\text{true}}(\infty)|]$.

2. **Descendant propagation lemma (the LB half).** Show formally that conditional on a root $r$ being truthful, every descendant of $r$ converges to truth a.s. (in the $\varepsilon_e = 0$, $n \geq 1$ regime). What is the minimal condition under which this holds? This is what makes $\mathrm{LB}$ a valid lower bound at all.

3. **Non-root excess $X(\infty)$.** Characterize $\mathbb{E}[X(\infty)]$ — the contribution to TS from nodes *not* downstream of any truthful root. Even an upper bound (e.g., "no more than $f(\varepsilon, n, G)$ of the network can be truth-believers without a truthful upstream root") would be very useful.

4. **Lag-phase bound.** Bound $\mathbb{E}[\mathrm{LB}(t) - \mathrm{TS}(t)]_+$ from above as a function of $t$ and a graph-structural parameter. We suspect graph diameter or some spectral object is the right knob.

5. **Phase transition in $\varepsilon_e$.** With non-zero exploration, the "commit at $t=0$" picture breaks. Is there a critical $\varepsilon_e^*$ below which roots have positive probability of locking onto $T_0$, and above which they almost surely converge to $T_1$?

6. **Generalization to cycles.** PUD is a DAG. For graphs with cycles (BA, WS, ER), "root" isn't always well-defined. Is there a natural surrogate? In-degree-$1$ nodes? Low-betweenness sources? Strongly-connected-component condensation? See §3.6 for our current best candidate (eigenvector centrality). In the cycle case, $X(\infty)$ in (3) becomes much more interesting because cycles can support consensus without a root.

### 3.6 Eigenvector centrality as a root surrogate

We suspect eigenvector centrality is the right generalization of "rootness" — both because it captures a similar intuition on DAGs and because it remains well-defined on graphs with no roots at all.

**Definition.** For a directed graph with adjacency matrix $A$ (recall $A_{uv}=1$ iff $u \to v$, i.e., $v$ listens to $u$), the *influence* eigenvector centrality $x \in \mathbb{R}_{\geq 0}^N$ is the leading right eigenvector of $A$:

$$
A\,x \;=\; \lambda_{\max}\, x, \qquad x_u \;=\; \frac{1}{\lambda_{\max}} \sum_{v\,:\,u\to v} x_v.
$$

That is, $u$ is influential to the extent that it points to influential listeners. (The dual, "receptivity" centrality, is the leading right eigenvector of $A^\top$ — high if you listen to influential sources.) Under Perron–Frobenius, $\lambda_{\max}$ is real and $x$ has non-negative entries; the cleanest setting is when the graph (or the relevant component) is strongly connected, otherwise PageRank-style regularization $M = (1-d) A/\text{out}(\cdot) + d \mathbf{1}\mathbf{1}^\top/N$ is the standard fix.

**Why it should help here.**

1. **Roots are the extreme case.** A root has no incoming edges, so its belief is set purely internally and then broadcast. In $A^\top$ terms, a root's row is zero. In $A$ terms, its column is zero — but its **row of $A$** can be dense and feed into many high-centrality listeners. So roots typically sit at the high end of influence centrality $x$. The "root reachability" lower bound implicitly weights every truthful root by $|\mathrm{Desc}(r)|/N$; eigenvector centrality is the spectral analogue of that weight, generalized to non-root nodes.

2. **Conjectured weighted lower bound.** A natural eigenvector-centrality version of the hypothesis: in any directed graph (with or without roots),

   $$
   \mathrm{TS}(\infty) \;\gtrsim\; \frac{\sum_{u} x_u\, \mathbb{1}[c_u^1(\infty) > c_u^0(\infty)]}{\sum_u x_u}.
   $$

   When $G$ is a DAG, the truthful-root nodes have outsized $x_u$ and the bound reduces (approximately) to the root-reachability LB. When $G$ has no roots, this still gives a meaningful weighted baseline: the agents whose locked-in belief carries the most spectral weight set the asymptotic truth share.

3. **Connection to mixing time.** The spectral gap $1 - |\lambda_2/\lambda_{\max}|$ of the row-normalized adjacency (the "listening matrix" $L = D^{-1} A^\top$ with $D$ the in-degree matrix) governs how fast a perturbation in any single agent's belief propagates to the rest of the network. This is the same spectral object that would naturally appear in a lag-phase bound (§3.5 question 4).

**Open analytic questions in this direction.**

- (a) Does the weighted bound above actually hold? Under what conditions on $G$ and parameters?
- (b) On strongly-connected graphs with no roots, what determines $\mathbb{P}(c_u^1(\infty) > c_u^0(\infty))$ for a fixed node $u$? We conjecture it depends on $x_u$ (or on a centrality-weighted average of upstream truthfulness), but have not formalized.
- (c) Is the second eigenvalue $\lambda_2$ a quantitative predictor of the empirical mixing time we observe in simulation?

---

## 4. References inside this repo (for context, not required)

- `model/vectorized_model.py` — the simulation engine (Beta agents, $\varepsilon$-greedy, network aggregation).
- `model/bandit.py` — the two-armed Bernoulli bandit.
- `model/convergence_analysis/root_node_influence/ROOTNODE_HYPOTHESIS.md` — the hypothesis statement with the PUD table reproduced above.
- `model/convergence_analysis/OPEN_QUESTIONS.md` — broader list of open conceptual questions, including the factorization conjecture in §3.4(b).
- `model/convergence_analysis/root_node_influence/root_influence_analysis.py` — the script that produced the PUD numbers above.

Happy to share any of these directly if useful.
