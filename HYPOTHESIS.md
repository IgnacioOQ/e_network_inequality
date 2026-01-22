# Left Eigenvector Centrality Hypothesis
- status: active

## Overview
- status: active

This document formalizes the hypothesis that the long-run outcomes of the network epistemology simulation can be analytically predicted using:
1. **Stochastic matrices** derived from the sampling and update mechanism (Binomial experiments, Bayesian updating)
2. **Left eigenvector centrality** derived from the network structure (DeGroot influence)
3. **Initial distribution** of agent beliefs (random priors)

## The Simulation as a Stochastic Process
- status: active

### State Space
- status: active

For **Beta Agents**, the full state at time $t$ is:
$$S_t = \{(\alpha_i^{(0)}, \beta_i^{(0)}, \alpha_i^{(1)}, \beta_i^{(1)})\}_{i=1}^N \subset \mathbb{R}^{4N}$$

The derived **credences** are:
$$c_i^{(k)} = \frac{\alpha_i^{(k)}}{\alpha_i^{(k)} + \beta_i^{(k)}} \quad \text{for theory } k \in \{0, 1\}$$

### Sources of Stochasticity
- status: active

The simulation has **four sources of randomness**:

#### 1. Initial State (Random Priors)
- status: active
```
α_i^(k), β_i^(k) ~ Uniform(0, 4)  independently for each agent i, theory k
```
This determines the initial credences and the "confidence" (sum α+β) each agent starts with.

#### 2. Theory Choice (ε-greedy)
- status: active
```
With probability ε: choose randomly
With probability 1-ε: choose argmax_k c_i^(k)
```
Default is ε=0 (pure greedy), making this deterministic given credences.

#### 3. Experimental Outcomes (Binomial Sampling) — PRIMARY SOURCE
- status: active
```
S_i^(t) ~ Binomial(n_experiments, p_theory)
where p_theory = 0.5 + uncertainty  if theory=1 (truth)
                 0.5 - uncertainty  if theory=0 (false)
```
This is the main ongoing stochasticity driving belief updates.

#### 4. Optional Sampling Update
- status: active
```
If sampling_update=True:
    c_i^(k) ~ Beta(α_i^(k), β_i^(k))  (sample from posterior)
If sampling_update=False:
    c_i^(k) = α_i^(k) / (α_i^(k) + β_i^(k))  (posterior mean)
```

### Update Dynamics
- status: active

At each time step, the update rule is:

$$\alpha_i^{(k)}(t+1) = \alpha_i^{(k)}(t) + \sum_{j \in \mathcal{N}(i) \cup \{i\}} S_j^{(k)}(t)$$

$$\beta_i^{(k)}(t+1) = \beta_i^{(k)}(t) + \sum_{j \in \mathcal{N}(i) \cup \{i\}} F_j^{(k)}(t)$$

where:
- $\mathcal{N}(i)$ = predecessors of agent $i$ in the directed graph (agents that $i$ listens to)
- $S_j^{(k)}(t)$ = successes observed by agent $j$ on theory $k$ at time $t$
- $F_j^{(k)}(t)$ = failures observed by agent $j$ on theory $k$ at time $t$
- Only agents testing theory $k$ contribute non-zero $(S, F)$ for that theory

## The Row-Stochastic Listening Matrix
- status: active

### Definition
- status: active

Construct the **listening matrix** $W \in \mathbb{R}^{N \times N}$ where:

$$W_{ij} = \begin{cases}
1 & \text{if } i \text{ is a root (no predecessors) and } j = i \\
\frac{1}{|\text{Pred}(i)|} & \text{if } j \in \text{Pred}(i) \\
0 & \text{otherwise}
\end{cases}$$

Note: This assumes equal weighting of predecessors. The matrix is **row-stochastic** (rows sum to 1).

### Relationship to Adjacency Matrix
- status: active

If $A$ is the adjacency matrix where $A_{ij} = 1$ means $i \to j$ (i.e., $j$ listens to $i$), then:
- $A^T$ has $A^T_{ij} = 1$ if $j$ influences $i$
- $W$ is the row-normalized version of $A^T$ with self-loops for roots

## Left Eigenvector Centrality (DeGroot Influence)
- status: active

### Definition
- status: active

The **left eigenvector** $\pi$ satisfies:
$$\pi W = \pi, \quad \sum_i \pi_i = 1, \quad \pi_i \geq 0$$

This is equivalent to finding the right eigenvector of $W^T$ for eigenvalue 1.

### Interpretation
- status: active

$\pi_i$ represents the **long-run influence** of agent $i$:
- In a DeGroot opinion dynamics model, consensus converges to $\sum_i \pi_i \cdot b_i^{(0)}$
- For DAGs: $\pi$ concentrates entirely on root nodes
- For cyclic graphs: $\pi$ identifies "effective sources" of information

## The Hypothesis
- status: active

### Main Claim
- status: active

The **proportion of agents believing truth** at convergence can be predicted by:

$$\hat{P}(\text{truth}) = \sum_{i=1}^{N} \pi_i \cdot \mathbb{1}[\text{agent } i \text{ converges to truth}]$$

where convergence to truth means $c_i^{(1)} > c_i^{(0)}$ at the final time step.

### Decomposition of Outcomes
- status: active

The final outcome depends on three separable components:

#### Component 1: Network Structure → Left Eigenvector $\pi$
- status: active
- Determined entirely by the directed graph topology
- Captures "who influences whom" in the long run
- For DAGs, equivalent to root-based reachability analysis

#### Component 2: Initial Conditions → Prior Advantage
- status: active
- Random priors create initial bias toward one theory
- Agents with higher initial $c_i^{(1)}$ are more likely to test theory 1
- This interacts with the uncertainty parameter

#### Component 3: Stochastic Evidence Accumulation
- status: active
- Binomial sampling creates variance in belief trajectories
- Theory 1 (truth) has higher expected successes: $E[S] = n \cdot (0.5 + u)$
- Over time, truth-testing agents accumulate evidence favoring truth
- The rate depends on how many predecessors also test truth

### Expected Dynamics
- status: active

In expectation (averaging over binomial randomness):

$$E[\alpha_i^{(1)}(t+1) | S_t] = \alpha_i^{(1)}(t) + n \cdot (0.5 + u) \cdot |\{j \in \mathcal{N}(i) \cup \{i\} : \text{choice}_j = 1\}|$$

The **expected evidence flow** follows the network structure, with the left eigenvector determining the steady-state distribution of "truth-testing mass."

## Testable Predictions
- status: active

### Prediction 1: Left Eigenvector Accuracy
- status: active
For networks with roots, left eigenvector prediction should match root-based prediction.
For cyclic networks, left eigenvector provides unique predictive value.

### Prediction 2: Variance Decomposition
- status: active
Total variance in outcomes = f(initial priors) + g(binomial sampling) + h(network structure)
The network structure component should be captured by spectral properties of $W$.

### Prediction 3: Convergence Time
- status: active
Mixing time should relate to the **spectral gap** $(1 - \lambda_2)$ where $\lambda_2$ is the second-largest eigenvalue of $W$.

### Prediction 4: Initial Condition Sensitivity
- status: active
Agents with high $\pi_i$ should have disproportionate influence on final outcomes.
Perturbing their initial beliefs should have larger effects than perturbing low-$\pi$ agents.

## Connection to Markov Chain Theory
- status: active

### The Process as a Markov Chain
- status: active

The simulation satisfies the Markov property: $P(S_{t+1} | S_t, S_{t-1}, ...) = P(S_{t+1} | S_t)$

However, it is:
- **Non-homogeneous**: Transition probabilities depend on the current state (which theory each agent tests)
- **Non-stationary in state**: The $\alpha, \beta$ parameters grow unboundedly
- **Absorbing in belief**: Once an agent strongly believes one theory, they rarely switch

### Relevant Markov Concepts
- status: active

1. **Absorption probabilities**: $P(\text{converge to truth} | S_0)$
2. **Hitting times**: Expected steps to reach consensus
3. **Coupling time**: When chains from different initial states merge
4. **Spectral gap**: Related to mixing time and convergence rate

## Open Questions
- status: active

1. Can we derive a closed-form expression for $P(\text{agent } i \text{ converges to truth})$ as a function of $\pi_i$, initial priors, and $u$?

2. How does the variance of binomial sampling affect the relationship between $\pi$ and outcomes?

3. Is there a phase transition in the dependence on initial conditions as network density changes?

4. How do cycles in the network affect the predictability of outcomes compared to DAGs?

## References
- status: active

- DeGroot, M. H. (1974). Reaching a consensus. Journal of the American Statistical Association.
- Golub, B., & Jackson, M. O. (2010). Naive learning in social networks and the wisdom of crowds.
- Zollman, K. J. (2007). The communication structure of epistemic communities.
