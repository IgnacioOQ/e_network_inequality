# Root Node Influence Hypothesis
- status: active

## Overview
- status: active

This document details the "Root Node Influence" hypothesis, which posits that in networks with root nodes (nodes with no influencers), the final epistemic state of the entire network is determined by the beliefs of these root nodes.

## The Hypothesis
- status: active

### Statement
- status: active

If a root node converges to the true hypothesis, all of its descendants (nodes reachable from it) will eventually converge to the true hypothesis.

### Prediction
- status: active

The share of agents believing the truth at convergence should be approximately equal to the proportion of the network nodes that are descendants of truthful root nodes.

$$ P(\text{Truth}) \approx \frac{|\bigcup_{r \in R_{true}} \text{Descendants}(r)|}{N} $$

Where $R_{true}$ is the set of root nodes that have converged to the true hypothesis.

## Empirical Verification

A simulation analysis (run with `root_influence_analysis.py`) on the `pud_final.json` empirical network (N=312, 72 roots) revealed the following dynamics over 1,000,000 steps:

| Steps | Predicted (Root Reach) | Actual (Truth Share) | Gap (Actual - Predicted) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| 1,000 | 0.7244 | 0.4455 | -0.2788 | **Lag Phase**: Network has not yet converged; updates are propagating from roots. |
| 10,000 | 0.7436 | 0.5994 | -0.1442 | **Convergence**: Gap is shrinking as descendants align with roots. |
| 100,000 | 0.7821 | 0.7821 | **0.0000** | **Alignment**: Perfect prediction observed in this run. |
| 1,000,000 | 0.7628 | 0.7724 | **+0.0096** | **Lower Bound**: Actual slightly exceeds prediction. |

### Interpretation of Results
1.  **Strong Predictive Power**: The prediction is highly accurate, with the gap narrowing from -28% to <1%.
2.  **Lower Bound Confirmation**: The final positive gap (+0.0096) indicates that `Actual > Predicted`. This suggests that while **root influence drives the vast majority of convergence**, a small fraction of agents (approx. 1%) converge to the truth through mechanisms *independent* of truthful roots (e.g., self-reinforcing cycles or favorable initial priors in isolated components).
3.  **Causality**: The initially negative gap confirms the direction of influence: roots converge first, and their status propagates outward to the rest of the network over time.

## Methodology for Verification
- status: active

### 1. Root Identification
- status: active

Identify all nodes with an in-degree of 0.

### 2. Simulation
- status: active

Run the simulation for a large number of steps (e.g., $10^6$) to ensure convergence propagation.

### 3. Metric Comparison
- status: active

Compare two values:
1. **Predicted**: The proportion of nodes reachable from roots that end up holding the true belief.
2. **Actual**: The empirical fraction of agents who hold the true belief at the end of the simulation.

### 4. Convergence Gap
- status: active

The hypothesis implies that the gap between Predicted and Actual values should shrink to zero as $t \to \infty$.

$$ \lim_{t \to \infty} | \text{Actual}_t - \text{Predicted} | = 0 $$

## Implication
- status: active

If confirmed, this hypothesis suggests that for Directed Acyclic Graphs (DAGs) or any graph with roots, the "wisdom of the crowds" is reducible to the "wisdom of the roots". The graph structure merely propagates the initial biases or learned truths of the root nodes.
