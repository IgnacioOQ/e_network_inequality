# Protocol Evolution via Reinforcement Learning

## Core Idea

Treat communication protocols (like MCP) as **signaling systems** that can be optimized using reinforcement learning. Instead of designing protocols top-down, let them evolve based on task success signals from RLHF.

The insight comes from signal-trading games: meaningful communication can emerge without explicitly cooperative payoffs. What matters is the network structure (signal channels exist) and learning dynamics (agents improve over time).

## Framework

**State**: Task context (what the LLM is trying to accomplish)

**Action**: Which description variant to use for a protocol function

**Reward**: RLHF signal (task success or failure)

**Learning**: DQN-style updates over description variants

## Design Choices

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Granularity | Individual function descriptions | Local, composable, testable evolution |
| Credit assignment | Blame the protocol | LLM-agnostic; forces protocol robustness |
| Learning rule | DQN | Handles continuous description space via function approximation |
| Local conventions | Feature, not bug | Different LLM↔Tool pairs can specialize |

## Why DQN

Tabular Q-learning (as in the signal-trading paper) works for small discrete signal spaces. But function descriptions live in high-dimensional natural language space. DQN provides:

- Generalization across similar descriptions
- Experience replay for stable learning
- Scalability to large variant pools

## Information Flow Measurement

Adapt Normalized Mutual Information (NMI) from the signal-trading paper:

**I(Description; TaskSuccess)** measures how well the choice of description predicts task outcome. Higher NMI means the protocol carries more useful information.

## The Distributed Aspect

The network topology mirrors signal-trading games:

```
    Human
      ↕ (RLHF)
    LLM ←—protocol—→ MCP Host
      ↕                  ↕
   Tool₁              Tool₂
```

Each edge is a signal channel with independent rewards. Cooperation emerges from the channel structure, not aligned payoffs—exactly as the paper predicts.

## Connection to Grice's Cooperative Principle

The paper argues that the Cooperative Principle isn't encoded in payoff matrices but in the network of signal channels itself. Opening a communication channel implies commitment to information exchange. The protocol schema embodies this commitment; RLHF teaches agents to honor it effectively.
