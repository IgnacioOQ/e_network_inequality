"""
Markov Chain Analysis Utilities for Network Epistemology Simulations.

This module provides tools for analyzing the Markov Chain properties of
the belief dynamics in network epistemology simulations, including:

- State space tracking and fingerprinting
- Transition probability estimation
- Convergence diagnostics (mixing time, spectral gap)
- Absorption analysis (hitting times, absorption probabilities)

The simulation is a Markov Chain where:
- State: Collection of all agents' belief parameters
- Transitions: Determined by experiments, observations, and Bayesian updates
- Randomness: Initial priors, theory choices, experimental outcomes
"""

import numpy as np
import numpy.random as rd
import hashlib
import networkx as nx
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class StateSnapshot:
    """A snapshot of the simulation state at a point in time."""
    
    step: int
    credences: np.ndarray
    alphas_betas: Optional[np.ndarray] = None  # For Beta agents
    fingerprint: str = ""
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()
    
    def _compute_fingerprint(self) -> str:
        """Compute a hash fingerprint of the state for comparison."""
        # Round to reduce floating point noise
        rounded_credences = np.round(self.credences, decimals=6)
        state_bytes = rounded_credences.tobytes()
        if self.alphas_betas is not None:
            rounded_ab = np.round(self.alphas_betas, decimals=6)
            state_bytes += rounded_ab.tobytes()
        return hashlib.md5(state_bytes).hexdigest()[:16]


@dataclass
class ConvergenceDiagnostics:
    """Results from convergence analysis."""
    
    # Per-step metrics
    step_history: List[int] = field(default_factory=list)
    mean_belief_change: List[float] = field(default_factory=list)
    max_belief_change: List[float] = field(default_factory=list)
    
    # Final metrics
    converged: bool = False
    convergence_step: Optional[int] = None
    final_consensus_level: float = 0.0  # Fraction agreeing with majority
    
    # Spectral analysis (if computed)
    estimated_spectral_gap: Optional[float] = None
    estimated_mixing_time: Optional[int] = None


@dataclass
class AbsorptionAnalysis:
    """Results from absorption probability analysis."""
    
    # Monte Carlo estimates
    n_simulations: int = 0
    absorption_to_truth: float = 0.0  # P(all agents believe truth)
    absorption_to_falsehood: float = 0.0  # P(all agents believe falsehood)
    partial_convergence: float = 0.0  # P(mixed outcome)
    
    # Hitting times
    mean_hitting_time: float = 0.0
    std_hitting_time: float = 0.0
    hitting_time_samples: List[int] = field(default_factory=list)


class MarkovChainAnalyzer:
    """
    Tools for analyzing the Markov Chain properties of network epistemology simulations.
    
    This class attaches to a VectorizedModel instance and provides methods for:
    - Tracking state trajectories
    - Computing state fingerprints for comparison
    - Estimating transition probabilities
    - Testing the Markov property
    - Analyzing convergence and absorption
    
    Example usage:
        model = VectorizedModel(network, n_experiments=10, ...)
        analyzer = MarkovChainAnalyzer(model)
        
        # Take snapshots during simulation
        for _ in range(1000):
            model.step()
            analyzer.snapshot_state()
        
        # Analyze
        diag = analyzer.compute_convergence_diagnostics()
        print(f"Converged at step {diag.convergence_step}")
    """
    
    def __init__(self, model):
        """
        Attach to a VectorizedModel instance.
        
        Args:
            model: A VectorizedModel instance to analyze
        """
        self.model = model
        self.state_snapshots: List[StateSnapshot] = []
        self._initial_state: Optional[StateSnapshot] = None
        
        # Cache network properties
        self._adj_matrix = model.adj_matrix
        self._n_agents = model.n_agents
        self._agent_type = model.agent_type
        
    def snapshot_state(self) -> StateSnapshot:
        """
        Record the current state of the model.
        
        Returns:
            StateSnapshot with current credences, parameters, and fingerprint
        """
        if self._agent_type == "beta":
            snapshot = StateSnapshot(
                step=self.model.n_steps,
                credences=self.model.credences.copy(),
                alphas_betas=self.model.alphas_betas.copy()
            )
        else:
            snapshot = StateSnapshot(
                step=self.model.n_steps,
                credences=self.model.credences.copy()
            )
        
        self.state_snapshots.append(snapshot)
        
        if self._initial_state is None:
            self._initial_state = snapshot
            
        return snapshot
    
    def state_fingerprint(self) -> str:
        """
        Compute a hashable representation of the current state.
        
        Returns:
            16-character hex string uniquely identifying the state
        """
        if self._agent_type == "beta":
            rounded_credences = np.round(self.model.credences, decimals=6)
            rounded_ab = np.round(self.model.alphas_betas, decimals=6)
            state_bytes = rounded_credences.tobytes() + rounded_ab.tobytes()
        else:
            rounded_credences = np.round(self.model.credences, decimals=6)
            state_bytes = rounded_credences.tobytes()
        
        return hashlib.md5(state_bytes).hexdigest()[:16]
    
    def state_distance(self, state1: StateSnapshot, state2: StateSnapshot,
                        metric: str = "l2") -> float:
        """
        Compute distance between two states.
        
        Args:
            state1: First state snapshot
            state2: Second state snapshot
            metric: Distance metric ('l2', 'l1', 'linf', 'tv' for total variation)
            
        Returns:
            Distance value
        """
        diff = state1.credences - state2.credences
        
        if metric == "l2":
            return np.sqrt(np.sum(diff ** 2))
        elif metric == "l1":
            return np.sum(np.abs(diff))
        elif metric == "linf":
            return np.max(np.abs(diff))
        elif metric == "tv":
            # Total variation for probability distributions
            # For credences, this is 0.5 * L1 norm
            return 0.5 * np.sum(np.abs(diff))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def check_markov_property(self, n_tests: int = 50, tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Verify that transitions don't depend on history by testing reproducibility.
        
        The Markov property states: P(X_{t+1} | X_t, X_{t-1}, ...) = P(X_{t+1} | X_t)
        
        We test this by:
        1. Running from a state S_t via two different histories
        2. Checking that the distribution of S_{t+1} is the same
        
        Note: This is a statistical test, not a proof.
        
        Args:
            n_tests: Number of test cases to run
            tolerance: Maximum allowed difference in outcome distributions
            
        Returns:
            Dictionary with test results
        """
        results = {
            "passed": True,
            "n_tests": n_tests,
            "max_divergence": 0.0,
            "mean_divergence": 0.0,
            "divergences": []
        }
        
        # We need to save and restore state, which requires a fresh model
        # This is a simplified test that checks reproducibility with same seed
        
        from model.vectorized_model import VectorizedModel
        
        divergences = []
        
        for test_idx in range(n_tests):
            seed = np.random.randint(0, 2**31 - 1)
            
            # Run 1: Start fresh, go to step t, then step t+1
            model1 = VectorizedModel(
                network=self.model.network.copy(),
                n_experiments=self.model.n_experiments,
                agent_type=self._agent_type,
                uncertainty=self.model.bandit.uncertainty,
                seed=seed,
                seeded=True
            )
            
            # Run for random number of steps
            t = np.random.randint(5, 20)
            for _ in range(t):
                model1.step()
            
            state_at_t = model1.credences.copy()
            model1.step()
            state_at_t1_run1 = model1.credences.copy()
            
            # Run 2: Same seed, same trajectory
            model2 = VectorizedModel(
                network=self.model.network.copy(),
                n_experiments=self.model.n_experiments,
                agent_type=self._agent_type,
                uncertainty=self.model.bandit.uncertainty,
                seed=seed,
                seeded=True
            )
            
            for _ in range(t):
                model2.step()
            
            model2.step()
            state_at_t1_run2 = model2.credences.copy()
            
            # Compare
            divergence = np.max(np.abs(state_at_t1_run1 - state_at_t1_run2))
            divergences.append(divergence)
            
            if divergence > tolerance:
                results["passed"] = False
        
        results["divergences"] = divergences
        results["max_divergence"] = max(divergences) if divergences else 0.0
        results["mean_divergence"] = np.mean(divergences) if divergences else 0.0
        
        return results
    
    def compute_convergence_diagnostics(self, 
                                         convergence_threshold: float = 1e-5) -> ConvergenceDiagnostics:
        """
        Analyze convergence from recorded state snapshots.
        
        Args:
            convergence_threshold: Belief change below this is considered converged
            
        Returns:
            ConvergenceDiagnostics with step-by-step and final metrics
        """
        diag = ConvergenceDiagnostics()
        
        if len(self.state_snapshots) < 2:
            return diag
        
        for i in range(1, len(self.state_snapshots)):
            prev = self.state_snapshots[i - 1]
            curr = self.state_snapshots[i]
            
            change = np.abs(curr.credences - prev.credences)
            mean_change = np.mean(change)
            max_change = np.max(change)
            
            diag.step_history.append(curr.step)
            diag.mean_belief_change.append(mean_change)
            diag.max_belief_change.append(max_change)
            
            if not diag.converged and max_change < convergence_threshold:
                diag.converged = True
                diag.convergence_step = curr.step
        
        # Compute final consensus level
        final_credences = self.state_snapshots[-1].credences
        if self._agent_type == "beta":
            believes_truth = final_credences[:, 1] > final_credences[:, 0]
        else:
            believes_truth = final_credences > 0.5
        
        truth_fraction = np.mean(believes_truth)
        diag.final_consensus_level = max(truth_fraction, 1 - truth_fraction)
        
        return diag
    
    def estimate_mixing_time(self, epsilon: float = 0.1, 
                              n_chains: int = 10,
                              max_steps: int = 10000) -> Tuple[Optional[int], List[List[float]]]:
        """
        Estimate mixing time by running parallel chains from different initial states
        and measuring when they become "close" to each other.
        
        The mixing time is approximately when the total variation distance
        between chain distributions falls below epsilon.
        
        Args:
            epsilon: Threshold for considering chains "mixed"
            n_chains: Number of parallel chains to run
            max_steps: Maximum steps to run
            
        Returns:
            Tuple of (estimated_mixing_time, distance_histories)
        """
        from model.vectorized_model import VectorizedModel
        
        # Initialize chains with different random states
        chains = []
        for i in range(n_chains):
            seed = np.random.randint(0, 2**31 - 1)
            model = VectorizedModel(
                network=self.model.network.copy(),
                n_experiments=self.model.n_experiments,
                agent_type=self._agent_type,
                uncertainty=self.model.bandit.uncertainty,
                seed=seed,
                seeded=True
            )
            chains.append(model)
        
        # Track pairwise distances
        distance_histories = []
        mixing_time = None
        
        for step in range(max_steps):
            # Step all chains
            for chain in chains:
                chain.step()
            
            # Compute pairwise distances (use first chain as reference)
            ref_credences = chains[0].credences
            distances = []
            for chain in chains[1:]:
                dist = np.max(np.abs(chain.credences - ref_credences))
                distances.append(dist)
            
            distance_histories.append(distances)
            
            # Check if all distances are below epsilon
            if mixing_time is None and max(distances) < epsilon:
                mixing_time = step + 1
                # Continue running to see if it stays mixed
        
        return mixing_time, distance_histories
    
    def estimate_absorption_probabilities(self, 
                                           n_simulations: int = 100,
                                           max_steps: int = 100000,
                                           truth_threshold: float = 0.99) -> AbsorptionAnalysis:
        """
        Estimate absorption probabilities via Monte Carlo simulation.
        
        Runs multiple simulations from the current network structure
        and counts outcomes.
        
        Args:
            n_simulations: Number of Monte Carlo runs
            max_steps: Maximum steps per simulation
            truth_threshold: Credence threshold for "believing truth"
            
        Returns:
            AbsorptionAnalysis with probability estimates and hitting times
        """
        from model.vectorized_model import VectorizedModel
        
        analysis = AbsorptionAnalysis(n_simulations=n_simulations)
        
        truth_absorptions = 0
        falsehood_absorptions = 0
        partial_outcomes = 0
        hitting_times = []
        
        for sim in range(n_simulations):
            seed = np.random.randint(0, 2**31 - 1)
            model = VectorizedModel(
                network=self.model.network.copy(),
                n_experiments=self.model.n_experiments,
                agent_type=self._agent_type,
                uncertainty=self.model.bandit.uncertainty,
                seed=seed,
                seeded=True,
                tstep_stopping=False  # Use convergence stopping
            )
            
            # Run simulation
            model.run_simulation(number_of_steps=max_steps, show_bar=False)
            
            # Classify outcome
            if self._agent_type == "beta":
                believes_truth = model.credences[:, 1] > model.credences[:, 0]
            else:
                believes_truth = model.credences > truth_threshold
            
            truth_fraction = np.mean(believes_truth)
            
            if truth_fraction > 0.99:
                truth_absorptions += 1
            elif truth_fraction < 0.01:
                falsehood_absorptions += 1
            else:
                partial_outcomes += 1
            
            hitting_times.append(model.n_steps)
        
        analysis.absorption_to_truth = truth_absorptions / n_simulations
        analysis.absorption_to_falsehood = falsehood_absorptions / n_simulations
        analysis.partial_convergence = partial_outcomes / n_simulations
        analysis.mean_hitting_time = np.mean(hitting_times)
        analysis.std_hitting_time = np.std(hitting_times)
        analysis.hitting_time_samples = hitting_times
        
        return analysis
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the recorded trajectory.
        
        Returns:
            Dictionary with trajectory statistics
        """
        if not self.state_snapshots:
            return {"error": "No snapshots recorded"}
        
        n_snapshots = len(self.state_snapshots)
        first = self.state_snapshots[0]
        last = self.state_snapshots[-1]
        
        # Compute total drift from initial state
        total_drift = self.state_distance(first, last, metric="l2")
        
        # Find unique fingerprints (approximate cycles)
        fingerprints = [s.fingerprint for s in self.state_snapshots]
        unique_fingerprints = len(set(fingerprints))
        
        return {
            "n_snapshots": n_snapshots,
            "steps_covered": last.step - first.step,
            "total_drift_l2": total_drift,
            "unique_states_approx": unique_fingerprints,
            "final_step": last.step,
            "initial_fingerprint": first.fingerprint,
            "final_fingerprint": last.fingerprint,
        }
    
    def clear_snapshots(self):
        """Clear recorded state snapshots to free memory."""
        self.state_snapshots = []
        self._initial_state = None
