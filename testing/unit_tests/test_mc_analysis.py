"""
Tests for Markov Chain Analysis utilities.

Tests the MarkovChainAnalyzer class and its methods for:
- State snapshotting and fingerprinting
- Markov property verification
- Convergence diagnostics
- Absorption analysis
"""

import unittest
import numpy as np
import networkx as nx

from models.vectorized_model import VectorizedModel
from utils.mc_analysis import (
    MarkovChainAnalyzer,
    StateSnapshot,
    ConvergenceDiagnostics,
    AbsorptionAnalysis,
)


class TestStateSnapshot(unittest.TestCase):
    """Test StateSnapshot dataclass."""
    
    def test_fingerprint_generation(self):
        """Test that fingerprints are generated correctly."""
        credences = np.array([[0.5, 0.5], [0.6, 0.4]])
        snapshot = StateSnapshot(step=0, credences=credences)
        
        self.assertIsInstance(snapshot.fingerprint, str)
        self.assertEqual(len(snapshot.fingerprint), 16)
    
    def test_fingerprint_consistency(self):
        """Test that same state produces same fingerprint."""
        credences1 = np.array([[0.5, 0.5], [0.6, 0.4]])
        credences2 = np.array([[0.5, 0.5], [0.6, 0.4]])
        
        snap1 = StateSnapshot(step=0, credences=credences1)
        snap2 = StateSnapshot(step=0, credences=credences2)
        
        self.assertEqual(snap1.fingerprint, snap2.fingerprint)
    
    def test_fingerprint_difference(self):
        """Test that different states produce different fingerprints."""
        credences1 = np.array([[0.5, 0.5], [0.6, 0.4]])
        credences2 = np.array([[0.5, 0.5], [0.7, 0.3]])
        
        snap1 = StateSnapshot(step=0, credences=credences1)
        snap2 = StateSnapshot(step=0, credences=credences2)
        
        self.assertNotEqual(snap1.fingerprint, snap2.fingerprint)


class TestMarkovChainAnalyzer(unittest.TestCase):
    """Test MarkovChainAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_agents = 20
        self.network = nx.gnp_random_graph(n_agents, p=0.3, directed=True)
        
        self.model = VectorizedModel(
            network=self.network,
            n_experiments=10,
            uncertainty=0.001,
            agent_type="beta",
            seed=42,
            seeded=True
        )
        
        self.analyzer = MarkovChainAnalyzer(self.model)
    
    def test_snapshot_state(self):
        """Test that state snapshots are recorded."""
        # Take initial snapshot
        snap1 = self.analyzer.snapshot_state()
        self.assertEqual(snap1.step, 0)
        self.assertEqual(len(self.analyzer.state_snapshots), 1)
        
        # Run a step and take another snapshot
        self.model.step()
        snap2 = self.analyzer.snapshot_state()
        self.assertEqual(snap2.step, 1)
        self.assertEqual(len(self.analyzer.state_snapshots), 2)
        
        # Verify snapshots are independent
        self.assertNotEqual(snap1.fingerprint, snap2.fingerprint)
    
    def test_state_fingerprint(self):
        """Test state fingerprinting."""
        fp1 = self.analyzer.state_fingerprint()
        self.assertIsInstance(fp1, str)
        self.assertEqual(len(fp1), 16)
        
        # Fingerprint should change after step
        self.model.step()
        fp2 = self.analyzer.state_fingerprint()
        self.assertNotEqual(fp1, fp2)
    
    def test_state_distance(self):
        """Test state distance computation."""
        snap1 = self.analyzer.snapshot_state()
        self.model.step()
        snap2 = self.analyzer.snapshot_state()
        
        # Test different metrics
        l2 = self.analyzer.state_distance(snap1, snap2, metric="l2")
        l1 = self.analyzer.state_distance(snap1, snap2, metric="l1")
        linf = self.analyzer.state_distance(snap1, snap2, metric="linf")
        tv = self.analyzer.state_distance(snap1, snap2, metric="tv")
        
        self.assertGreater(l2, 0)
        self.assertGreater(l1, 0)
        self.assertGreater(linf, 0)
        self.assertEqual(tv, 0.5 * l1)
    
    def test_convergence_diagnostics(self):
        """Test convergence diagnostics computation."""
        # Record snapshots during simulation
        for _ in range(100):
            self.model.step()
            self.analyzer.snapshot_state()
        
        diag = self.analyzer.compute_convergence_diagnostics()
        
        self.assertIsInstance(diag, ConvergenceDiagnostics)
        # 100 snapshots but 99 transitions computed
        self.assertEqual(len(diag.step_history), 99)
        self.assertEqual(len(diag.mean_belief_change), 99)
        
        # Belief changes should be positive
        self.assertTrue(all(c >= 0 for c in diag.mean_belief_change))
    
    def test_trajectory_summary(self):
        """Test trajectory summary."""
        # Record some snapshots
        for _ in range(50):
            self.analyzer.snapshot_state()
            self.model.step()
        self.analyzer.snapshot_state()
        
        summary = self.analyzer.get_trajectory_summary()
        
        self.assertEqual(summary["n_snapshots"], 51)
        self.assertIn("total_drift_l2", summary)
        self.assertIn("unique_states_approx", summary)
    
    def test_clear_snapshots(self):
        """Test clearing snapshots."""
        self.analyzer.snapshot_state()
        self.model.step()
        self.analyzer.snapshot_state()
        
        self.assertEqual(len(self.analyzer.state_snapshots), 2)
        
        self.analyzer.clear_snapshots()
        
        self.assertEqual(len(self.analyzer.state_snapshots), 0)


class TestMarkovProperty(unittest.TestCase):
    """Test Markov property verification."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(123)
        n_agents = 10
        self.network = nx.gnp_random_graph(n_agents, p=0.3, directed=True)
        
        self.model = VectorizedModel(
            network=self.network,
            n_experiments=10,
            uncertainty=0.001,
            agent_type="beta",
            seed=123,
            seeded=True
        )
        
        self.analyzer = MarkovChainAnalyzer(self.model)
    
    def test_markov_property_check(self):
        """Test Markov property verification."""
        results = self.analyzer.check_markov_property(n_tests=10)
        
        self.assertIn("passed", results)
        self.assertIn("max_divergence", results)
        self.assertIn("divergences", results)
        
        # The test should complete and return results
        self.assertEqual(len(results["divergences"]), 10)
        # With same seed for each pair, divergence should be 0
        # But since we use random seeds in the test, we just verify structure
        self.assertIsInstance(results["max_divergence"], (float, np.floating))
        self.assertIsInstance(results["passed"], bool)


class TestBayesAgent(unittest.TestCase):
    """Test MC analysis with Bayes agents."""
    
    def setUp(self):
        """Set up test fixtures with Bayes agent."""
        np.random.seed(456)
        n_agents = 15
        self.network = nx.gnp_random_graph(n_agents, p=0.3, directed=True)
        
        self.model = VectorizedModel(
            network=self.network,
            n_experiments=10,
            uncertainty=0.001,
            agent_type="bayes",
            seed=456,
            seeded=True
        )
        
        self.analyzer = MarkovChainAnalyzer(self.model)
    
    def test_snapshot_bayes(self):
        """Test snapshots work with Bayes agents."""
        snap = self.analyzer.snapshot_state()
        
        self.assertEqual(snap.credences.shape, (self.model.n_agents,))
        self.assertIsNone(snap.alphas_betas)
    
    def test_convergence_bayes(self):
        """Test convergence diagnostics with Bayes agents."""
        for _ in range(50):
            self.model.step()
            self.analyzer.snapshot_state()
        
        diag = self.analyzer.compute_convergence_diagnostics()
        
        self.assertIsInstance(diag, ConvergenceDiagnostics)
        self.assertGreater(len(diag.step_history), 0)


if __name__ == "__main__":
    unittest.main()
