import unittest
import networkx as nx
import numpy as np

# Adjust imports to find the source code from tests/
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vectorized_model import VectorizedModel

class TestVectorizedModelStopping(unittest.TestCase):
    def setUp(self):
        # Create a small completely connected graph so convergence is fast
        self.G = nx.complete_graph(10, create_using=nx.DiGraph())
        self.max_steps = 1000

    def test_tolerance_stopping_default(self):
        """Test Priority 1: tolerance stopping halts the simulation early."""
        # Using Beta agent because np.allclose(prior, post) will reliably hit when they stop updating.
        model = VectorizedModel(
            self.G,
            n_experiments=50,
            agent_type="beta",
            tolerance_stopping=True,
            tstep_stopping=False
        )
        model.run_simulation(number_of_steps=self.max_steps, show_bar=False)
        
        # Should stop well before max_steps because it converges
        self.assertLess(model.n_steps, self.max_steps, "Simulation did not stop early using tolerance_stopping.")

    def test_tstep_stopping_only(self):
        """Test Priority 2: tstep_stopping runs exactly for number_of_steps."""
        model = VectorizedModel(
            self.G,
            n_experiments=10,
            agent_type="beta",
            tolerance_stopping=False,
            tstep_stopping=True
        )
        model.run_simulation(number_of_steps=self.max_steps, show_bar=False)
        
        # Should run exactly for max steps
        self.assertEqual(model.n_steps, self.max_steps, "Simulation did not run for the exact number of max steps.")

    def test_auc_stopping_only(self):
        """Test Priority 3: auc_stopping stops early if reached, checked at intervals."""
        model = VectorizedModel(
            self.G,
            n_experiments=50, # High experiments to converge faster
            agent_type="beta",
            tolerance_stopping=False,
            tstep_stopping=False
        )
        
        # We set check interval to 50 so it triggers early.
        model.run_simulation(
            number_of_steps=self.max_steps,
            show_bar=False,
            auc_stopping=True,
            auc_threshold=0.95,
            auc_check_interval=50
        )
        
        if model.n_steps < self.max_steps:
             self.assertEqual(model.n_steps % 50, 0, "AUC stopping did not break on the correct interval step.")

if __name__ == '__main__':
    unittest.main()
