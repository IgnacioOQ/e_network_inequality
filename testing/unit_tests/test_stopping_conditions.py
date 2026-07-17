import unittest
import networkx as nx
import numpy as np

# Adjust imports to find the source code from tests/
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.vectorized_model import VectorizedModel

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

def _offline_stop(flip_history, window, max_steps):
    """Derive the choice-stability stop step for a given window from a
    record_choice_flips history. Returns (stop_step, truth_share) or
    (None, None) if no flip-free gap of length >= window exists.

    flip_history is [(step, truth_share), ...]: a baseline entry at run start
    plus one entry per step where at least one agent flipped. Truth_share is
    constant between flips, so the first inter-event gap of length >= window
    determines the stop.
    """
    steps = [s for s, _ in flip_history] + [max_steps]
    shares = [t for _, t in flip_history]
    for i in range(len(flip_history)):
        if steps[i + 1] - steps[i] >= window:
            return steps[i] + window, shares[i]
    return None, None


class _OscillatingModel(VectorizedModel):
    """Force agent 0 to flip its choice every step, so choice-stability can
    never accumulate a flip-free window — used to exercise the 'never
    stabilizes -> hits cap' path deterministically."""

    def step(self):
        super().step()
        # Toggle agent 0's credence to guarantee a decision-boundary crossing.
        if self.n_steps % 2 == 0:
            self.credences[0] = np.array([1.0, 0.0])
        else:
            self.credences[0] = np.array([0.0, 1.0])


class TestChoiceStabilityStopping(unittest.TestCase):
    def setUp(self):
        self.G = nx.complete_graph(10, create_using=nx.DiGraph())
        self.max_steps = 20000
        self.kwargs = dict(
            n_experiments=100,
            agent_type="beta",
            uncertainty=0.05,
            tolerance_stopping=False,
        )

    def test_stop_gap_equals_window(self):
        """When choice-stability fires, exactly `window` flip-free steps have
        elapsed since the last flip (we break the first step the gap reaches W)."""
        for window in (100, 250, 500, 1000):
            model = VectorizedModel(
                self.G,
                choice_stability_stopping=True,
                choice_stability_window=window,
                seed=7,
                seeded=True,
                **self.kwargs,
            )
            model.run_simulation(number_of_steps=self.max_steps, show_bar=False)
            self.assertLess(model.n_steps, self.max_steps,
                            f"Did not stabilize before cap for window={window}.")
            self.assertEqual(model.n_steps - model._last_flip_step, window,
                             f"Stop gap != window for window={window}.")

    def test_native_matches_offline_derivation(self):
        """Native choice-stability stop step equals the offline-derived stop
        step (and truth share) from a single record_choice_flips run."""
        # One record-once run in tstep mode; derive every window from it.
        rec = VectorizedModel(
            self.G,
            tstep_stopping=True,
            record_choice_flips=True,
            seed=11,
            seeded=True,
            **self.kwargs,
        )
        rec.run_simulation(number_of_steps=self.max_steps, show_bar=False)

        for window in (100, 250, 500, 1000):
            native = VectorizedModel(
                self.G,
                choice_stability_stopping=True,
                choice_stability_window=window,
                seed=11,
                seeded=True,
                **self.kwargs,
            )
            native.run_simulation(number_of_steps=self.max_steps, show_bar=False)

            off_step, off_share = _offline_stop(
                rec.choice_flip_history, window, self.max_steps)
            self.assertEqual(native.n_steps, off_step,
                             f"Native vs offline stop step mismatch (window={window}).")
            self.assertAlmostEqual(native.conclusion, off_share, places=9,
                                   msg=f"Truth share mismatch (window={window}).")

    def test_seeded_determinism(self):
        """Same seed reproduces the stop step exactly."""
        def run():
            m = VectorizedModel(
                self.G,
                choice_stability_stopping=True,
                choice_stability_window=200,
                seed=99,
                seeded=True,
                **self.kwargs,
            )
            m.run_simulation(number_of_steps=self.max_steps, show_bar=False)
            return m.n_steps
        self.assertEqual(run(), run(), "Seeded runs produced different stop steps.")

    def test_never_stabilizes_hits_cap(self):
        """A perpetually-oscillating agent blocks stopping -> run hits the cap."""
        model = _OscillatingModel(
            self.G,
            choice_stability_stopping=True,
            choice_stability_window=100,
            seed=3,
            seeded=True,
            **self.kwargs,
        )
        cap = 500
        model.run_simulation(number_of_steps=cap, show_bar=False)
        self.assertEqual(model.n_steps, cap,
                         "Oscillating agent should have prevented early stop.")

    def test_defaults_off_preserve_behavior(self):
        """New flags default OFF: with tolerance_stopping the model ignores the
        choice-stability machinery entirely."""
        model = VectorizedModel(
            self.G,
            **{**self.kwargs, "tolerance_stopping": True},
        )
        model.run_simulation(number_of_steps=self.max_steps, show_bar=False)
        self.assertFalse(model.choice_stability_stopping)
        self.assertEqual(model.choice_flip_history, [])
        self.assertLess(model.n_steps, self.max_steps)


if __name__ == '__main__':
    unittest.main()
