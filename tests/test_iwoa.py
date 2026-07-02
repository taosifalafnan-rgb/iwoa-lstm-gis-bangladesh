"""
test_iwoa.py — Tests for the IWOA optimizer core (no torch required).

Covers the search mechanics that don't need a real model: Levy flight, OBL,
position decoding constraints, the improvement toggles that define WOA vs IWOA,
and a full optimize() run against a cheap analytic fitness function.
"""

import numpy as np
import pytest

from src.optimization.iwoa import IWOA, levy_flight, obl_initialization
from src.optimization.woa import WOA


def _analytic_fitness(features, hp):
    """Cheap deterministic fitness: best around 8 features and mid LR."""
    return 0.1 * (len(features) - 8) ** 2 + abs(hp["learning_rate"] - 0.005)


# ── Levy flight ─────────────────────────────────────────────────────────────

def test_levy_flight_shape_and_finite():
    """Levy flight returns the requested shape with all-finite steps."""
    steps = levy_flight(5, 29, lam=1.5)
    assert steps.shape == (5, 29)
    assert np.isfinite(steps).all()


# ── OBL ─────────────────────────────────────────────────────────────────────

def test_obl_doubles_population():
    """OBL returns the original plus opposite population (2N agents)."""
    pop = np.random.rand(6, 10)
    lb, ub = np.zeros(10), np.ones(10)
    combined = obl_initialization(pop, lb, ub)
    assert combined.shape == (12, 10)
    assert (combined >= 0).all() and (combined <= 1).all()


# ── Decoding constraints ────────────────────────────────────────────────────

def test_decode_respects_max_features():
    """Decoded feature count never exceeds cfg max_features."""
    iwoa = IWOA(fitness_fn=_analytic_fitness, n_whales=4, max_iter=1,
                obl_enabled=False)
    pos = np.ones(iwoa.dim)                     # all features "selected"
    feats, hp = iwoa._decode_position(pos)
    assert len(feats) <= iwoa.max_features


def test_decode_enforces_minimum_three():
    """At least three features are always selected, even if none pass 0.5."""
    iwoa = IWOA(fitness_fn=_analytic_fitness, n_whales=4, max_iter=1,
                obl_enabled=False)
    pos = np.zeros(iwoa.dim)                    # nothing "selected"
    feats, _ = iwoa._decode_position(pos)
    assert len(feats) >= 3


def test_hyperparameters_decoded_in_bounds():
    """Decoded hyperparameters fall within their configured bounds."""
    iwoa = IWOA(fitness_fn=_analytic_fitness, n_whales=4, max_iter=1,
                obl_enabled=False)
    _, hp = iwoa._decode_position(np.random.rand(iwoa.dim))
    for key, (lo, hi) in IWOA.HP_BOUNDS.items():
        assert lo <= hp[key] <= hi


# ── Improvement toggles (IWOA vs WOA) ───────────────────────────────────────

def test_woa_disables_all_improvements():
    """The WOA baseline must have every IWOA improvement switched off."""
    woa = WOA(fitness_fn=_analytic_fitness, n_whales=4, max_iter=1)
    assert woa.obl_enabled is False
    assert woa.use_inertia is False
    assert woa.use_levy is False
    assert woa.nonlinear_a is False


def test_adaptive_a_linear_vs_nonlinear():
    """Nonlinear a(t) differs from linear at the midpoint of the run."""
    iwoa = IWOA(fitness_fn=_analytic_fitness, n_whales=4, max_iter=100)
    woa  = WOA(fitness_fn=_analytic_fitness, n_whales=4, max_iter=100)
    # At t = T/2: linear → 1.0, nonlinear → 2 − 2·0.25 = 1.5
    assert abs(woa._adaptive_a(50) - 1.0) < 1e-9
    assert abs(iwoa._adaptive_a(50) - 1.5) < 1e-9


# ── Full optimize() run ─────────────────────────────────────────────────────

def test_optimize_returns_valid_result():
    """A short IWOA run returns a well-formed, constraint-satisfying result."""
    iwoa = IWOA(fitness_fn=_analytic_fitness, n_whales=6, max_iter=5,
                obl_enabled=False)
    result = iwoa.optimize()
    assert 3 <= result.n_selected <= iwoa.max_features
    assert result.n_selected == len(result.selected_features)
    assert np.isfinite(result.best_fitness)
    # Convergence never worsens (best-so-far is monotone non-increasing).
    conv = result.convergence_history
    assert all(conv[i + 1] <= conv[i] + 1e-9 for i in range(len(conv) - 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
