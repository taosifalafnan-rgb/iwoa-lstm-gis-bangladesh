"""
test_ahp.py — Tests for the AHP weight-derivation stage.

Covers config loadability (regression for the integer-keyed ri_values bug),
priority-vector properties, and consistency-ratio validation.
"""

import pytest

from src.analysis.ahp import AHP


def test_config_loads():
    """Config must load despite integer-keyed dicts (ri_values, lulc_classes)."""
    from src.utils.config import cfg
    assert cfg.seed == 42
    assert isinstance(cfg.ahp.ri_values, dict)


def test_weights_sum_to_one():
    """AHP priority vector must sum to 1.0."""
    weights, _ = AHP().compute()
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_consistency_ratio_acceptable():
    """The configured pairwise matrix must be consistent (CR < 0.10)."""
    _, cr = AHP().compute()
    assert cr < 0.10


def test_air_quality_is_dominant_criterion():
    """Air quality should carry the largest weight for this study."""
    weights, _ = AHP().compute()
    assert max(weights, key=weights.get) == "air_quality"


def test_all_criteria_present():
    """Every configured criterion must appear in the weights dict."""
    weights, _ = AHP().compute()
    for c in ["air_quality", "water_quality", "thermal_stress",
              "green_cover", "socioeconomic"]:
        assert c in weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
