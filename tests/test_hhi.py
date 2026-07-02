"""
test_hhi.py — Tests for the HHI computation, panel builder, and full flow.

Covers sub-index bounds, HHI range, monotonicity (worse environment → higher
HHI), vulnerability classification, panel schema, scenario ordering, and the
end-to-end assessment.
"""

import pandas as pd
import pytest

from src.analysis.hhi import HHIComputer
from src.analysis.hhi_panel import (
    build_historical_panel, build_all_scenarios, WARDS,
)

# Raw physical columns the HHI computer consumes.
RAW_COLS = ["pm25", "no2", "so2", "co", "bod", "do", "ph", "turbidity",
            "lst", "uhi_intensity", "ndvi", "green_loss_fraction",
            "pop_density", "dist_industrial", "industrial_fraction"]


def _row(**overrides):
    """Build a single-row raw panel with baseline values, overridable."""
    base = dict(date=pd.Timestamp("2024-01-01"), year=2024, ward_id=1,
                district="Gazipur", upazila="Test", lon=90.4, lat=23.9,
                pm25=40, no2=20, so2=10, co=1.0, bod=3, do=6, ph=7.2,
                turbidity=8, lst=30, uhi_intensity=2, ndvi=0.5,
                green_loss_fraction=0.1, pop_density=10000,
                dist_industrial=8, industrial_fraction=0.2)
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def hist_panel():
    return build_historical_panel(force_synthetic=True)


# ── Sub-index / HHI range ───────────────────────────────────────────────────

def test_hhi_within_bounds(hist_panel):
    """HHI and every sub-index must stay in [0, 100]."""
    res = HHIComputer().compute(hist_panel)
    for col in ["HHI", "C1_air", "C2_water", "C3_thermal", "C4_green", "C5_socio"]:
        assert res[col].min() >= -1e-6
        assert res[col].max() <= 100 + 1e-6


def test_vulnerability_zone_assigned(hist_panel):
    """Every record must receive a vulnerability-zone label."""
    res = HHIComputer().compute(hist_panel)
    assert res["vulnerability_zone"].notna().all()


# ── Monotonicity: worse environment → higher HHI ───────────────────────────

def test_dirty_worse_than_clean():
    """A polluted ward must score a higher (worse) HHI than a clean one."""
    clean = _row(pm25=15, no2=8, so2=3, co=0.5, bod=1, do=8, turbidity=2,
                 lst=25, uhi_intensity=0.5, ndvi=0.7, green_loss_fraction=0.0,
                 industrial_fraction=0.05)
    dirty = _row(pm25=140, no2=90, so2=70, co=7, bod=15, do=1.5, turbidity=35,
                 lst=44, uhi_intensity=6, ndvi=0.1, green_loss_fraction=0.6,
                 industrial_fraction=0.9)
    df = pd.DataFrame([clean, dirty])
    res = HHIComputer().compute(df)
    assert res.loc[1, "HHI"] > res.loc[0, "HHI"]


def test_raw_subindices_keys(hist_panel):
    """raw_subindices returns all five named components."""
    raw = HHIComputer().raw_subindices(hist_panel)
    assert set(raw) == {"C1_air", "C2_water", "C3_thermal", "C4_green", "C5_socio"}


# ── Panel builder ───────────────────────────────────────────────────────────

def test_panel_schema(hist_panel):
    """Historical panel must contain all raw columns and one row per ward-year."""
    for col in RAW_COLS:
        assert col in hist_panel.columns
    n_years = hist_panel["year"].nunique()
    assert len(hist_panel) == len(WARDS) * n_years


def test_panel_physical_ranges(hist_panel):
    """Raw values must sit in plausible physical ranges."""
    assert hist_panel["ndvi"].between(0, 1).all()
    assert hist_panel["ph"].between(0, 14).all()
    assert (hist_panel["pm25"] > 0).all()


# ── Scenario ordering ───────────────────────────────────────────────────────

def test_scenario_ordering(hist_panel):
    """By 2040, BAU must be worse (higher HHI) than S1, and S1 worse than S2."""
    hc = HHIComputer()
    panels = build_all_scenarios(hist_panel)
    pooled = pd.concat([hist_panel] + list(panels.values()), ignore_index=True)
    bounds = hc.fit_reference_bounds(pooled)

    end = {}
    for sc, df in panels.items():
        res = hc.compute(df, ref_bounds=bounds)
        end[sc] = res[res["year"] == res["year"].max()]["HHI"].mean()

    assert end["BAU"] > end["S1"] > end["S2"]


# ── Shared bounds keep scale comparable ─────────────────────────────────────

def test_shared_bounds_comparable(hist_panel):
    """Using shared bounds, scenario HHI must be directly comparable (0-100)."""
    hc = HHIComputer()
    panels = build_all_scenarios(hist_panel)
    pooled = pd.concat([hist_panel] + list(panels.values()), ignore_index=True)
    bounds = hc.fit_reference_bounds(pooled)
    res = hc.compute(panels["BAU"], ref_bounds=bounds)
    assert res["HHI"].between(0, 100).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
