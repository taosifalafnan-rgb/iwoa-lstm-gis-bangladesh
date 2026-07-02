"""
hhi_panel.py — Raw ward-level environmental panel for HHI assessment.

The Human Habitability Index is computed from *raw physical measurements*
(PM2.5 in μg/m³, BOD in mg/L, LST in °C, ...) evaluated against reference
standards — NOT from the [0, 1]-normalized LSTM feature matrix produced by
`src/data/preprocessor.py`. This module builds that raw panel and supplies it,
with a ward (spatial) dimension, to `src/analysis/hhi.py`.

Two sources are supported, in priority order:

  1. A real processed panel on disk (if a prior stage produced one).
  2. A reproducible **synthetic demo panel** — generated from the study-area
     ward geography and physically plausible industrialization trends — so the
     entire HHI flow is runnable end-to-end before field datasets are dropped
     into data/. The synthetic panel is deterministic given `cfg.seed`.

It also projects **scenario panels** (BAU / S1 / S2) from 2025 to 2040 by
applying the growth / compliance / green-loss rates defined in
`configs/config.yaml → scenarios` to the last observed (2024) ward state. When
real LSTM scenario forecasts become available, the orchestrator can replace
this projection with `src/models/forecaster.py` output — the HHI computation
downstream is identical.

Columns produced (all raw physical units):
    id:   date, ward_id, district, upazila, lat, lon
    C1:   pm25 (μg/m³), no2 (μg/m³), so2 (μg/m³), co (mg/m³)
    C2:   bod (mg/L), do (mg/L), ph, turbidity (NTU)
    C3:   lst (°C), uhi_intensity (°C)
    C4:   ndvi (0-1), green_loss_fraction (0-1)
    C5:   pop_density (persons/km²), dist_industrial (km), industrial_fraction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


# ── STUDY-AREA WARD GEOGRAPHY ───────────────────────────────────────────────
# Ten administrative units (upazilas) across the two study districts, used as
# the spatial reporting units ("wards") for HHI. Centroids fall inside the
# study-area bounding box (CLAUDE.md: 90.20–90.70 E, 23.55–24.10 N).
# `urban` is a 0–1 baseline industrialization/urbanization intensity that
# drives the synthetic generator (higher = more industrial).
WARDS = [
    # ward_id, district,        upazila,             lon,    lat,   urban
    (1,  "Gazipur",     "Gazipur Sadar",     90.42, 23.99, 0.90),
    (2,  "Gazipur",     "Kaliakair",         90.22, 24.06, 0.70),
    (3,  "Gazipur",     "Sreepur",           90.47, 24.09, 0.50),
    (4,  "Gazipur",     "Kaliganj",          90.57, 23.92, 0.55),
    (5,  "Gazipur",     "Kapasia",           90.57, 24.08, 0.35),
    (6,  "Narayanganj", "Narayanganj Sadar", 90.50, 23.62, 0.95),
    (7,  "Narayanganj", "Bandar",            90.52, 23.60, 0.75),
    (8,  "Narayanganj", "Rupganj",           90.53, 23.75, 0.80),
    (9,  "Narayanganj", "Sonargaon",         90.62, 23.65, 0.50),
    (10, "Narayanganj", "Araihazar",         90.65, 23.78, 0.45),
]

# Rivers in Narayanganj (Shitalakhya/Buriganga) carry heavy industrial effluent,
# so water-quality degradation is amplified there.
DISTRICT_RIVER_LOAD = {"Gazipur": 0.7, "Narayanganj": 1.0}


def _clip(x, lo, hi):
    """Clip a scalar/array to [lo, hi]."""
    return float(np.clip(x, lo, hi))


# ── SYNTHETIC HISTORICAL PANEL ──────────────────────────────────────────────

def _generate_synthetic_panel(
    start_year: int, end_year: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate a reproducible synthetic raw environmental panel (annual).

    Each ward's pollution/thermal/green state degrades as its industrial
    fraction rises over time, at a pace set by its baseline `urban` intensity.
    Values are drawn in physically plausible ranges for the Dhaka industrial
    belt and are deterministic for a fixed `rng`.

    Args:
        start_year: First year (inclusive).
        end_year:   Last year (inclusive).
        rng:        Seeded NumPy random generator.

    Returns:
        DataFrame with one row per (ward, year) and all raw HHI columns.
    """
    years = list(range(start_year, end_year + 1))
    span = max(end_year - start_year, 1)
    rows = []

    for wid, district, upazila, lon, lat, urban in WARDS:
        river = DISTRICT_RIVER_LOAD[district]
        for y in years:
            t = (y - start_year) / span            # 0 → 1 across the record
            n = lambda s: rng.normal(0, s)         # convenience noise helper

            # Industrialization grows over time, faster in urban wards.
            ind = _clip(0.03 + 0.55 * urban * t + 0.08 * urban + n(0.02),
                        0.0, 0.95)

            # ── C1 Air quality (μg/m³, CO in mg/m³) ──
            pm25 = _clip(30 + 65 * ind + 12 * urban + n(6), 12, 160)
            no2  = _clip(14 + 55 * ind + n(4), 5, 110)
            so2  = _clip(7 + 45 * ind + n(3), 2, 100)
            co   = _clip(0.7 + 4.5 * ind + n(0.3), 0.3, 9)

            # ── C2 Water quality (mg/L, NTU) — worse on industrial rivers ──
            bod  = _clip(2 + 13 * ind * river + n(0.8), 0.5, 18)
            do   = _clip(8.2 - 6.5 * ind * river + n(0.4), 0.3, 9)
            ph   = _clip(7.2 + 1.4 * ind * river + n(0.2), 5.5, 9.2)
            turb = _clip(4 + 22 * ind + n(1.5), 1, 40)

            # ── C3 Thermal (°C) ──
            lst  = _clip(26 + 13 * ind + 6 * urban * t + n(0.8), 22, 46)
            uhi  = _clip(0.5 + 5.2 * ind + n(0.3), 0.0, 6.5)

            # ── C4 Green cover ──
            ndvi = _clip(0.62 - 0.42 * ind - 0.15 * urban * t + n(0.03),
                         0.05, 0.8)
            gloss = _clip(0.30 * ind + 0.20 * t + n(0.03), 0.0, 0.7)

            # ── C5 Socioeconomic ──
            pop  = _clip((4000 + 38000 * urban) * (1 + 0.02 * (y - start_year))
                         + n(800), 500, 60000)
            dist = _clip(12 * (1 - urban) + n(0.5), 0.3, 15)

            rows.append({
                "date": pd.Timestamp(year=y, month=1, day=1),
                "year": y, "ward_id": wid, "district": district,
                "upazila": upazila, "lon": lon, "lat": lat,
                "pm25": pm25, "no2": no2, "so2": so2, "co": co,
                "bod": bod, "do": do, "ph": ph, "turbidity": turb,
                "lst": lst, "uhi_intensity": uhi,
                "ndvi": ndvi, "green_loss_fraction": gloss,
                "pop_density": pop, "dist_industrial": dist,
                "industrial_fraction": ind,
            })

    df = pd.DataFrame(rows)
    log.info(f"Synthetic raw panel generated: {len(df)} rows "
             f"({len(WARDS)} wards × {len(years)} years, "
             f"{start_year}–{end_year}).")
    return df


# ── HISTORICAL PANEL (real if available, else synthetic) ────────────────────

def _raw_panel_path() -> Path:
    """Config-derived path for a real raw panel, if a prior stage saved one."""
    return Path(cfg.data.processed.features_file).parent / "hhi_panel_raw.csv"


def build_historical_panel(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Build the historical (2000–2024) raw ward-level panel.

    Loads a real processed panel from disk if present; otherwise generates the
    reproducible synthetic demo panel so the flow is runnable immediately.

    Args:
        force_synthetic: If True, always generate the synthetic panel and
                         ignore any real panel on disk.

    Returns:
        DataFrame with raw HHI columns for every (ward, year).
    """
    set_seed(cfg.seed)
    path = _raw_panel_path()

    if path.exists() and not force_synthetic:
        df = pd.read_csv(path, parse_dates=["date"])
        log.info(f"Loaded real raw HHI panel: {path} ({len(df)} rows).")
        return df

    if not force_synthetic:
        log.warning(f"No real raw panel at {path} — generating synthetic demo "
                    f"panel. Drop a real panel there to override.")

    start = pd.to_datetime(cfg.dates.historical_start).year
    end   = pd.to_datetime(cfg.dates.historical_end).year
    rng = np.random.default_rng(cfg.seed)
    df = _generate_synthetic_panel(start, end, rng)

    # Persist the synthetic panel so results are traceable/reproducible.
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Synthetic raw panel saved: {path}")
    return df


# ── SCENARIO PROJECTION (2025–2040) ─────────────────────────────────────────

def _scenario_rates(scenario: str):
    """
    Fetch the growth/compliance/loss rates for a scenario from config.

    Args:
        scenario: One of "BAU", "S1", "S2".

    Returns:
        Namespace with industrial_growth_rate, etp_compliance_rate,
        green_loss_rate, pop_growth_rate.
    """
    return getattr(cfg.scenarios, scenario)


def project_scenario_panel(
    historical: pd.DataFrame,
    scenario: str,
    forecast_end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Project a raw ward-level panel forward under a policy scenario.

    Starting from each ward's last observed (final historical) year, iterate
    year-by-year applying the scenario's annual rates. Air pollutants and
    thermal stress rise with industrial growth but are abated by ETP
    compliance; water quality improves with compliance; green cover erodes at
    the scenario green-loss rate; population density grows.

    Args:
        historical:        Historical raw panel from `build_historical_panel`.
        scenario:          "BAU", "S1", or "S2".
        forecast_end_year: Last projected year (defaults to config forecast_end).

    Returns:
        DataFrame of projected raw columns for every (ward, future-year),
        with a "scenario" column added.
    """
    r = _scenario_rates(scenario)
    g   = r.industrial_growth_rate     # annual industrial expansion
    etp = r.etp_compliance_rate        # fraction of factories treating effluent
    gl  = r.green_loss_rate            # annual vegetation loss
    pg  = r.pop_growth_rate            # annual population growth

    last_year = int(historical["year"].max())
    end_year  = forecast_end_year or pd.to_datetime(cfg.dates.forecast_end).year
    future_years = list(range(last_year + 1, end_year + 1))

    log.info(f"Projecting scenario '{scenario}' {last_year + 1}–{end_year} "
             f"(g={g}, etp={etp}, green_loss={gl}, pop={pg})")

    projected = []
    # Seed each ward's trajectory from its final observed (2024) state.
    state = {int(rw["ward_id"]): dict(rw)
             for _, rw in historical[historical["year"] == last_year].iterrows()}

    for y in future_years:
        for wid, s in state.items():
            ind = _clip(s["industrial_fraction"] * (1 + g), 0.0, 0.98)

            # Air: grows with industry, abated by compliance.
            air_mult = (1 + g) * (1 - 0.10 * etp)
            pm25 = _clip(s["pm25"] * air_mult, 8, 200)
            no2  = _clip(s["no2"]  * air_mult, 3, 140)
            so2  = _clip(s["so2"]  * air_mult, 1, 130)
            co   = _clip(s["co"]   * air_mult, 0.2, 12)

            # Water: improves with ETP compliance, slight pressure from growth.
            bod  = _clip(s["bod"] * (1 - 0.15 * etp) * (1 + 0.3 * g), 0.3, 20)
            do   = _clip(s["do"]  * (1 + 0.10 * etp), 0.3, 9.5)
            ph   = _clip(s["ph"] + (7.0 - s["ph"]) * 0.15 * etp, 5.5, 9.5)
            turb = _clip(s["turbidity"] * (1 - 0.10 * etp) * (1 + 0.2 * g), 1, 45)

            # Thermal: rises with industry, mitigated by compliance/greening.
            lst  = _clip(s["lst"] + 3.0 * g - 1.0 * gl_effect(gl, etp), 22, 48)
            uhi  = _clip(s["uhi_intensity"] * (1 + g) * (1 - 0.10 * etp), 0, 7)

            # Green: erodes at the scenario loss rate (S2 = 0 → stable).
            ndvi = _clip(s["ndvi"] * (1 - gl), 0.03, 0.85)
            gloss = _clip(s["green_loss_fraction"] + gl, 0.0, 0.9)

            # Socioeconomic.
            pop  = _clip(s["pop_density"] * (1 + pg), 500, 80000)
            dist = _clip(s["dist_industrial"] * (1 - 0.15 * g), 0.2, 15)

            new = {
                "date": pd.Timestamp(year=y, month=1, day=1), "year": y,
                "ward_id": wid, "district": s["district"],
                "upazila": s["upazila"], "lon": s["lon"], "lat": s["lat"],
                "pm25": pm25, "no2": no2, "so2": so2, "co": co,
                "bod": bod, "do": do, "ph": ph, "turbidity": turb,
                "lst": lst, "uhi_intensity": uhi,
                "ndvi": ndvi, "green_loss_fraction": gloss,
                "pop_density": pop, "dist_industrial": dist,
                "industrial_fraction": ind, "scenario": scenario,
            }
            projected.append(new)
            state[wid] = new   # carry forward for next year's iteration

    df = pd.DataFrame(projected)
    log.info(f"Scenario '{scenario}' projected: {len(df)} rows.")

    # Persist for traceability.
    out_dir = Path(cfg.data.processed.scenarios_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"panel_{scenario}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Scenario panel saved: {out_path}")
    return df


def gl_effect(green_loss_rate: float, etp: float) -> float:
    """
    Net greening effect on temperature relief.

    When green-loss is zero (S2) and compliance is high, some cooling relief is
    applied; otherwise none. Returns a small positive relief factor.

    Args:
        green_loss_rate: Scenario annual vegetation loss rate.
        etp:             ETP compliance fraction.

    Returns:
        Relief factor (°C-scale) subtracted from projected LST growth.
    """
    return etp * (0.02 - green_loss_rate) if green_loss_rate < 0.02 else 0.0


def build_all_scenarios(
    historical: pd.DataFrame, scenarios: Optional[List[str]] = None
) -> dict:
    """
    Project every scenario panel from the historical panel.

    Args:
        historical: Historical raw panel.
        scenarios:  Scenario names (defaults to BAU, S1, S2).

    Returns:
        Dict mapping scenario name to its projected raw panel DataFrame.
    """
    scenarios = scenarios or ["BAU", "S1", "S2"]
    return {sc: project_scenario_panel(historical, sc) for sc in scenarios}


if __name__ == "__main__":
    log.info("hhi_panel smoke test...")
    hist = build_historical_panel(force_synthetic=True)
    print(f"\nHistorical panel: {hist.shape}")
    print(hist[["year", "district", "upazila", "pm25", "bod", "lst", "ndvi",
                "industrial_fraction"]].head(6).to_string(index=False))

    scenarios = build_all_scenarios(hist)
    for name, df in scenarios.items():
        last = df[df["year"] == df["year"].max()]
        print(f"\n{name} 2040 mean PM2.5={last['pm25'].mean():.1f} "
              f"BOD={last['bod'].mean():.1f} NDVI={last['ndvi'].mean():.2f}")
    log.info("hhi_panel smoke test passed.")
