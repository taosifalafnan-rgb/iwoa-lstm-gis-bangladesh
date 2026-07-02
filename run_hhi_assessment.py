"""
run_hhi_assessment.py — End-to-end Human Habitability Index (HHI) assessment.

This is the single entry point for the *complete* HHI flow, top to bottom. It
is self-contained and does NOT require the LSTM/IWOA deep-learning stages to
have run: it builds (or loads) the raw ward-level environmental panel directly,
so a researcher can assess and create the HHI immediately.

Flow (each stage logs to console + outputs/results/run_log.txt):

    ┌─ Stage 0  Reproducibility ── set global seed from config
    ├─ Stage 1  AHP weights ────── pairwise matrix → criteria weights + CR check
    ├─ Stage 2  Raw panel ──────── historical ward panel (real or synthetic)
    ├─ Stage 3  Scenario panels ── BAU/S1/S2 projected 2025–2040
    ├─ Stage 4  Shared scale ───── fit one 0–100 normalization across all panels
    ├─ Stage 5  HHI compute ────── sub-indices C1–C5 → weighted HHI → zones
    ├─ Stage 6  Persist ────────── hhi_historical.csv + hhi_forecast_*.csv
    ├─ Stage 7  Visualize ──────── figures + maps
    └─ Stage 8  Report ─────────── console summary of results

Usage:
    python run_hhi_assessment.py
    python run_hhi_assessment.py --force-synthetic     # ignore any real panel
    python run_hhi_assessment.py --no-viz              # skip figures/maps
    python run_hhi_assessment.py --scenarios BAU S2    # subset of scenarios
"""

import argparse
import time
import pandas as pd

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.analysis.ahp import AHP
from src.analysis.hhi import HHIComputer
from src.analysis.hhi_panel import build_historical_panel, build_all_scenarios

log = get_logger(__name__)


def run(force_synthetic: bool = False, scenarios=None, make_viz: bool = True) -> dict:
    """
    Execute the full HHI assessment.

    Args:
        force_synthetic: If True, always use the synthetic demo panel.
        scenarios:       List of scenario names (default BAU, S1, S2).
        make_viz:        If True, generate figures and maps.

    Returns:
        Dict with keys: weights, cr, historical (HHI DataFrame),
        scenarios (dict of HHI DataFrames), outputs (saved file paths).
    """
    scenarios = scenarios or ["BAU", "S1", "S2"]
    t0 = time.time()

    log.info("=" * 70)
    log.info("HUMAN HABITABILITY INDEX — FULL ASSESSMENT")
    log.info("=" * 70)

    # ── Stage 0: Reproducibility ──────────────────────────────────────────
    set_seed(cfg.seed)

    # ── Stage 1: AHP criteria weights ─────────────────────────────────────
    log.info("STAGE 1: AHP weight derivation")
    ahp = AHP()
    weights, cr = ahp.compute()   # raises if CR >= max_cr (inconsistent)

    # ── Stage 2: Historical raw panel ─────────────────────────────────────
    log.info("STAGE 2: Historical raw ward-level panel")
    hist_panel = build_historical_panel(force_synthetic=force_synthetic)

    # ── Stage 3: Scenario panels (2025–2040) ──────────────────────────────
    log.info("STAGE 3: Scenario projection")
    scenario_panels = build_all_scenarios(hist_panel, scenarios=scenarios)

    # ── Stage 4: Shared 0–100 normalization scale ─────────────────────────
    # Fit sub-index bounds ONCE over the pooled data so historical and all
    # scenarios share a single comparable HHI scale.
    log.info("STAGE 4: Fitting shared HHI normalization scale")
    hc = HHIComputer(weights=weights)
    pooled = pd.concat([hist_panel] + list(scenario_panels.values()),
                       ignore_index=True)
    ref_bounds = hc.fit_reference_bounds(pooled)

    # ── Stage 5: HHI computation ──────────────────────────────────────────
    log.info("STAGE 5: HHI computation")
    hist_hhi = hc.compute(hist_panel, ref_bounds=ref_bounds)
    scenario_hhi = {sc: hc.compute(df, ref_bounds=ref_bounds)
                    for sc, df in scenario_panels.items()}

    # ── Stage 6: Persist results ──────────────────────────────────────────
    log.info("STAGE 6: Saving results")
    outputs = {"historical": hc.save(hist_hhi, "historical")}
    for sc, df in scenario_hhi.items():
        # CLAUDE.md naming convention: hhi_forecast_<SCENARIO>.csv
        outputs[sc] = hc.save(df, f"forecast_{sc}")

    # ── Stage 7: Visualization ────────────────────────────────────────────
    if make_viz:
        log.info("STAGE 7: Visualization")
        try:
            from src.visualization import hhi_plots, hhi_maps
            hhi_plots.generate_all(hist_hhi, scenario_hhi, weights=weights)
            hhi_maps.generate_all(hist_hhi, scenario_hhi)
        except Exception as e:  # viz is non-critical to the numeric result
            log.warning(f"Visualization step failed (non-fatal): {e}")
    else:
        log.info("STAGE 7: Visualization skipped (--no-viz)")

    # ── Stage 8: Summary report ───────────────────────────────────────────
    log.info("STAGE 8: Summary")
    _report(hist_hhi, scenario_hhi)

    log.info("=" * 70)
    log.info(f"HHI ASSESSMENT COMPLETE — {time.time() - t0:.1f}s")
    log.info("=" * 70)

    return {"weights": weights, "cr": cr, "historical": hist_hhi,
            "scenarios": scenario_hhi, "outputs": outputs}


def _report(hist_hhi: pd.DataFrame, scenario_hhi: dict) -> None:
    """
    Log a concise study-area summary: latest historical HHI and each
    scenario's 2040 endpoint, with the dominant vulnerability zone.

    Args:
        hist_hhi:     Historical HHI results.
        scenario_hhi: {scenario: HHI DataFrame}.

    Returns:
        None
    """
    last_year = int(hist_hhi["year"].max())
    latest = hist_hhi[hist_hhi["year"] == last_year]
    log.info(f"  Historical {last_year}: mean HHI = {latest['HHI'].mean():.1f} "
             f"({_dominant_zone(latest)})")

    for sc, df in scenario_hhi.items():
        yr = int(df["year"].max())
        end = df[df["year"] == yr]
        log.info(f"  {sc} {yr}: mean HHI = {end['HHI'].mean():.1f} "
                 f"({_dominant_zone(end)})")


def _dominant_zone(df: pd.DataFrame) -> str:
    """Return the most common vulnerability zone label in df."""
    if "vulnerability_zone" not in df.columns or df.empty:
        return "n/a"
    vc = df["vulnerability_zone"].value_counts()
    return str(vc.index[0]) if len(vc) else "n/a"


def main():
    parser = argparse.ArgumentParser(description="HHI assessment (end-to-end)")
    parser.add_argument("--force-synthetic", action="store_true",
                        help="Always use the synthetic demo panel")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip figure/map generation")
    parser.add_argument("--scenarios", nargs="+", default=["BAU", "S1", "S2"],
                        help="Scenarios to project (default: BAU S1 S2)")
    args = parser.parse_args()

    run(force_synthetic=args.force_synthetic,
        scenarios=args.scenarios,
        make_viz=not args.no_viz)


if __name__ == "__main__":
    main()
