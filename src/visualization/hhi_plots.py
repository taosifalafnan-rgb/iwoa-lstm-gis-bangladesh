"""
hhi_plots.py — Publication-grade figures for the HHI assessment.

All figures are driven by config (`cfg.visualization`) for DPI, format,
colormap, and output directory. Inputs are the HHI result DataFrames produced
by `src/analysis/hhi.py` (columns: date, year, ward_id, district, C1_air …
C5_socio, HHI, vulnerability_zone).

Figures produced:
    1. hhi_timeseries.png        — study-area & per-district HHI, history + scenarios
    2. hhi_zone_distribution.png — vulnerability-zone composition over time
    3. hhi_subindex_contrib.png  — weighted sub-index contributions by scenario
    4. hhi_ward_ranking.png      — ward HHI ranking for a target year
"""

import matplotlib
matplotlib.use("Agg")  # headless — never requires a display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.analysis.hhi import annual_ward_hhi

log = get_logger(__name__)

# Consistent colors for the four vulnerability zones (green → red).
ZONE_COLORS = {
    "Acceptable": "#1a9850",
    "Moderate":   "#fee08b",
    "At-Risk":    "#fc8d59",
    "Critical":   "#d73027",
}
SUBINDEX_COLS = ["C1_air", "C2_water", "C3_thermal", "C4_green", "C5_socio"]
SUBINDEX_LABELS = {
    "C1_air": "C1 Air", "C2_water": "C2 Water", "C3_thermal": "C3 Thermal",
    "C4_green": "C4 Green", "C5_socio": "C5 Socio",
}
SCENARIO_STYLE = {"BAU": "#d73027", "S1": "#fc8d59", "S2": "#1a9850"}


def _fig_dir() -> Path:
    """Return (and create) the configured figure output directory."""
    d = Path(cfg.visualization.figure_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str) -> str:
    """
    Save a figure to the configured directory/format/DPI.

    Args:
        fig:  Matplotlib figure.
        name: File stem (no extension).

    Returns:
        Full output path as a string.
    """
    ext = cfg.visualization.figure_format
    path = _fig_dir() / f"{name}.{ext}"
    fig.savefig(path, dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved: {path}")
    return str(path)


def plot_hhi_timeseries(
    hist_hhi: pd.DataFrame,
    scenario_hhis: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """
    Plot study-area mean HHI over time, historical then per-scenario forecast.

    Args:
        hist_hhi:      Historical HHI results DataFrame.
        scenario_hhis: Optional {scenario: HHI DataFrame} for 2025–2040.

    Returns:
        Output figure path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Historical study-area mean (solid black).
    h = hist_hhi.groupby("year")["HHI"].mean()
    ax.plot(h.index, h.values, "-o", color="black", lw=2,
            label="Historical (observed)")
    last_year, last_val = h.index[-1], h.values[-1]

    # Scenario projections, each anchored to the last historical point.
    for sc, df in (scenario_hhis or {}).items():
        s = df.groupby("year")["HHI"].mean()
        xs = np.concatenate([[last_year], s.index.values])
        ys = np.concatenate([[last_val], s.values])
        ax.plot(xs, ys, "--", color=SCENARIO_STYLE.get(sc, None), lw=2,
                label=f"{sc} forecast")

    # Vulnerability zone reference bands.
    for zone, (lo, hi) in [("Acceptable", (0, 25)), ("Moderate", (25, 50)),
                           ("At-Risk", (50, 75)), ("Critical", (75, 100))]:
        ax.axhspan(lo, hi, color=ZONE_COLORS[zone], alpha=0.08)

    ax.axvline(last_year, color="gray", ls=":", lw=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("HHI (0–100, higher = less habitable)")
    ax.set_title("Human Habitability Index — Study-Area Mean & Scenarios")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    return _save(fig, "hhi_timeseries")


def plot_zone_distribution(hist_hhi: pd.DataFrame) -> str:
    """
    Stacked-area plot of ward counts per vulnerability zone over time.

    Args:
        hist_hhi: Historical HHI results DataFrame.

    Returns:
        Output figure path.
    """
    # Collapse quarterly rows to one HHI/zone per ward-year before counting,
    # so the stacked totals equal the number of wards (not ward-quarters).
    annual = annual_ward_hhi(hist_hhi)
    counts = (annual.groupby(["year", "vulnerability_zone"], observed=False)
              .size().unstack(fill_value=0))
    order = ["Acceptable", "Moderate", "At-Risk", "Critical"]
    counts = counts.reindex(columns=[c for c in order if c in counts.columns])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(counts.index, counts.T.values,
                 labels=counts.columns,
                 colors=[ZONE_COLORS[c] for c in counts.columns], alpha=0.9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of wards")
    ax.set_title("Vulnerability-Zone Composition Over Time (Historical)")
    ax.legend(loc="upper left", title="Zone")
    ax.margins(x=0)
    return _save(fig, "hhi_zone_distribution")


def plot_subindex_contribution(
    hist_hhi: pd.DataFrame,
    scenario_hhis: Optional[Dict[str, pd.DataFrame]] = None,
    target_year: Optional[int] = None,
    weights: Optional[dict] = None,
) -> str:
    """
    Stacked bar of weighted sub-index contributions to HHI, comparing the
    latest historical year against each scenario's final-year state.

    Args:
        hist_hhi:      Historical HHI results.
        scenario_hhis: Optional {scenario: HHI DataFrame}.
        target_year:   Historical year to show (defaults to last historical).
        weights:       AHP weights dict; if given, bars show weighted shares.

    Returns:
        Output figure path.
    """
    wmap = weights or {}
    wkey = {"C1_air": "air_quality", "C2_water": "water_quality",
            "C3_thermal": "thermal_stress", "C4_green": "green_cover",
            "C5_socio": "socioeconomic"}

    def contrib(df: pd.DataFrame) -> pd.Series:
        means = df[SUBINDEX_COLS].mean()
        return pd.Series({c: means[c] * wmap.get(wkey[c], 0.2)
                          for c in SUBINDEX_COLS})

    ty = target_year or int(hist_hhi["year"].max())
    bars = {f"Hist {ty}": contrib(hist_hhi[hist_hhi["year"] == ty])}
    for sc, df in (scenario_hhis or {}).items():
        yr = int(df["year"].max())
        bars[f"{sc} {yr}"] = contrib(df[df["year"] == yr])

    labels = list(bars.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(labels))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(SUBINDEX_COLS):
        vals = [bars[l][c] for l in labels]
        ax.bar(labels, vals, bottom=bottom, label=SUBINDEX_LABELS[c],
               color=cmap(i / len(SUBINDEX_COLS)))
        bottom += vals

    ax.set_ylabel("Weighted contribution to HHI")
    ax.set_title("Sub-Index Contribution to HHI (mean across wards)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "hhi_subindex_contrib")


def plot_ward_ranking(
    hhi_df: pd.DataFrame, target_year: Optional[int] = None,
    title_suffix: str = "",
) -> str:
    """
    Horizontal bar chart ranking wards by HHI for one year.

    Args:
        hhi_df:       Any HHI results DataFrame (historical or scenario).
        target_year:  Year to rank (defaults to max year in the frame).
        title_suffix: Extra text appended to the plot title.

    Returns:
        Output figure path.
    """
    ty = target_year or int(hhi_df["year"].max())
    # One HHI value per ward for the target year (mean of the four quarters).
    annual = annual_ward_hhi(hhi_df)
    sub = annual[annual["year"] == ty].copy()
    label_col = "upazila" if "upazila" in sub.columns else "ward_id"
    sub = sub.sort_values("HHI")

    colors = [ZONE_COLORS.get(str(z), "#999999")
              for z in sub["vulnerability_zone"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sub[label_col].astype(str), sub["HHI"], color=colors)
    ax.set_xlabel("HHI (0–100)")
    ax.set_xlim(0, 100)
    ax.set_title(f"Ward HHI Ranking — {ty}{title_suffix}")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "hhi_ward_ranking")


def generate_all(
    hist_hhi: pd.DataFrame,
    scenario_hhis: Optional[Dict[str, pd.DataFrame]] = None,
    weights: Optional[dict] = None,
) -> Dict[str, str]:
    """
    Generate the full HHI figure set.

    Args:
        hist_hhi:      Historical HHI results.
        scenario_hhis: Optional scenario HHI results.
        weights:       AHP weights dict (for contribution plot).

    Returns:
        Dict mapping figure key to saved path.
    """
    log.info("Generating HHI figures...")
    out = {
        "timeseries":    plot_hhi_timeseries(hist_hhi, scenario_hhis),
        "zones":         plot_zone_distribution(hist_hhi),
        "contributions": plot_subindex_contribution(hist_hhi, scenario_hhis,
                                                     weights=weights),
        "ranking":       plot_ward_ranking(hist_hhi,
                                           title_suffix=" (Historical)"),
    }
    log.info(f"HHI figures complete: {len(out)} figures.")
    return out


if __name__ == "__main__":
    # Standalone test with synthetic HHI results.
    from src.analysis.hhi_panel import build_historical_panel, build_all_scenarios
    from src.analysis.hhi import HHIComputer
    from src.analysis.ahp import AHP

    weights, _ = AHP().compute()
    hc = HHIComputer(weights=weights)

    hist_panel = build_historical_panel(force_synthetic=True)
    scen_panels = build_all_scenarios(hist_panel)
    bounds = hc.fit_reference_bounds(
        pd.concat([hist_panel] + list(scen_panels.values()), ignore_index=True))

    hist_hhi = hc.compute(hist_panel, ref_bounds=bounds)
    scen_hhi = {k: hc.compute(v, ref_bounds=bounds) for k, v in scen_panels.items()}

    generate_all(hist_hhi, scen_hhi, weights=weights)
    print("HHI plots smoke test passed.")
