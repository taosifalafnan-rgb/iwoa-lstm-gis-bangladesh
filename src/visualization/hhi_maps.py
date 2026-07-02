"""
hhi_maps.py — Spatial HHI maps for the study area.

Renders ward-level HHI as a spatial map over the study-area bounding box.
Ward polygons require a shapefile that is not shipped with the repo, so this
module uses a dependency-light representation: ward centroids plotted as a
colored/sized scatter over the bounding box, annotated with upazila names.
When a real ward shapefile is available and geopandas is installed, the
`try_choropleth` helper will draw a true choropleth instead.

Config-driven: `cfg.visualization.map_dir`, `dpi`, `colormap_hhi`, and the
study-area bbox from `cfg.study_area.bbox`.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def _map_dir() -> Path:
    """Return (and create) the configured map output directory."""
    d = Path(cfg.visualization.map_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def map_hhi(
    hhi_df: pd.DataFrame,
    year: Optional[int] = None,
    scenario: str = "historical",
) -> str:
    """
    Render a ward-centroid HHI map for a single year/scenario.

    Args:
        hhi_df:   HHI results with lon/lat and HHI columns.
        year:     Year to map (defaults to max year in the frame).
        scenario: Label used in the title and output filename.

    Returns:
        Output map path.
    """
    if not {"lon", "lat"}.issubset(hhi_df.columns):
        log.warning("map_hhi: lon/lat columns absent — skipping map.")
        return ""

    yr = year or int(hhi_df["year"].max())
    sub = hhi_df[hhi_df["year"] == yr].copy()
    bbox = cfg.study_area.bbox
    cmap = cfg.visualization.colormap_hhi        # RdYlGn_r: red = critical
    norm = Normalize(vmin=0, vmax=100)           # fixed HHI scale for comparability

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(sub["lon"], sub["lat"], c=sub["HHI"], cmap=cmap, norm=norm,
               s=650, edgecolors="black", linewidths=0.8, zorder=3)

    # Annotate each ward with its name and HHI value.
    label_col = "upazila" if "upazila" in sub.columns else "ward_id"
    for _, r in sub.iterrows():
        ax.annotate(f"{r[label_col]}\n{r['HHI']:.0f}",
                    (r["lon"], r["lat"]), ha="center", va="center",
                    fontsize=7, zorder=4)

    # Separate the two districts visually with a reference line at ~23.85 N.
    ax.axhline(23.85, color="gray", ls=":", lw=1, alpha=0.6)
    ax.text(bbox.xmin + 0.01, 24.05, "GAZIPUR", fontsize=11, weight="bold",
            color="gray", alpha=0.7)
    ax.text(bbox.xmin + 0.01, 23.58, "NARAYANGANJ", fontsize=11, weight="bold",
            color="gray", alpha=0.7)

    ax.set_xlim(bbox.xmin, bbox.xmax)
    ax.set_ylim(bbox.ymin, bbox.ymax)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Human Habitability Index — {scenario} {yr}")
    ax.set_aspect("equal", adjustable="box")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("HHI (0 = habitable, 100 = critical)")

    ext = cfg.visualization.figure_format
    path = _map_dir() / f"hhi_{scenario}_{yr}.{ext}"
    fig.savefig(path, dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Map saved: {path}")
    return str(path)


def try_choropleth(hhi_df: pd.DataFrame, shapefile: str,
                   year: Optional[int] = None, scenario: str = "historical") -> str:
    """
    Draw a true ward choropleth if geopandas and a ward shapefile are available.

    Args:
        hhi_df:    HHI results with a `ward_id` column.
        shapefile: Path to a ward-boundary shapefile with a matching id field.
        year:      Year to map (defaults to max year).
        scenario:  Label for title/filename.

    Returns:
        Output map path, or "" if geopandas/shapefile unavailable.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        log.info("geopandas not installed — using centroid map instead.")
        return map_hhi(hhi_df, year, scenario)

    if not Path(shapefile).exists():
        log.info(f"Shapefile {shapefile} not found — using centroid map.")
        return map_hhi(hhi_df, year, scenario)

    import geopandas as gpd
    yr = year or int(hhi_df["year"].max())
    gdf = gpd.read_file(shapefile)
    merged = gdf.merge(hhi_df[hhi_df["year"] == yr], on="ward_id", how="left")

    fig, ax = plt.subplots(figsize=(9, 8))
    merged.plot(column="HHI", cmap=cfg.visualization.colormap_hhi, vmin=0,
                vmax=100, legend=True, edgecolor="black", ax=ax)
    ax.set_title(f"Human Habitability Index — {scenario} {yr}")

    ext = cfg.visualization.figure_format
    path = _map_dir() / f"hhi_choropleth_{scenario}_{yr}.{ext}"
    fig.savefig(path, dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Choropleth saved: {path}")
    return str(path)


def generate_all(hist_hhi: pd.DataFrame, scenario_hhis: Optional[dict] = None) -> dict:
    """
    Generate the standard HHI map set: latest historical + each scenario's
    2040 endpoint.

    Args:
        hist_hhi:      Historical HHI results.
        scenario_hhis: Optional {scenario: HHI DataFrame}.

    Returns:
        Dict mapping map key to saved path.
    """
    log.info("Generating HHI maps...")
    out = {"historical": map_hhi(hist_hhi, scenario="historical")}
    for sc, df in (scenario_hhis or {}).items():
        out[sc] = map_hhi(df, year=int(df["year"].max()), scenario=sc)
    log.info(f"HHI maps complete: {len(out)} maps.")
    return out


if __name__ == "__main__":
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
    generate_all(hist_hhi, scen_hhi)
    print("HHI maps smoke test passed.")
