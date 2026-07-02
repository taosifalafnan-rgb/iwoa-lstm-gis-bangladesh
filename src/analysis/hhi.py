"""
hhi.py — Human Habitability Index (HHI) computation.

Formula:
    HHI = Σ wᵢ · Cᵢ    where each Cᵢ ∈ [0, 100]

Sub-indices:
    C1 — Air Quality    (PM2.5, NO2, SO2, CO)
    C2 — Water Quality  (BOD, DO, pH, turbidity)
    C3 — Thermal Stress (LST, UHI intensity)
    C4 — Green Cover    (NDVI, green area loss)
    C5 — Socioeconomic  (pop density, industrial proximity)

Convention: higher HHI = worse habitability
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.analysis.ahp import AHP

log = get_logger(__name__)

# ── Sub-index normalization reference constants ─────────────────────────────
# These convert raw physical measurements to unit-less stress scores [0, 1]
# before AHP weighting. They are documented here (not in config) because they
# encode fixed physical/geographic reference bands, mirroring the existing
# convention in this module (e.g. UHI /6°C, pop_density /50000).
LST_REF_MIN_C   = 20.0    # °C — comfortable land-surface temperature floor
LST_REF_MAX_C   = 45.0    # °C — extreme heat-stress ceiling for the region
UHI_REF_MAX_C   = 6.0     # °C — typical max urban-heat-island intensity (BD)
POP_DENSITY_MAX = 50000.0 # persons/km² — dense Bangladeshi ward ceiling
LST_NORM_HINT   = 1.5     # if LST max <= this, values are already in [0, 1]


class HHIComputer:
    """
    Computes ward-level Human Habitability Index from LSTM outputs
    and spatial data.

    Usage:
        hhi = HHIComputer()
        hhi_df = hhi.compute(predictions_df, ward_df)
    """

    def __init__(self, weights: Optional[dict] = None):
        """
        Args:
            weights: AHP-derived weights dict. If None, runs AHP automatically.
        """
        if weights is None:
            ahp = AHP()
            self.weights, _ = ahp.compute()
        else:
            self.weights = weights

        self.thresholds = cfg.hhi.thresholds
        log.info(f"HHIComputer initialized with weights: {self.weights}")

    # ── SUB-INDEX COMPUTATIONS ──────────────────────────────────────────────

    def compute_c1_air(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute air quality sub-index C1.
        Higher value = worse air quality.

        Args:
            df: DataFrame with columns [pm25, no2, so2, co].

        Returns:
            c1_raw: Raw air quality score per row.
        """
        w = vars(cfg.hhi.c1_air_weights) if hasattr(cfg.hhi.c1_air_weights, '__dict__') \
            else {"pm25": 0.40, "no2": 0.25, "so2": 0.20, "co": 0.15}

        t = self.thresholds

        pm25_norm = df.get("pm25", pd.Series(0, index=df.index)) / \
                    (getattr(t, 'pm25_24hr_naaqs', 65.0))
        no2_norm  = df.get("no2",  pd.Series(0, index=df.index)) / \
                    (getattr(t, 'no2_24hr_naaqs', 80.0))
        so2_norm  = df.get("so2",  pd.Series(0, index=df.index)) / \
                    (getattr(t, 'so2_24hr_naaqs', 80.0))
        co_norm   = df.get("co",   pd.Series(0, index=df.index)) / \
                    (getattr(t, 'co_8hr_naaqs', 5.0))

        c1_raw = (w.get("pm25", 0.40) * pm25_norm +
                  w.get("no2",  0.25) * no2_norm  +
                  w.get("so2",  0.20) * so2_norm  +
                  w.get("co",   0.15) * co_norm)

        return c1_raw.clip(0, None)

    def compute_c2_water(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute water quality sub-index C2.
        Higher value = worse water quality.

        Args:
            df: DataFrame with columns [bod, do, ph, turbidity].

        Returns:
            c2_raw: Raw water quality score per row.
        """
        w = vars(cfg.hhi.c2_water_weights) if hasattr(cfg.hhi.c2_water_weights, '__dict__') \
            else {"bod": 0.35, "do": 0.35, "ph": 0.15, "turbidity": 0.15}

        t = self.thresholds
        bod_max   = getattr(t, 'bod_max', 6.0)
        do_sat    = getattr(t, 'do_saturation', 8.0)
        turb_max  = getattr(t, 'turbidity_max', 10.0)

        bod_norm  = df.get("bod", pd.Series(0, index=df.index)) / bod_max
        do_norm   = 1 - (df.get("do", pd.Series(do_sat, index=df.index)) / do_sat)
        ph_norm   = (df.get("ph", pd.Series(7, index=df.index)) - 7).abs() / 3.0
        turb_norm = df.get("turbidity", pd.Series(0, index=df.index)) / turb_max

        c2_raw = (w.get("bod",       0.35) * bod_norm  +
                  w.get("do",        0.35) * do_norm   +
                  w.get("ph",        0.15) * ph_norm   +
                  w.get("turbidity", 0.15) * turb_norm)

        return c2_raw.clip(0, None)

    def compute_c3_thermal(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute thermal stress sub-index C3.
        Higher value = more thermal stress.

        Args:
            df: DataFrame with columns [lst, uhi_intensity].

        Returns:
            c3_raw: Raw thermal stress score per row.
        """
        w = vars(cfg.hhi.c3_thermal_weights) if hasattr(cfg.hhi.c3_thermal_weights, '__dict__') \
            else {"lst": 0.60, "uhi_intensity": 0.40}

        lst  = df.get("lst", pd.Series(0.5, index=df.index))
        uhi  = df.get("uhi_intensity", pd.Series(0, index=df.index))

        # LST may arrive either already normalized to [0, 1] (from the LSTM
        # feature matrix) or as raw land-surface temperature in °C (from the
        # raw HHI panel). Auto-detect and normalize raw °C against the regional
        # reference band so higher temperature → higher thermal stress.
        if float(lst.max()) > LST_NORM_HINT:
            lst_norm = ((lst - LST_REF_MIN_C) /
                        (LST_REF_MAX_C - LST_REF_MIN_C)).clip(0, 1)
        else:
            lst_norm = lst.clip(0, 1)

        # UHI: normalize by max observed (6°C typical for Bangladesh industrial zones)
        uhi_norm = (uhi / UHI_REF_MAX_C).clip(0, 1)

        c3_raw = (w.get("lst",           0.60) * lst_norm +
                  w.get("uhi_intensity", 0.40) * uhi_norm)

        return c3_raw.clip(0, None)

    def compute_c4_green(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute green cover sub-index C4.
        Higher value = less green = worse habitability.

        Args:
            df: DataFrame with columns [ndvi, green_loss_fraction].

        Returns:
            c4_raw: Raw green cover degradation score.
        """
        w = vars(cfg.hhi.c4_green_weights) if hasattr(cfg.hhi.c4_green_weights, '__dict__') \
            else {"ndvi": 0.70, "green_loss": 0.30}

        ndvi       = df.get("ndvi", pd.Series(0.5, index=df.index))
        green_loss = df.get("green_loss_fraction", pd.Series(0, index=df.index))

        # Invert NDVI: low NDVI = high degradation
        ndvi_score = (1 - ndvi.clip(0, 1))
        loss_score = green_loss.clip(0, 1)

        c4_raw = (w.get("ndvi",       0.70) * ndvi_score +
                  w.get("green_loss", 0.30) * loss_score)

        return c4_raw.clip(0, None)

    def compute_c5_socio(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute socioeconomic pressure sub-index C5.
        Higher value = more pressure = worse habitability.

        Args:
            df: DataFrame with columns [pop_density, dist_industrial, industrial_fraction].

        Returns:
            c5_raw: Raw socioeconomic stress score.
        """
        w = vars(cfg.hhi.c5_socio_weights) if hasattr(cfg.hhi.c5_socio_weights, '__dict__') \
            else {"pop_density": 0.40, "dist_industrial_inv": 0.35, "industrial_fraction": 0.25}

        # Normalize population density (max ~50,000/km² for dense BD wards)
        pop_norm  = (df.get("pop_density", pd.Series(0, index=df.index)) / POP_DENSITY_MAX).clip(0, 1)

        # Industrial proximity: closer = worse (inverse distance)
        dist = df.get("dist_industrial", pd.Series(10.0, index=df.index))
        dist_norm = (1 / (dist + 1)).clip(0, 1)

        ind_frac = df.get("industrial_fraction", pd.Series(0, index=df.index)).clip(0, 1)

        c5_raw = (w.get("pop_density",         0.40) * pop_norm  +
                  w.get("dist_industrial_inv",  0.35) * dist_norm +
                  w.get("industrial_fraction",  0.25) * ind_frac)

        return c5_raw.clip(0, None)

    # ── NORMALIZATION ───────────────────────────────────────────────────────

    def normalize_subindex(
        self,
        series: pd.Series,
        global_min: float = None,
        global_max: float = None
    ) -> pd.Series:
        """
        Normalize sub-index to 0-100 scale using min-max normalization.

        Args:
            series:     Raw sub-index values.
            global_min: Override minimum for consistent scaling across time.
            global_max: Override maximum for consistent scaling across time.

        Returns:
            normalized: Sub-index scaled to [0, 100].
        """
        vmin = global_min if global_min is not None else series.min()
        vmax = global_max if global_max is not None else series.max()

        if vmax - vmin < 1e-9:
            return pd.Series(50.0, index=series.index)  # Constant → mid-scale

        return ((series - vmin) / (vmax - vmin) * 100).clip(0, 100)

    # ── VULNERABILITY CLASSIFICATION ────────────────────────────────────────

    def classify_vulnerability(self, hhi: pd.Series) -> pd.Series:
        """
        Classify HHI scores into vulnerability zones.

        Args:
            hhi: HHI scores [0-100].

        Returns:
            zones: Categorical Series with zone labels.
        """
        thresholds = vars(cfg.hhi.vulnerability_zones) if \
                     hasattr(cfg.hhi.vulnerability_zones, '__dict__') else \
                     {"acceptable": [0,25], "moderate": [25,50],
                      "at_risk": [50,75], "critical": [75,100]}

        zones = pd.cut(
            hhi,
            bins=[0, 25, 50, 75, 100],
            labels=["Acceptable", "Moderate", "At-Risk", "Critical"],
            include_lowest=True
        )
        return zones

    # ── RAW SUB-INDICES ─────────────────────────────────────────────────────

    def raw_subindices(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Compute the five raw (un-normalized) sub-index stress scores.

        Kept separate from `compute` so a caller can pool raw scores across
        several DataFrames (e.g. historical + BAU/S1/S2 scenarios) and fit a
        single shared normalization scale via `fit_reference_bounds`.

        Args:
            df: DataFrame with the required environmental columns.

        Returns:
            Dict mapping sub-index name to its raw pd.Series.
        """
        return {
            "C1_air":     self.compute_c1_air(df),
            "C2_water":   self.compute_c2_water(df),
            "C3_thermal": self.compute_c3_thermal(df),
            "C4_green":   self.compute_c4_green(df),
            "C5_socio":   self.compute_c5_socio(df),
        }

    def fit_reference_bounds(
        self, df: pd.DataFrame
    ) -> Dict[str, Tuple[float, float]]:
        """
        Fit per-sub-index (min, max) bounds from raw scores, for reuse across
        multiple DataFrames so their normalized HHI values stay comparable.

        Args:
            df: Pooled DataFrame (e.g. historical + all scenarios concatenated).

        Returns:
            Dict mapping sub-index name to (min, max) tuple.
        """
        raw = self.raw_subindices(df)
        bounds = {k: (float(v.min()), float(v.max())) for k, v in raw.items()}
        log.info(f"Fitted shared HHI reference bounds: "
                 f"{ {k: (round(a,3), round(b,3)) for k,(a,b) in bounds.items()} }")
        return bounds

    # ── MAIN COMPUTE ────────────────────────────────────────────────────────

    def compute(
        self,
        df: pd.DataFrame,
        time_col: str = "date",
        ward_col: str = "ward_id",
        ref_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> pd.DataFrame:
        """
        Compute HHI for all rows in df.

        Args:
            df:         DataFrame with all required environmental columns.
            time_col:   Column name for time dimension.
            ward_col:   Column name for spatial unit identifier.
            ref_bounds: Optional shared (min, max) bounds per sub-index from
                        `fit_reference_bounds`. If None, each sub-index is
                        min-max normalized using this DataFrame alone.

        Returns:
            result: DataFrame with HHI score, sub-indices, and vulnerability zone.
        """
        log.info(f"Computing HHI for {len(df)} records...")

        # Preserve identifier columns (and any spatial metadata) when present.
        id_cols = [c for c in [time_col, "year", ward_col, "district", "upazila",
                               "lat", "lon", "scenario"] if c in df.columns]
        result = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)

        # Compute raw sub-indices
        raw = self.raw_subindices(df)
        rb = ref_bounds or {}

        # Normalize each to 0-100 (shared bounds if provided, else per-frame)
        for name, series in raw.items():
            lo, hi = rb.get(name, (None, None))
            result[name] = self.normalize_subindex(series, lo, hi)

        # Weighted HHI
        result["HHI"] = (
            self.weights.get("air_quality",    0.384) * result["C1_air"]     +
            self.weights.get("water_quality",  0.241) * result["C2_water"]   +
            self.weights.get("thermal_stress", 0.170) * result["C3_thermal"] +
            self.weights.get("green_cover",    0.108) * result["C4_green"]   +
            self.weights.get("socioeconomic",  0.097) * result["C5_socio"]
        )

        result["vulnerability_zone"] = self.classify_vulnerability(result["HHI"])

        # Summary statistics
        log.info(f"HHI Summary:")
        log.info(f"  Mean:     {result['HHI'].mean():.2f}")
        log.info(f"  Min:      {result['HHI'].min():.2f}")
        log.info(f"  Max:      {result['HHI'].max():.2f}")
        log.info(f"  Zone distribution:")
        for zone, count in result["vulnerability_zone"].value_counts().items():
            pct = 100 * count / len(result)
            log.info(f"    {zone}: {count} ({pct:.1f}%)")

        return result

    def save(self, hhi_df: pd.DataFrame, scenario: str = "historical") -> str:
        """
        Save HHI results to CSV.

        Args:
            hhi_df:   HHI DataFrame.
            scenario: Label for filename (historical, BAU, S1, S2).

        Returns:
            path: Output file path.
        """
        Path("outputs/results").mkdir(parents=True, exist_ok=True)
        path = f"outputs/results/hhi_{scenario}.csv"
        hhi_df.to_csv(path, index=False)
        log.info(f"HHI saved: {path}")
        return path


if __name__ == "__main__":
    log.info("HHI smoke test with synthetic data...")

    n = 100
    synthetic = pd.DataFrame({
        "date":                np.repeat(pd.date_range("2000-01", periods=10, freq="YS"), 10),
        "ward_id":             np.tile(np.arange(10), 10),
        "pm25":                np.random.uniform(20, 120, n),
        "no2":                 np.random.uniform(10, 90, n),
        "so2":                 np.random.uniform(5, 70, n),
        "co":                  np.random.uniform(1, 8, n),
        "bod":                 np.random.uniform(1, 15, n),
        "do":                  np.random.uniform(2, 8, n),
        "ph":                  np.random.uniform(5.5, 9.0, n),
        "turbidity":           np.random.uniform(2, 25, n),
        "lst":                 np.random.uniform(0.2, 0.9, n),
        "uhi_intensity":       np.random.uniform(0, 6, n),
        "ndvi":                np.random.uniform(0.1, 0.6, n),
        "green_loss_fraction": np.random.uniform(0, 0.4, n),
        "pop_density":         np.random.uniform(5000, 40000, n),
        "dist_industrial":     np.random.uniform(0.5, 15, n),
        "industrial_fraction": np.random.uniform(0, 0.5, n),
    })

    hhi_computer = HHIComputer()
    result = hhi_computer.compute(synthetic)
    hhi_computer.save(result, scenario="test")

    print(f"\nHHI sample:\n{result[['date','ward_id','HHI','vulnerability_zone']].head(10)}")
    log.info("HHI smoke test passed.")
