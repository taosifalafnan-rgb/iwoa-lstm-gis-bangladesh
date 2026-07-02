"""
forecaster.py — Auto-regressive rollout of the IWOA-LSTM to 2040.

Produces monthly forecasts of the four target variables (pm25, bod, lst, ndvi)
from 2025-01 to 2040-12 under the three policy scenarios (BAU / S1 / S2), with
Monte-Carlo-Dropout uncertainty bands.

How the rollout works
---------------------
The LSTM consumes a window of `lookback` months of the IWOA-selected features
and predicts the next month's targets. To march years into the future we feed
predictions back in — but the input window also contains *exogenous* drivers
(precipitation, industrial fraction, population density, lag features, …) that
are not model outputs. Each future month is therefore assembled from three
kinds of columns:

  1. Target-derived features (pm25, bod, lst, ndvi if selected)
        → filled with the model's own prediction for that month.
  2. Lag features of targets (pm25_lag3, ndvi_lag12, lst_lag1, bod_lag3)
        → filled from the running history buffer at the right offset.
  3. Exogenous drivers (industrial_fraction, pop_density, no2, precipitation …)
        → projected forward with transparent, scenario-dependent rules
          (`_project_exogenous`): industrial/air terms grow with the scenario
          industrial-growth rate and are abated by ETP compliance; water/green
          terms respond to compliance and green-loss; climate variables repeat
          the monthly climatology; anything unmapped simply persists.

Everything happens in the model's normalized [0, 1] space; if a fitted scaler is
supplied the target outputs are additionally inverse-transformed to physical
units (µg/m³, mg/L, °C, index). Uncertainty comes from `EnvironmentalLSTM`'s
MC-Dropout (`predict_with_uncertainty`).

This module is intentionally self-contained and heavily commented: it is the
bridge between the trained model and the HHI scenario inputs.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, List, Dict

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.device import get_device

log = get_logger(__name__)


class Forecaster:
    """
    Auto-regressive multi-scenario forecaster for the trained LSTM.

    Usage:
        fc = Forecaster(model, feature_matrix, iwoa_result, scaler=scaler)
        bau = fc.forecast("BAU")          # DataFrame of monthly forecasts
        allf = fc.forecast_all()          # {scenario: DataFrame}
    """

    def __init__(
        self,
        model,
        feature_matrix: pd.DataFrame,
        iwoa_result,
        scaler=None,
        target_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model:          Trained EnvironmentalLSTM.
            feature_matrix: Normalized, date-indexed historical DataFrame
                            (all pool columns; the same frame used to train).
            iwoa_result:    IWOAResult — supplies selected_features and lookback.
            scaler:         Optional fitted MinMaxScaler (for physical units).
            target_names:   LSTM targets. Defaults to cfg.features.targets.
            device:         Torch device. Auto-detected if None.
        """
        self.device        = device or get_device()
        self.model         = model.to(self.device)
        self.df            = feature_matrix.sort_index()
        self.iwoa          = iwoa_result
        self.scaler        = scaler
        self.target_names  = target_names or cfg.features.targets

        # Selected features define the model input vector order.
        self.selected = list(iwoa_result.selected_features)
        self.lookback = int(iwoa_result.lstm_lookback)

        # Numeric column order the scaler was fitted on (for inverse transform).
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)

        # Forecast horizon from config.
        self.hist_end      = pd.to_datetime(cfg.dates.historical_end)
        self.forecast_end  = pd.to_datetime(cfg.dates.forecast_end)

        # Monthly climatology (mean value per calendar month) for climate drivers.
        self._climatology = self._compute_climatology()

        log.info(f"Forecaster ready: {len(self.selected)} input features, "
                 f"lookback={self.lookback}, targets={self.target_names}, "
                 f"scaler={'yes' if scaler else 'no'}")

    # ── SETUP HELPERS ───────────────────────────────────────────────────────

    def _compute_climatology(self) -> Dict[str, Dict[int, float]]:
        """
        Mean of each column by calendar month (1–12), used to repeat seasonal
        climate for exogenous climate drivers during the forecast.

        Returns:
            {column: {month: mean_value}} for every numeric column.
        """
        clim = {}
        months = self.df.index.month
        for col in self.numeric_cols:
            clim[col] = {m: float(self.df[col][months == m].mean())
                         for m in range(1, 13)}
        return clim

    def _inverse_target(self, norm_value: float, target: str) -> float:
        """
        Inverse-transform a single normalized target value to physical units.

        Uses the MinMaxScaler affine mapping X = (X_scaled − min_)/scale_ for the
        target's column, which is exact and avoids building a full-width row.

        Args:
            norm_value: Normalized prediction in [0, 1].
            target:     Target column name.

        Returns:
            Physical-unit value (or the normalized value if no scaler/column).
        """
        if self.scaler is None or target not in self.numeric_cols:
            return float(norm_value)
        idx = self.numeric_cols.index(target)
        return float((norm_value - self.scaler.min_[idx]) / self.scaler.scale_[idx])

    # ── EXOGENOUS DRIVER PROJECTION ─────────────────────────────────────────

    def _project_exogenous(
        self, feature: str, prev_value: float, month: int,
        step: int, rates: Dict[str, float],
    ) -> float:
        """
        Project one exogenous (non-target, non-lag) selected feature forward
        one month under a scenario.

        Rules (all in normalized space, clipped to [0, 1]):
          - industrial_fraction, ndbi   → grow with industrial growth rate
          - no2, so2, co, uhi_intensity → grow with industry, abate with ETP
          - do, ph                      → improve with ETP compliance
          - turbidity                   → improve with ETP, pressured by growth
          - mndwi                       → mild decline with built-up growth
          - pop_density                 → grow with population rate
          - dist_industrial             → slowly shrink as industry spreads
          - precipitation/temperature/humidity → repeat monthly climatology
          - anything else               → persistence (hold previous value)

        Args:
            feature:    Feature name.
            prev_value: Previous month's normalized value.
            month:      Calendar month (1–12) of the month being generated.
            step:       Months elapsed since forecast start (0-based).
            rates:      Per-month scenario factors (see `_monthly_rates`).

        Returns:
            Projected normalized value in [0, 1].
        """
        g   = rates["ind"]     # per-month industrial growth factor increment
        etp = rates["etp"]     # ETP compliance fraction (0–1)
        gl  = rates["green"]   # per-month green-loss increment
        pg  = rates["pop"]     # per-month population growth increment

        climate = {"precipitation", "temperature", "humidity"}

        if feature in climate:
            # Seasonal climate: repeat the calendar-month climatology.
            return float(np.clip(self._climatology.get(feature, {}).get(month, prev_value), 0, 1))

        if feature in {"industrial_fraction", "ndbi"}:
            return float(np.clip(prev_value * (1 + g), 0, 1))

        if feature in {"no2", "so2", "co", "uhi_intensity"}:
            return float(np.clip(prev_value * (1 + g) * (1 - 0.10 * etp), 0, 1))

        if feature == "do":
            return float(np.clip(prev_value * (1 + 0.10 * etp), 0, 1))
        if feature == "ph":
            # Drift toward neutral (0.5 in normalized space is a rough midpoint).
            return float(np.clip(prev_value + (0.5 - prev_value) * 0.02 * etp, 0, 1))
        if feature == "turbidity":
            return float(np.clip(prev_value * (1 - 0.05 * etp) * (1 + 0.3 * g), 0, 1))

        if feature == "mndwi":
            return float(np.clip(prev_value * (1 - 0.2 * g), 0, 1))
        if feature == "pop_density":
            return float(np.clip(prev_value * (1 + pg), 0, 1))
        if feature == "dist_industrial":
            return float(np.clip(prev_value * (1 - 0.5 * g), 0, 1))

        # Default: persistence.
        return float(prev_value)

    def _monthly_rates(self, scenario: str) -> Dict[str, float]:
        """
        Convert a scenario's annual rates into per-month increments.

        Args:
            scenario: "BAU", "S1", or "S2".

        Returns:
            Dict with per-month 'ind', 'green', 'pop' increments and the
            (unitless) 'etp' compliance fraction.
        """
        r = getattr(cfg.scenarios, scenario)
        to_month = lambda annual: (1 + annual) ** (1 / 12) - 1
        return {
            "ind":   to_month(r.industrial_growth_rate),
            "green": to_month(r.green_loss_rate),
            "pop":   to_month(r.pop_growth_rate),
            "etp":   r.etp_compliance_rate,
        }

    # ── CORE ROLLOUT ────────────────────────────────────────────────────────

    def forecast(self, scenario: str, use_mc_dropout: bool = True) -> pd.DataFrame:
        """
        Auto-regressively roll the model forward to `forecast_end` for one scenario.

        Args:
            scenario:       "BAU", "S1", or "S2".
            use_mc_dropout: If True, use MC-Dropout to add lower/upper bands.

        Returns:
            DataFrame with columns: date, scenario, <target> (physical units) and
            <target>_lower / <target>_upper when MC dropout is on. Also keeps the
            normalized predictions as <target>_norm for downstream use.
        """
        log.info(f"Forecasting scenario '{scenario}' "
                 f"{self.hist_end.year + 1}–{self.forecast_end.year} ...")
        rates = self._monthly_rates(scenario)

        # Seed the input window with the last `lookback` months of selected
        # features (normalized). Shape: [lookback, n_selected].
        sel_hist = self.df[self.selected].tail(self.lookback).values.astype(float)
        window = list(sel_hist)   # list of feature vectors, we slide this

        # History buffers for target lag features (normalized), seeded from data.
        # Keyed by target name → list of past values (oldest first).
        lag_hist = {t: list(self.df[t].values.astype(float))
                    for t in self.target_names if t in self.df.columns}

        # Build the future month index.
        future_dates = pd.date_range(
            self.hist_end + pd.offsets.MonthBegin(1),
            self.forecast_end, freq="MS")

        rows = []
        for step, date in enumerate(future_dates):
            # ── 1. Predict next month's targets from the current window ──
            x = torch.FloatTensor(np.array(window)[None, :, :]).to(self.device)

            if use_mc_dropout and hasattr(self.model, "predict_with_uncertainty"):
                mean, lower, upper = self.model.predict_with_uncertainty(x)
                pred = mean[0]          # [n_targets]
                lo, hi = lower[0], upper[0]
            else:
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(x).cpu().numpy()[0]
                lo = hi = pred

            pred = np.clip(pred, 0, 1)   # keep normalized targets in-range

            # ── 2. Record physical-unit forecast for this month ──
            row = {"date": date, "scenario": scenario}
            for j, t in enumerate(self.target_names):
                row[f"{t}_norm"]  = float(pred[j])
                row[t]            = self._inverse_target(pred[j], t)
                if use_mc_dropout:
                    row[f"{t}_lower"] = self._inverse_target(np.clip(lo[j], 0, 1), t)
                    row[f"{t}_upper"] = self._inverse_target(np.clip(hi[j], 0, 1), t)
            rows.append(row)

            # ── 3. Update lag history with the new predictions ──
            for j, t in enumerate(self.target_names):
                if t in lag_hist:
                    lag_hist[t].append(float(pred[j]))

            # ── 4. Assemble the NEXT feature vector for the sliding window ──
            prev_vec = window[-1]
            next_vec = np.zeros(len(self.selected))
            target_idx = {t: j for j, t in enumerate(self.target_names)}

            for k, feat in enumerate(self.selected):
                if feat in target_idx:
                    # (1) target-derived feature → model prediction
                    next_vec[k] = pred[target_idx[feat]]
                elif feat.endswith(tuple(f"_lag{n}" for n in range(1, 25))):
                    # (2) lag feature of a target → look back in the buffer
                    next_vec[k] = self._lag_value(feat, lag_hist, prev_vec[k])
                else:
                    # (3) exogenous driver → scenario projection
                    next_vec[k] = self._project_exogenous(
                        feat, float(prev_vec[k]), date.month, step, rates)

            # Slide the window forward by one month.
            window.append(next_vec)
            window.pop(0)

        df = pd.DataFrame(rows)
        df["year"] = df["date"].dt.year
        log.info(f"  scenario '{scenario}': {len(df)} monthly forecasts produced.")
        self._save(df, scenario)
        return df

    def _lag_value(self, feat: str, lag_hist: Dict[str, list], fallback: float) -> float:
        """
        Resolve a target lag feature (e.g. 'pm25_lag3') from the history buffer.

        Args:
            feat:     Lag feature name '<target>_lag<n>'.
            lag_hist: {target: [values...]} running history (oldest first).
            fallback: Value to use if the lag cannot be resolved.

        Returns:
            The target's value n months ago (normalized), or fallback.
        """
        try:
            base, lag = feat.rsplit("_lag", 1)
            lag = int(lag)
            hist = lag_hist.get(base)
            if hist is not None and len(hist) >= lag:
                return float(hist[-lag])
        except Exception:
            pass
        return float(fallback)

    # ── DRIVERS ─────────────────────────────────────────────────────────────

    def forecast_all(self, scenarios: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Forecast every scenario.

        Args:
            scenarios: Scenario names. Defaults to BAU, S1, S2.

        Returns:
            {scenario: forecast DataFrame}.
        """
        scenarios = scenarios or ["BAU", "S1", "S2"]
        return {sc: self.forecast(sc) for sc in scenarios}

    def _save(self, df: pd.DataFrame, scenario: str) -> str:
        """Persist a scenario forecast to the processed scenarios directory."""
        out_dir = Path(cfg.data.processed.scenarios_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"forecast_{scenario}.csv"
        df.to_csv(path, index=False)
        log.info(f"  forecast saved: {path}")
        return str(path)


if __name__ == "__main__":
    # Standalone smoke test with a synthetic model + feature matrix.
    # (No training — just verifies the rollout mechanics and shapes.)
    import numpy as np
    from src.models.lstm import EnvironmentalLSTM
    from src.optimization.iwoa import IWOAResult
    from src.utils.seed import set_seed

    log.info("Forecaster smoke test...")
    set_seed(cfg.seed)

    cols = cfg.features.pool
    dates = pd.date_range("2000-01-01", periods=288, freq="MS")
    fm = pd.DataFrame(np.random.rand(288, len(cols)), index=dates, columns=cols)

    selected = ["pm25", "bod", "lst", "ndvi", "no2", "industrial_fraction",
                "pop_density", "precipitation", "pm25_lag3"]
    result = IWOAResult(selected_features=selected, n_selected=len(selected),
                        lstm_lookback=12)

    model = EnvironmentalLSTM(input_size=len(selected), hidden_1=32,
                              hidden_2=16, output_size=len(cfg.features.targets))

    fc = Forecaster(model, fm, result, scaler=None)
    out = fc.forecast("BAU", use_mc_dropout=False)
    log.info(f"Forecast shape: {out.shape}, cols={list(out.columns)}")
    log.info("Forecaster smoke test passed.")
