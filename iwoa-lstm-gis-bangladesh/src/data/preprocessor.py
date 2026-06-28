"""
preprocessor.py — Full preprocessing pipeline.

Steps:
  1. Merge all data sources on monthly date index
  2. Impute missing values (KNN)
  3. Compute lag features
  4. Detect and handle outliers
  5. Min-Max normalize to [0, 1]
  6. Save feature matrix to data/processed/features/

Run after loader.py validates all sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.loader import load_doe_aqi, load_bwdb, load_census, load_era5, load_chirps

log = get_logger(__name__)


class Preprocessor:
    """
    Full preprocessing pipeline for IWOA-LSTM feature matrix construction.

    Usage:
        prep = Preprocessor()
        feature_matrix = prep.run()
        # Saves to data/processed/features/feature_matrix.csv
    """

    def __init__(self):
        set_seed(cfg.seed)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.imputer = KNNImputer(n_neighbors=cfg.preprocessing.knn_neighbors)
        self.feature_names = cfg.features.pool
        self.target_names = cfg.features.targets

    def build_monthly_index(self) -> pd.DatetimeIndex:
        """
        Build a complete monthly date index from historical_start to historical_end.

        Returns:
            index: DatetimeIndex with monthly frequency, first day of each month.
        """
        start = pd.to_datetime(cfg.dates.historical_start)
        end   = pd.to_datetime(cfg.dates.historical_end)
        index = pd.date_range(start=start, end=end, freq="MS")  # Month Start
        log.info(f"Monthly index: {index[0].date()} → {index[-1].date()} "
                 f"({len(index)} months)")
        return index

    def merge_sources(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Load all data sources and merge onto the monthly date index.
        Missing months become NaN — handled by imputer later.

        Args:
            index: Monthly DatetimeIndex as the merge backbone.

        Returns:
            merged: DataFrame with all available features, indexed by date.
        """
        log.info("Merging all data sources onto monthly index...")

        # Start with the full monthly index
        base = pd.DataFrame(index=index)
        base.index.name = "date"

        # ── DoE AQI ────────────────────────────────────────
        doe = load_doe_aqi()
        if len(doe) > 0:
            # Aggregate to monthly mean per study area
            doe["date"] = pd.to_datetime(doe["date"]).dt.to_period("M").dt.to_timestamp()
            doe_monthly = (
                doe.groupby("date")[["pm25", "no2", "so2", "co", "aqi"]]
                .mean()
                .rename(columns={"aqi": "aqi_score"})
            )
            base = base.join(doe_monthly, how="left")
            log.info(f"DoE merged: {doe_monthly.columns.tolist()}")
        else:
            # Placeholder columns with NaN — pipeline continues
            for col in ["pm25", "no2", "so2", "co", "aqi_score"]:
                base[col] = np.nan
            log.warning("DoE AQI not available — placeholder NaN columns added.")

        # ── BWDB Water Quality ──────────────────────────────
        bwdb = load_bwdb()
        if len(bwdb) > 0:
            bwdb["date"] = pd.to_datetime(bwdb["date"]).dt.to_period("M").dt.to_timestamp()
            bwdb_monthly = (
                bwdb.groupby("date")[["bod", "do", "ph", "turbidity"]]
                .mean()
            )
            base = base.join(bwdb_monthly, how="left")
            log.info(f"BWDB merged: {bwdb_monthly.columns.tolist()}")
        else:
            for col in ["bod", "do", "ph", "turbidity"]:
                base[col] = np.nan
            log.warning("BWDB not available — placeholder NaN columns added.")

        # ── ERA5 Climate ────────────────────────────────────
        era5 = load_era5()
        if len(era5) > 0:
            era5["date"] = pd.to_datetime(era5["date"]).dt.to_period("M").dt.to_timestamp()
            era5 = era5.set_index("date")[["temp_mean", "humidity_mean", "wind_speed"]]
            era5.columns = ["temperature", "humidity", "wind_speed"]
            base = base.join(era5, how="left")
            log.info(f"ERA5 merged: temperature, humidity, wind_speed")
        else:
            for col in ["temperature", "humidity", "wind_speed"]:
                base[col] = np.nan
            log.warning("ERA5 not available — placeholder NaN columns added.")

        # ── CHIRPS Precipitation ────────────────────────────
        chirps = load_chirps()
        if len(chirps) > 0:
            chirps["date"] = pd.to_datetime(chirps["date"]).dt.to_period("M").dt.to_timestamp()
            chirps = chirps.set_index("date")[["precipitation_mm"]]
            chirps.columns = ["precipitation"]
            base = base.join(chirps, how="left")
            log.info("CHIRPS merged: precipitation")
        else:
            base["precipitation"] = np.nan
            log.warning("CHIRPS not available — placeholder NaN column added.")

        # ── Satellite-derived indices ───────────────────────
        # These are filled by src/data/indices.py — placeholder here
        for col in ["ndvi", "ndbi", "mndwi", "lst", "uhi_intensity"]:
            if col not in base.columns:
                base[col] = np.nan

        # ── Industrial and spatial features ────────────────
        # Filled by LULC classification output
        for col in ["industrial_fraction", "pop_density", "dist_industrial"]:
            if col not in base.columns:
                base[col] = np.nan

        log.info(f"Merged DataFrame: {base.shape[0]} rows × {base.shape[1]} columns")
        log.info(f"Missing value summary:\n{base.isna().sum().to_string()}")
        return base

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features as specified in cfg.preprocessing.lag_features.

        Args:
            df: Input feature DataFrame indexed by date.

        Returns:
            df: DataFrame with additional lag columns.
        """
        log.info("Computing lag features...")
        lag_config = cfg.preprocessing.lag_features

        # Convert namespace to dict
        if hasattr(lag_config, "__dict__"):
            lag_dict = vars(lag_config)
        else:
            lag_dict = {
                "pm25": [1, 3, 6],
                "ndvi": [1, 3, 12],
                "lst":  [1, 3],
                "bod":  [1, 3]
            }

        new_cols = []
        for feature, lags in lag_dict.items():
            if feature in df.columns:
                for lag in lags:
                    col_name = f"{feature}_lag{lag}"
                    df[col_name] = df[feature].shift(lag)
                    new_cols.append(col_name)

        log.info(f"Added {len(new_cols)} lag features: {new_cols}")
        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and cap outliers using IQR method.
        Outliers are capped (winsorized) not removed — preserves time series continuity.

        Args:
            df: Input DataFrame.

        Returns:
            df: DataFrame with outliers capped at IQR bounds.
        """
        if cfg.preprocessing.outlier_method == "none":
            return df

        log.info(f"Handling outliers via {cfg.preprocessing.outlier_method} method...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_capped = 0

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - cfg.preprocessing.outlier_threshold * IQR
            upper = Q3 + cfg.preprocessing.outlier_threshold * IQR

            before = df[col].isna().sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            n_capped += (df[col] == lower).sum() + (df[col] == upper).sum()

        log.info(f"Outlier capping complete: {n_capped} values capped.")
        return df

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using KNN imputation.
        Only applied to numeric columns with missing values.

        Args:
            df: DataFrame with potential NaN values.

        Returns:
            df: DataFrame with no NaN values in numeric columns.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_missing_before = df[numeric_cols].isna().sum().sum()

        if n_missing_before == 0:
            log.info("No missing values found — imputation skipped.")
            return df

        log.info(f"Imputing {n_missing_before} missing values using "
                 f"KNN (k={cfg.preprocessing.knn_neighbors})...")

        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        n_missing_after = df[numeric_cols].isna().sum().sum()
        log.info(f"Imputation complete. Missing after: {n_missing_after}")
        return df

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Min-Max normalize all numeric features to [0, 1].
        Saves scaler for inverse transform during forecast output.

        Args:
            df: DataFrame after imputation.

        Returns:
            df_norm:  Normalized DataFrame.
            scaler:   Fitted MinMaxScaler (used for inverse transform later).
        """
        log.info("Normalizing features to [0, 1]...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_norm = df.copy()
        df_norm[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        log.info(f"Normalization complete: {len(numeric_cols)} columns scaled.")
        return df_norm, self.scaler

    def save(self, df: pd.DataFrame, path: Optional[str] = None) -> None:
        """
        Save processed feature matrix to CSV.

        Args:
            df:   Processed DataFrame.
            path: Output file path. Defaults to cfg value.
        """
        path = path or cfg.data.processed.features_file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        log.info(f"Feature matrix saved: {path} ({df.shape[0]} rows × {df.shape[1]} cols)")

    def run(self) -> pd.DataFrame:
        """
        Execute the full preprocessing pipeline end-to-end.

        Returns:
            feature_matrix: Cleaned, normalized, lag-augmented DataFrame.
        """
        log.info("=" * 60)
        log.info("PREPROCESSING PIPELINE START")
        log.info("=" * 60)

        index   = self.build_monthly_index()
        merged  = self.merge_sources(index)
        lagged  = self.add_lag_features(merged)
        cleaned = self.handle_outliers(lagged)
        imputed = self.impute(cleaned)
        normed, scaler = self.normalize(imputed)
        self.save(normed)

        # Save scaler for later inverse transform
        import joblib
        scaler_path = "data/processed/features/scaler.pkl"
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        log.info(f"Scaler saved: {scaler_path}")

        log.info("=" * 60)
        log.info("PREPROCESSING PIPELINE COMPLETE")
        log.info(f"Final feature matrix: {normed.shape}")
        log.info("=" * 60)
        return normed


if __name__ == "__main__":
    prep = Preprocessor()
    feature_matrix = prep.run()
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Columns: {list(feature_matrix.columns)}")
    print(f"\nFirst 3 rows:\n{feature_matrix.head(3)}")
