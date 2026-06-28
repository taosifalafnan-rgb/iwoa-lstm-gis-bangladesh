"""
loader.py — Load raw data from all sources and validate schemas.

Handles:
  - DoE AQI CSVs (data/doe/)
  - BWDB water quality CSVs (data/bwdb/)
  - BBS census CSV (data/census/)
  - ERA5 climate CSV (data/satellite/era5/)
  - CHIRPS precipitation rasters (data/satellite/chirps/)
  - Satellite GeoTIFF rasters (data/satellite/)

Outputs:
  - Validated pandas DataFrames
  - Schema validation report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from glob import glob

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── DOE AQI ────────────────────────────────────────────────────────────────

def load_doe_aqi(doe_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load and concatenate all DoE AQI CSV files from data/doe/.

    Expected columns: date, station, pm25, no2, so2, co, aqi
    Optional columns: pm10, o3

    Args:
        doe_dir: Path to DoE data directory. Defaults to cfg value.

    Returns:
        df: Combined DataFrame sorted by date, missing values as NaN.
    """
    doe_dir = doe_dir or cfg.data.doe_dir
    files = sorted(glob(str(Path(doe_dir) / "*.csv")))

    if not files:
        log.warning(f"No CSV files found in {doe_dir}. "
                    f"Drop DoE AQI files there when available.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip().str.lower()
            dfs.append(df)
            log.info(f"Loaded DoE file: {Path(f).name} — {len(df)} rows")
        except Exception as e:
            log.error(f"Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Validate required columns
    required = cfg.data.doe_columns.required
    missing_cols = [c for c in required if c not in combined.columns]
    if missing_cols:
        log.error(f"DoE data missing required columns: {missing_cols}")
        log.error(f"  Available columns: {list(combined.columns)}")
    else:
        log.info(f"DoE schema validated. Columns OK.")

    # Parse dates
    combined["date"] = pd.to_datetime(
        combined["date"], format=cfg.data.doe_columns.date_format, errors="coerce"
    )
    n_bad_dates = combined["date"].isna().sum()
    if n_bad_dates > 0:
        log.warning(f"DoE: {n_bad_dates} rows have unparseable dates — set to NaT.")

    combined = combined.sort_values("date").reset_index(drop=True)
    log.info(f"DoE AQI loaded: {len(combined)} rows | "
             f"{combined['date'].min()} → {combined['date'].max()}")
    return combined


# ── BWDB WATER QUALITY ─────────────────────────────────────────────────────

def load_bwdb(bwdb_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load and concatenate all BWDB water quality CSV files from data/bwdb/.

    Expected columns: date, station, river, bod, do, ph, turbidity

    Args:
        bwdb_dir: Path to BWDB data directory. Defaults to cfg value.

    Returns:
        df: Combined DataFrame sorted by date.
    """
    bwdb_dir = bwdb_dir or cfg.data.bwdb_dir
    files = sorted(glob(str(Path(bwdb_dir) / "*.csv")))

    if not files:
        log.warning(f"No CSV files found in {bwdb_dir}. "
                    f"Drop BWDB water quality files there when available.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip().str.lower()
            dfs.append(df)
            log.info(f"Loaded BWDB file: {Path(f).name} — {len(df)} rows")
        except Exception as e:
            log.error(f"Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    required = cfg.data.bwdb_columns.required
    missing_cols = [c for c in required if c not in combined.columns]
    if missing_cols:
        log.error(f"BWDB data missing required columns: {missing_cols}")
    else:
        log.info(f"BWDB schema validated. Columns OK.")

    combined["date"] = pd.to_datetime(
        combined["date"], format=cfg.data.bwdb_columns.date_format, errors="coerce"
    )
    combined = combined.sort_values("date").reset_index(drop=True)
    log.info(f"BWDB loaded: {len(combined)} rows | "
             f"{combined['date'].min()} → {combined['date'].max()}")
    return combined


# ── BBS CENSUS ─────────────────────────────────────────────────────────────

def load_census(census_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load BBS ward-level population census data.

    Expected columns: ward_id, district, upazila, year, population, area_km2

    Args:
        census_file: Path to census CSV. Defaults to cfg value.

    Returns:
        df: Census DataFrame with derived population density column.
    """
    census_file = census_file or cfg.data.census_file
    path = Path(census_file)

    if not path.exists():
        log.warning(f"Census file not found at {census_file}. "
                    f"Drop bbs_population_ward.csv in data/census/.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    required = cfg.data.census_columns.required
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        log.error(f"Census data missing required columns: {missing_cols}")
    else:
        log.info(f"Census schema validated. Columns OK.")
        # Derive population density if not present
        if "pop_density" not in df.columns:
            df["pop_density"] = df["population"] / df["area_km2"]
            log.info("Derived pop_density = population / area_km2")

    log.info(f"Census loaded: {len(df)} ward-year records | "
             f"Years: {sorted(df['year'].unique()) if 'year' in df.columns else 'N/A'}")
    return df


# ── ERA5 CLIMATE ────────────────────────────────────────────────────────────

def load_era5(era5_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load ERA5 monthly climate reanalysis data.

    Expected columns: date, temp_mean, humidity_mean, wind_speed

    Args:
        era5_file: Path to ERA5 CSV. Defaults to cfg value.

    Returns:
        df: ERA5 DataFrame sorted by date.
    """
    era5_file = era5_file or cfg.data.era5_file
    path = Path(era5_file)

    if not path.exists():
        log.warning(f"ERA5 file not found at {era5_file}. "
                    f"Drop era5_monthly.csv in data/satellite/era5/.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    log.info(f"ERA5 loaded: {len(df)} monthly records | "
             f"{df['date'].min()} → {df['date'].max()}")
    return df


# ── CHIRPS PRECIPITATION ────────────────────────────────────────────────────

def load_chirps(chirps_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load CHIRPS monthly precipitation data.
    Accepts either pre-aggregated CSV or individual GeoTIFF rasters.
    For rasters, uses zonal mean over the study area bounding box.

    Args:
        chirps_dir: Path to CHIRPS directory. Defaults to cfg value.

    Returns:
        df: DataFrame with columns [date, precipitation_mm].
    """
    chirps_dir = chirps_dir or cfg.data.chirps_dir
    path = Path(chirps_dir)

    # Try CSV first (pre-aggregated)
    csv_files = sorted(path.glob("*.csv"))
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip().str.lower()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        log.info(f"CHIRPS CSV loaded: {len(df)} monthly records")
        return df

    # Try GeoTIFF rasters
    tif_files = sorted(path.glob("*.tif"))
    if not tif_files:
        log.warning(f"No CHIRPS files found in {chirps_dir}.")
        return pd.DataFrame()

    try:
        import rasterio
        import numpy as np
        records = []
        bbox = cfg.study_area.bbox
        for tif in tif_files:
            with rasterio.open(tif) as src:
                # Extract date from filename: chirps_YYYYMM.tif
                stem = tif.stem  # chirps_200001
                parts = stem.split("_")
                if len(parts) >= 2:
                    date_str = parts[-1]  # 200001
                    date = pd.to_datetime(date_str, format="%Y%m")
                else:
                    continue
                data = src.read(1)
                data = np.where(data == src.nodata, np.nan, data)
                mean_precip = np.nanmean(data)
                records.append({"date": date, "precipitation_mm": mean_precip})

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        log.info(f"CHIRPS rasters loaded: {len(df)} monthly records")
        return df

    except ImportError:
        log.error("rasterio not installed. Run: pip install rasterio")
        return pd.DataFrame()


# ── SATELLITE RASTERS ───────────────────────────────────────────────────────

def list_satellite_files(satellite_dir: Optional[str] = None) -> dict:
    """
    Inventory all satellite GeoTIFF files in data/satellite/.
    Does not load them — returns a dict mapping band to file list.

    Args:
        satellite_dir: Path to satellite data directory.

    Returns:
        inventory: Dict mapping band suffix to sorted list of file paths.

    Example:
        inventory["B4"] → ["data/satellite/L5_200001_B4.tif", ...]
    """
    satellite_dir = satellite_dir or cfg.data.satellite_dir
    path = Path(satellite_dir)

    if not path.exists():
        log.warning(f"Satellite directory not found: {satellite_dir}")
        return {}

    tif_files = sorted(path.glob("*.tif"))
    inventory = {}
    for f in tif_files:
        # Parse band from filename: L5_YYYYMM_B4.tif
        parts = f.stem.split("_")
        if len(parts) >= 3:
            band = parts[-1]  # B3, B4, B5, B6, B7, QA_PIXEL, ST_B6
            if band not in inventory:
                inventory[band] = []
            inventory[band].append(str(f))

    total = sum(len(v) for v in inventory.values())
    log.info(f"Satellite inventory: {total} files across {len(inventory)} bands")
    for band, files in inventory.items():
        log.info(f"  Band {band}: {len(files)} files")

    return inventory


# ── DATA AVAILABILITY REPORT ────────────────────────────────────────────────

def run_data_check() -> dict:
    """
    Check which data sources are available and report status.
    Run this first to see what you have before running the pipeline.

    Returns:
        report: Dict with availability status per data source.
    """
    log.info("=" * 60)
    log.info("DATA AVAILABILITY CHECK")
    log.info("=" * 60)

    report = {}

    # DoE
    doe = load_doe_aqi()
    report["doe_aqi"] = {"available": len(doe) > 0, "rows": len(doe)}

    # BWDB
    bwdb = load_bwdb()
    report["bwdb"] = {"available": len(bwdb) > 0, "rows": len(bwdb)}

    # Census
    census = load_census()
    report["census"] = {"available": len(census) > 0, "rows": len(census)}

    # ERA5
    era5 = load_era5()
    report["era5"] = {"available": len(era5) > 0, "rows": len(era5)}

    # CHIRPS
    chirps = load_chirps()
    report["chirps"] = {"available": len(chirps) > 0, "rows": len(chirps)}

    # Satellite
    inventory = list_satellite_files()
    report["satellite"] = {"available": len(inventory) > 0,
                           "bands": list(inventory.keys()),
                           "total_files": sum(len(v) for v in inventory.values())}

    log.info("=" * 60)
    log.info("SUMMARY:")
    for source, info in report.items():
        status = "✓ AVAILABLE" if info["available"] else "✗ MISSING"
        log.info(f"  {source:20s}: {status}")
    log.info("=" * 60)
    log.info("Missing sources will be loaded as empty DataFrames.")
    log.info("Pipeline will run with placeholder NaN values until data is added.")

    return report


if __name__ == "__main__":
    report = run_data_check()
