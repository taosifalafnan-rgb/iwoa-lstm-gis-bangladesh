# iwoa-lstm-gis-bangladesh

**GIS-Based Assessment and IWOA-LSTM Prediction of Industrialization-Induced Environmental Change and Human Habitability in Gazipur and Narayanganj, Bangladesh (2000–2040)**

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/iwoa-lstm-gis-bangladesh.git
cd iwoa-lstm-gis-bangladesh
pip install -r requirements.txt

# Check what data you have
python src/data/loader.py

# Run full pipeline (once data is added)
python run_pipeline.py --config configs/config.yaml

# Run specific steps only
python run_pipeline.py --steps 1 2 3

# Skip IWOA re-run (use saved result)
python run_pipeline.py --skip-iwoa
```

---

## HHI Assessment (standalone, runs without the LSTM/IWOA stages)

The Human Habitability Index can be assessed and created top-to-bottom on its
own. If no real data is present, a reproducible **synthetic demo panel** is
generated so the flow runs immediately.

```bash
python run_hhi_assessment.py                 # full flow → outputs/results, figures, maps
python run_hhi_assessment.py --force-synthetic
python run_hhi_assessment.py --no-viz
python run_hhi_assessment.py --scenarios BAU S2
```

Flow: **AHP weights → raw ward panel → BAU/S1/S2 projection to 2040 →
sub-indices C1–C5 → weighted HHI → vulnerability zones → figures + maps.**
Full technical description in [`docs/hhi_flow.md`](docs/hhi_flow.md).

---

## Data Setup

Drop your data files in the correct folders before running:

| Source | Folder | Format |
|---|---|---|
| DoE Bangladesh AQI | `data/doe/` | `doe_aqi_YYYY.csv` |
| BWDB Water Quality | `data/bwdb/` | `bwdb_waterquality_YYYY.csv` |
| Landsat / Sentinel rasters | `data/satellite/` | `L5_YYYYMM_B{band}.tif` |
| BBS Census | `data/census/` | `bbs_population_ward.csv` |
| CHIRPS Precipitation | `data/satellite/chirps/` | `chirps_YYYYMM.tif` |
| ERA5 Climate | `data/satellite/era5/` | `era5_monthly.csv` |

See `docs/data_schema.md` for exact column requirements.

---

## Pipeline Steps

| Step | Script | Description |
|---|---|---|
| 1 | `src/data/loader.py` | Validate all data sources |
| 2 | `src/data/preprocessor.py` | Clean, impute, normalize |
| 3 | `src/data/sequencer.py` | Build LSTM sequences |
| 4 | `src/optimization/iwoa.py` | Feature selection + HP tuning |
| 5 | `src/models/trainer.py` | Train final LSTM |
| 6 | `src/models/evaluator.py` | Test set evaluation |
| 7 | `src/models/forecaster.py` | 2025–2040 projections |
| 8 | `src/analysis/hhi.py` | HHI computation |
| 9 | `src/visualization/` | Figures and maps |

---

## For Claude Code

Upload `CLAUDE.md` to Claude Code before starting any work on this repo.
It contains all architectural decisions, parameter conventions, and pipeline rules.

---

## Progress Checklist

- [ ] Step 1: Data validated
- [ ] Step 2: Feature matrix built
- [ ] Step 3: Sequences saved
- [x] Step 4: IWOA complete — `src/optimization/iwoa.py` (+ `fitness.py`, baseline `woa.py`)
- [x] Step 5: LSTM trained — `src/models/lstm.py` + `trainer.py`
- [x] Step 6: Evaluation done — `src/models/evaluator.py` (metrics + ARIMA/SVR/GRU/WOA-LSTM)
- [x] Step 7: Forecasts generated — `src/models/forecaster.py` (2025–2040, MC-dropout)
- [x] Step 8: HHI computed — end-to-end flow in `run_hhi_assessment.py` (AHP + panel + scenarios + HHI)
- [x] Step 9: Figures ready — HHI figures/maps in `src/visualization/hhi_plots.py`, `hhi_maps.py`

> Steps 4–7 are implemented and verified end-to-end on synthetic data; they need
> real data + `pip install -r requirements.txt` (PyTorch) to produce final
> results. See [`docs/pipeline_iwoa_lstm.md`](docs/pipeline_iwoa_lstm.md).

---

## Citation

> [Paper title and authors — to be filled after publication]
