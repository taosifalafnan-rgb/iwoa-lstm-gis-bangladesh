# HHI Assessment — Technical Flow

This document describes the end-to-end flow for **assessing and creating the
Human Habitability Index (HHI)** for Gazipur and Narayanganj. It is the
standalone analysis path (Steps 9–12 of the master pipeline) and can run
**without** the LSTM/IWOA deep-learning stages.

Run it with:

```bash
python run_hhi_assessment.py                 # full flow (synthetic panel if no real data)
python run_hhi_assessment.py --force-synthetic
python run_hhi_assessment.py --no-viz
python run_hhi_assessment.py --scenarios BAU S2
```

---

## 1. Conceptual model

The HHI aggregates five environmental sub-indices, each scaled to `[0, 100]`
(higher = **less** habitable), weighted by AHP-derived criteria weights:

```
HHI = w1·C1_air + w2·C2_water + w3·C3_thermal + w4·C4_green + w5·C5_socio
```

| Sub-index | Drivers (raw units) | Reference standard |
|-----------|---------------------|--------------------|
| **C1 Air** | PM2.5, NO2, SO2 (μg/m³), CO (mg/m³) | Bangladesh NAAQS (Air Pollution Rules 2022) |
| **C2 Water** | BOD, DO (mg/L), pH, turbidity (NTU) | ECR 1997 Class-C thresholds |
| **C3 Thermal** | LST (°C), UHI intensity (°C) | Regional 20–45 °C reference band, UHI /6 °C |
| **C4 Green** | NDVI, green-loss fraction | NDVI inverted (low NDVI = high stress) |
| **C5 Socio** | pop density, dist-to-industry, industrial fraction | /50 000 persons·km⁻², inverse distance |

All thresholds live in `configs/config.yaml → hhi`; the weights come from AHP.

---

## 2. Data contract — raw vs normalized

> **Key design point.** The HHI consumes a **raw physical panel** (PM2.5 in
> μg/m³, BOD in mg/L, LST in °C …), *not* the `[0, 1]`-normalized LSTM feature
> matrix from `src/data/preprocessor.py`. Normalizing before threshold
> comparison would collapse the NAAQS/ECR reference logic. The HHI therefore
> has its own panel builder (`src/analysis/hhi_panel.py`).

The panel carries a **spatial dimension** (`ward_id`, `district`, `upazila`,
`lon`, `lat`) so HHI is reported per ward — required for the choropleth maps
and hotspot narrative.

**Temporal resolution: quarterly.** Data is collected at **4 sampling points
per year** — January, April, July, October (`month` / `period` columns) — over
2000–2024. That is 10 wards × 25 years × 4 quarters = **1 000 historical rows**.
Ward-level visuals (maps, ranking, zone composition) average the four quarters
into one value per ward per year via `hhi.annual_ward_hhi`; the time-series plot
uses the annual mean. A blank data-entry template with every ID row pre-filled
lives at `templates/hhi_panel_template.csv` — regenerate it with
`python -c "from src.analysis.hhi_panel import write_template; write_template()"`.

---

## 3. Stage-by-stage flow

```
run_hhi_assessment.py
│
├─ Stage 0  set_seed(cfg.seed)                         reproducibility
│
├─ Stage 1  AHP().compute()                            src/analysis/ahp.py
│             pairwise matrix → priority vector → CR
│             ✗ raises if CR ≥ 0.10 (inconsistent)
│
├─ Stage 2  build_historical_panel()                   src/analysis/hhi_panel.py
│             real panel on disk?  → load
│             else                 → synthetic demo panel (deterministic)
│             → data/processed/features/hhi_panel_raw.csv
│
├─ Stage 3  build_all_scenarios(hist)                  src/analysis/hhi_panel.py
│             project 2025–2040 from 2024 state using
│             cfg.scenarios rates (growth / ETP / green-loss / pop)
│             → data/processed/scenarios/panel_{BAU,S1,S2}.csv
│
├─ Stage 4  HHIComputer.fit_reference_bounds(pooled)   src/analysis/hhi.py
│             one shared 0–100 scale across hist + all scenarios
│             so trajectories are directly comparable
│
├─ Stage 5  HHIComputer.compute(panel, ref_bounds)     src/analysis/hhi.py
│             raw sub-indices C1–C5 → normalize → weighted HHI
│             → classify_vulnerability() into 4 zones
│
├─ Stage 6  HHIComputer.save(...)                       outputs/results/
│             hhi_historical.csv, hhi_forecast_{BAU,S1,S2}.csv
│
├─ Stage 7  hhi_plots.generate_all()                    outputs/figures/
│           hhi_maps.generate_all()                     outputs/maps/
│
└─ Stage 8  console + run_log.txt summary
```

---

## 4. Scenario engine

Projection starts from each ward's **2024** state and steps forward one year at
a time (`src/analysis/hhi_panel.py → project_scenario_panel`):

| Driver | Rule | Effect |
|--------|------|--------|
| Air (PM2.5/NO2/SO2/CO) | `× (1+g)·(1 − 0.10·etp)` | grows with industry, abated by ETP compliance |
| Water (BOD/turbidity) | `× (1 − 0.15·etp)·(1 + 0.3·g)` | improves with compliance |
| DO | `× (1 + 0.10·etp)` | recovers with compliance |
| Thermal (LST/UHI) | additive `+3g`, `×(1+g)(1−0.1·etp)` | warms with industry |
| Green (NDVI) | `× (1 − green_loss_rate)` | erodes at scenario loss rate (S2 = 0) |
| Population density | `× (1 + pop_growth_rate)` | rises |

Rates by scenario (`configs/config.yaml → scenarios`):

| Scenario | industrial_growth | etp_compliance | green_loss | pop_growth |
|----------|------------------:|---------------:|-----------:|-----------:|
| **BAU** (Business As Usual) | 0.045 | 0.30 | 0.020 | 0.018 |
| **S1** (Moderate Regulation) | 0.025 | 0.60 | 0.010 | 0.015 |
| **S2** (Green Industrialization) | 0.015 | 0.95 | 0.000 | 0.012 |

Illustrative study-area mean HHI from the synthetic demo panel:

| | 2024 | 2040 BAU | 2040 S1 | 2040 S2 |
|--|-----:|---------:|--------:|--------:|
| Mean HHI | ~54 (At-Risk) | ~58 (At-Risk) | ~33 (Moderate) | ~21 (Acceptable) |

---

## 5. Outputs

```
outputs/results/
├── ahp_weights.json            AHP weights + λ_max, CI, CR
├── hhi_historical.csv          ward × year HHI 2000–2024 + sub-indices + zone
├── hhi_forecast_BAU.csv        ward × year HHI 2025–2040
├── hhi_forecast_S1.csv
└── hhi_forecast_S2.csv

outputs/figures/
├── hhi_timeseries.png          study-area HHI, history + scenarios
├── hhi_zone_distribution.png   vulnerability-zone composition over time
├── hhi_subindex_contrib.png    weighted C1–C5 contributions
└── hhi_ward_ranking.png        ward HHI ranking

outputs/maps/
├── hhi_historical_2024.png
└── hhi_{BAU,S1,S2}_2040.png    ward-centroid HHI maps
```

---

## 6. Swapping in real data

1. **Real environmental panel** — drop a CSV at
   `data/processed/features/hhi_panel_raw.csv` with the columns in §2; Stage 2
   loads it automatically instead of synthesizing.
2. **Real LSTM forecasts** — replace Stage 3's `build_all_scenarios` with
   `src/models/forecaster.py` output (same raw columns). Everything downstream
   (Stages 4–8) is unchanged.
3. **True choropleth** — provide a ward-boundary shapefile and call
   `hhi_maps.try_choropleth(df, shapefile)`; it uses geopandas if available and
   falls back to the centroid map otherwise.

---

## 7. Reproducibility & tests

* Deterministic given `cfg.seed` (config single source of truth).
* `tests/test_ahp.py` — config loads, weights sum to 1, CR < 0.10.
* `tests/test_hhi.py` — sub-index bounds, monotonicity (worse env → higher HHI),
  panel schema, scenario ordering (BAU > S1 > S2), shared-scale comparability.

```bash
python -m pytest tests/test_ahp.py tests/test_hhi.py -v
```
