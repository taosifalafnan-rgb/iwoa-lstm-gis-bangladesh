# IWOA-LSTM Pipeline — Technical Flow

End-to-end machine-learning pipeline: from the normalized feature matrix to
2025–2040 scenario forecasts. This is the prediction engine of the study; the
HHI assessment (see `docs/hhi_flow.md`) consumes its outputs.

> **Status:** every stage is implemented and was verified end-to-end on
> synthetic data (IWOA → train → evaluate → forecast all run). Metrics are only
> meaningful once real data is dropped in and PyTorch is installed
> (`pip install -r requirements.txt`).

---

## Pipeline order (`run_pipeline.py`)

```
Step 1  loader.py        validate raw data sources
Step 2  preprocessor.py  merge → impute → lag → normalize → feature_matrix (+scaler)
Step 4  iwoa.py          IWOA: feature selection + LSTM hyperparameter tuning
Step 3b sequencer.py     rebuild sequences on the IWOA-selected features + lookback
Step 5  trainer.py       train the final EnvironmentalLSTM
Step 6  evaluator.py     test metrics + ARIMA/SVR/GRU/WOA-LSTM baselines
Step 7  forecaster.py    auto-regressive rollout to 2040 (BAU/S1/S2, MC-dropout)
```

Run it:
```bash
python run_pipeline.py --config configs/config.yaml
python run_pipeline.py --skip-iwoa            # reuse saved IWOA result
python run_pipeline.py --steps 4 5 6          # specific stages
```

---

## The IWOA (`src/optimization/iwoa.py`)

**Joint encoding** — one position vector per whale, dimension = 23 + 6 = 29:

| Segment | Indices | Meaning |
|---------|---------|---------|
| Binary  | `[0:23]` | ≥ 0.5 ⇒ feature selected (capped at `max_features`, floor 3) |
| Continuous | `[23:29]` | LSTM hyperparameters (hidden_1, hidden_2, dropout, LR, batch, lookback) decoded from `HP_BOUNDS` |

**The four improvements over standard WOA** (each a toggle, so the baseline is
the *same* code with them off — see below):

1. **OBL initialization** — evaluate original + opposite population, keep best N.
2. **Nonlinear inertia weight** `ω(t) = ω_max − (ω_max−ω_min)(t/T)²` blended into moves.
3. **Lévy flight** perturbation (Mantegna) applied with 30 % probability to escape local optima.
4. **Nonlinear convergence factor** `a(t) = 2 − 2(t/T)²` (vs linear `2 − 2t/T`).

**Fitness** (`src/optimization/fitness.py`) — each candidate is scored by the
**validation RMSE of a fast proxy LSTM** (1 layer, ~32 units, ~30 epochs) trained
on *only* the selected features windowed at the candidate's lookback, plus a
small parsimony penalty `0.01·(n_selected/23)`. Infeasible candidates (lookback
too long, empty split, NaN loss) return `+inf`.

```
fitness(features, hp) = val_RMSE(proxy_LSTM)  +  0.01 · n_selected/23
```

### IWOA vs WOA baseline (`src/optimization/woa.py`)

`WOA` **is** `IWOA` with all four improvements disabled — guaranteeing any
performance gap is attributable purely to the improvements (a clean ablation):

| Component | IWOA | WOA |
|-----------|:----:|:---:|
| OBL init | ✓ | ✗ |
| Nonlinear a(t) | ✓ | ✗ (linear) |
| Inertia weight | ✓ | ✗ |
| Lévy flight | ✓ | ✗ |

Outputs are saved under distinct names (`iwoa_*` vs `woa_*`) so both survive.

---

## The LSTM (`src/models/lstm.py`)

```
Input [batch, lookback, n_selected]
  → LSTM1 (hidden_1) → Dropout
  → LSTM2 (hidden_2) → Dropout
  → BatchNorm → Dense(hidden_2//2) + ReLU → Dense(4)
Output [batch, 4] = (pm25, bod, lst, ndvi)
```

- All hyperparameters come from the IWOA result (`build_lstm_from_iwoa`).
- **MC-Dropout** (`predict_with_uncertainty`, T=50): dropout stays active at
  inference to produce mean + confidence bands.

**Training** (`src/models/trainer.py`): Huber loss, AdamW (weight decay),
cosine-annealing LR, gradient clipping, early stopping, best-checkpoint saving,
optional W&B logging (guarded by `cfg.wandb.enabled`).

---

## Evaluation & baselines (`src/models/evaluator.py`)

Per-target (pm25/bod/lst/ndvi) **RMSE, MAE, MAPE, R²** on the temporal test
split → `lstm_test_metrics.csv`.

Required baselines, each wrapped so a missing optional dependency degrades to a
logged NaN row instead of crashing → `baseline_comparison.csv`:

| Baseline | How | Dependency |
|----------|-----|------------|
| **ARIMA** | ARIMA(1,1,1) per target, naive fallback | statsmodels |
| **SVR** | RBF SVR on flattened windows, one per target | scikit-learn |
| **GRU** | 2-layer GRU, same I/O shape as the LSTM | torch |
| **WOA-LSTM** | LSTM tuned by the baseline WOA (supplied by caller) | torch |

---

## Forecasting to 2040 (`src/models/forecaster.py`)

Auto-regressive monthly rollout, 2025-01 → 2040-12, per scenario. Each future
month's input vector is assembled from three column types:

1. **Target-derived features** → the model's own prediction fed back in.
2. **Target lag features** (`pm25_lag3`, `ndvi_lag12`, …) → resolved from a
   running history buffer at the correct offset.
3. **Exogenous drivers** → projected with transparent, scenario-dependent rules
   (`_project_exogenous`): industrial/air terms grow with the scenario growth
   rate and are abated by ETP compliance; water/green terms respond to
   compliance and green-loss; climate variables repeat monthly climatology;
   anything unmapped persists.

Annual scenario rates (`cfg.scenarios`) are converted to per-month increments
`(1+annual)^(1/12) − 1`. Targets are inverse-transformed to physical units via
the saved MinMax scaler, and MC-Dropout adds `*_lower` / `*_upper` bands.
Outputs → `data/processed/scenarios/forecast_{BAU,S1,S2}.csv`.

---

## Bugs fixed while wiring this up

- `iwoa.py`: `np.math.gamma` (removed in NumPy 2.0) → `math.gamma`;
  `load_iwoa_result` raised on duplicate keys → merged/filtered to dataclass fields.
- `trainer.py`: read a non-existent `cfg.lstm.learning_rate` → falls back to the
  IWOA-tuned LR passed in, else 1e-3.
- `run_pipeline.py`: IWOA now runs on the feature matrix (so `lookback` is
  actually searched), sequences are rebuilt on the *selected* features before
  training, the tuned LR flows into the trainer, and the scaler is loaded for
  physical-unit forecasts.

---

## Reproducibility & tests

`tests/test_iwoa.py` (torch-free): Lévy shape, OBL doubling, decode constraints
(`max_features`, floor 3, HP bounds), IWOA-vs-WOA toggles, and a full `optimize()`
run whose convergence curve is monotone non-increasing.

```bash
python -m pytest tests/ -v      # 22 tests (HHI + AHP + IWOA)
```
