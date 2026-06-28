"""
run_pipeline.py — Master pipeline runner.

Executes all steps in order:
  1. Data validation
  2. Preprocessing
  3. Sequence building
  4. IWOA optimization
  5. LSTM training
  6. Evaluation
  7. Scenario forecasting
  8. AHP + HHI computation
  9. Visualization

Usage:
    python run_pipeline.py --config configs/config.yaml
    python run_pipeline.py --skip-iwoa   (if IWOA already run)
    python run_pipeline.py --steps 1 2 3 (run specific steps only)
"""

import argparse
import sys
import time
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.loader import run_data_check
from src.data.preprocessor import Preprocessor
from src.data.sequencer import Sequencer
from src.analysis.ahp import AHP
from src.analysis.hhi import HHIComputer

log = get_logger(__name__)


def step_1_data_check(cfg):
    log.info("STEP 1: Data Availability Check")
    report = run_data_check()
    available = sum(1 for v in report.values() if v.get("available", False))
    log.info(f"  {available}/{len(report)} data sources available")
    return report


def step_2_preprocessing(cfg):
    log.info("STEP 2: Preprocessing")
    prep = Preprocessor()
    feature_matrix = prep.run()
    log.info(f"  Feature matrix: {feature_matrix.shape}")
    return feature_matrix


def step_3_sequencing(cfg, feature_matrix):
    log.info("STEP 3: Sequence Building")
    feature_cols = cfg.features.pool
    target_cols  = cfg.features.targets

    seq = Sequencer(feature_matrix, feature_cols, target_cols)
    seq.save_sequences()
    train_loader, val_loader, test_loader = seq.get_dataloaders()
    log.info(f"  DataLoaders ready")
    return train_loader, val_loader, test_loader, seq


def step_4_iwoa(cfg, train_loader, val_loader, skip=False):
    log.info("STEP 4: IWOA Feature Selection + Hyperparameter Tuning")

    if skip:
        log.info("  Skipping IWOA — loading saved result...")
        from src.optimization.iwoa import load_iwoa_result
        return load_iwoa_result()

    from src.optimization.fitness import FitnessEvaluator
    from src.optimization.iwoa import IWOA

    evaluator = FitnessEvaluator(train_loader, val_loader)
    iwoa = IWOA(fitness_fn=evaluator.evaluate)
    result = iwoa.optimize()
    return result


def step_5_train(cfg, iwoa_result, train_loader, val_loader):
    log.info("STEP 5: LSTM Training")
    from src.models.lstm import build_lstm_from_iwoa
    from src.models.trainer import Trainer

    model = build_lstm_from_iwoa(iwoa_result)
    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train()
    log.info(f"  Training complete. Best val loss: {min(history['val_loss']):.6f}")
    return model, history


def step_6_evaluate(cfg, model, test_loader):
    log.info("STEP 6: Evaluation")
    from src.models.evaluator import Evaluator

    evaluator = Evaluator(model, test_loader)
    metrics = evaluator.evaluate()
    return metrics


def step_7_forecast(cfg, model, feature_matrix, iwoa_result):
    log.info("STEP 7: Scenario Forecasting (2025-2040)")
    from src.models.forecaster import Forecaster

    forecaster = Forecaster(model, feature_matrix, iwoa_result)
    forecasts = {}
    for scenario in ["BAU", "S1", "S2"]:
        forecast_df = forecaster.forecast(scenario)
        forecasts[scenario] = forecast_df
        log.info(f"  {scenario}: {len(forecast_df)} monthly predictions")
    return forecasts


def step_8_hhi(cfg, feature_matrix, forecasts):
    log.info("STEP 8: AHP + HHI Computation")

    ahp = AHP()
    weights, cr = ahp.compute()

    hhi_computer = HHIComputer(weights=weights)

    # Historical HHI
    hist_hhi = hhi_computer.compute(feature_matrix)
    hhi_computer.save(hist_hhi, "historical")

    # Scenario HHI
    for scenario, forecast_df in forecasts.items():
        scenario_hhi = hhi_computer.compute(forecast_df)
        hhi_computer.save(scenario_hhi, scenario)

    log.info("  HHI computation complete for all scenarios")
    return hist_hhi, {k: hhi_computer.compute(v) for k, v in forecasts.items()}


def step_9_visualize(cfg):
    log.info("STEP 9: Visualization")
    try:
        from src.visualization.plots import plot_all
        from src.visualization.maps import map_all
        plot_all()
        map_all()
        log.info("  All figures and maps generated")
    except Exception as e:
        log.warning(f"  Visualization step failed: {e}")
        log.warning("  Run notebooks/10_results_visualization.ipynb manually")


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IWOA-LSTM GIS Bangladesh Pipeline"
    )
    parser.add_argument("--config",    default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--skip-iwoa", action="store_true",
                        help="Load saved IWOA result instead of re-running")
    parser.add_argument("--steps",    nargs="+", type=int, default=None,
                        help="Run specific steps only (1-9)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()

    run_steps = set(args.steps) if args.steps else set(range(1, 10))

    log.info("=" * 70)
    log.info("IWOA-LSTM GIS BANGLADESH — FULL PIPELINE")
    log.info(f"Config: {args.config} | Device: {device} | Seed: {cfg.seed}")
    log.info("=" * 70)

    start_time = time.time()

    # Step 1
    if 1 in run_steps:
        step_1_data_check(cfg)

    # Step 2
    feature_matrix = None
    if 2 in run_steps:
        feature_matrix = step_2_preprocessing(cfg)

    # Step 3
    train_loader = val_loader = test_loader = seq = None
    if 3 in run_steps and feature_matrix is not None:
        train_loader, val_loader, test_loader, seq = \
            step_3_sequencing(cfg, feature_matrix)

    # Step 4
    iwoa_result = None
    if 4 in run_steps and train_loader is not None:
        iwoa_result = step_4_iwoa(cfg, train_loader, val_loader,
                                  skip=args.skip_iwoa)

    # Step 5
    model = None
    if 5 in run_steps and iwoa_result is not None:
        model, history = step_5_train(cfg, iwoa_result, train_loader, val_loader)

    # Step 6
    if 6 in run_steps and model is not None:
        metrics = step_6_evaluate(cfg, model, test_loader)

    # Step 7
    forecasts = {}
    if 7 in run_steps and model is not None and feature_matrix is not None:
        forecasts = step_7_forecast(cfg, model, feature_matrix, iwoa_result)

    # Step 8
    if 8 in run_steps and feature_matrix is not None:
        step_8_hhi(cfg, feature_matrix, forecasts)

    # Step 9
    if 9 in run_steps:
        step_9_visualize(cfg)

    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info(f"PIPELINE COMPLETE — Total time: {elapsed/60:.1f} minutes")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
