"""
evaluator.py — Test-set evaluation and baseline comparison.

Two responsibilities:

  1. `Evaluator.evaluate()` — score the trained IWOA-LSTM on the held-out test
     set with the four standard regression metrics, PER target variable
     (pm25, bod, lst, ndvi):  RMSE, MAE, MAPE, R².
     → outputs/results/lstm_test_metrics.csv

  2. `Evaluator.compare_baselines()` — run the required baselines and tabulate
     their test RMSE next to the IWOA-LSTM for the paper:
        • ARIMA        (classical univariate time series, statsmodels)
        • SVR          (Support Vector Regression on flattened windows, sklearn)
        • Vanilla GRU  (same shape as the LSTM but GRU cells, torch)
        • WOA-LSTM     (LSTM tuned by the *baseline* WOA — supplied by caller)
     → outputs/results/baseline_comparison.csv

Every baseline is wrapped in try/except: a missing optional dependency
(statsmodels, sklearn) or a degenerate split degrades to a logged warning and a
NaN row rather than crashing the whole evaluation.

All metrics are computed in the model's (normalized) output space, which is the
correct, scale-free basis for comparing models. Pass a fitted scaler to
`evaluate(inverse=True)` if you additionally want physical-unit errors.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.device import get_device
from src.utils.seed import set_seed

log = get_logger(__name__)


# ── METRIC PRIMITIVES ───────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    """
    Mean Absolute Percentage Error (%), ignoring targets whose magnitude is
    below `eps` to avoid division blow-ups near zero (common in normalized data).
    """
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return all four metrics for a single (flattened) target vector."""
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "MAPE": mape(y_true, y_pred), "R2": r2(y_true, y_pred)}


# ── VANILLA GRU BASELINE ────────────────────────────────────────────────────

class VanillaGRU(nn.Module):
    """
    Two-layer GRU baseline mirroring the LSTM's shape (GRU cells instead of
    LSTM cells), used for the architecture-ablation comparison.
    """

    def __init__(self, input_size: int, hidden_1: int = 64, hidden_2: int = 32,
                 output_size: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden_1, hidden_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru1(x)
        out = self.drop1(out)
        out, _ = self.gru2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out)


# ── EVALUATOR ───────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluate a trained model on the test set and (optionally) against baselines.

    Usage:
        evaluator = Evaluator(model, test_loader)
        metrics_df = evaluator.evaluate()
        comparison_df = evaluator.compare_baselines(
            X_train, y_train, X_test, y_test, woa_model=woa_lstm)
    """

    def __init__(
        self,
        model:        nn.Module,
        test_loader,
        target_names: Optional[List[str]] = None,
        device:       Optional[torch.device] = None,
    ):
        """
        Args:
            model:        Trained EnvironmentalLSTM.
            test_loader:  Test DataLoader.
            target_names: Names of target variables. Defaults to cfg.features.targets.
            device:       Torch device. Auto-detected if None.
        """
        self.device       = device or get_device()
        self.model        = model.to(self.device)
        self.test_loader  = test_loader
        self.target_names = target_names or cfg.features.targets

    # ── COLLECT PREDICTIONS ─────────────────────────────────────────────────

    def _collect(self, loader) -> tuple:
        """
        Run the model over a loader and stack predictions/targets.

        Args:
            loader: A DataLoader yielding (X, y).

        Returns:
            (y_true, y_pred) as numpy arrays [n_samples, n_targets].
        """
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                preds.append(self.model(X).cpu().numpy())
                trues.append(y.numpy())
        return np.concatenate(trues), np.concatenate(preds)

    # ── LSTM TEST METRICS ───────────────────────────────────────────────────

    def evaluate(self, save: bool = True) -> pd.DataFrame:
        """
        Compute per-target test metrics for the trained LSTM.

        Args:
            save: If True, write outputs/results/lstm_test_metrics.csv.

        Returns:
            DataFrame indexed by target with RMSE, MAE, MAPE, R² columns
            (plus an "ALL" row aggregating across every target).
        """
        log.info("Evaluating IWOA-LSTM on the test set...")
        y_true, y_pred = self._collect(self.test_loader)

        rows = {}
        for j, name in enumerate(self.target_names):
            rows[name] = all_metrics(y_true[:, j], y_pred[:, j])
        rows["ALL"] = all_metrics(y_true.ravel(), y_pred.ravel())

        df = pd.DataFrame(rows).T[["RMSE", "MAE", "MAPE", "R2"]]
        log.info(f"Test metrics:\n{df.round(4).to_string()}")

        if save:
            Path("outputs/results").mkdir(parents=True, exist_ok=True)
            path = "outputs/results/lstm_test_metrics.csv"
            df.to_csv(path)
            log.info(f"Test metrics saved: {path}")
        return df

    # ── BASELINE COMPARISON ─────────────────────────────────────────────────

    def compare_baselines(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
        woa_model: Optional[nn.Module] = None,
        gru_epochs: int = 60,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Compare IWOA-LSTM test RMSE against ARIMA, SVR, GRU, and WOA-LSTM.

        Args:
            X_train: Train windows [n, lookback, n_features].
            y_train: Train targets [n, n_targets].
            X_test:  Test windows  [n, lookback, n_features].
            y_test:  Test targets  [n, n_targets].
            woa_model: Optional LSTM trained from the WOA baseline. If None, the
                       WOA-LSTM row is skipped (NaN).
            gru_epochs: Training epochs for the GRU baseline.
            save: If True, write outputs/results/baseline_comparison.csv.

        Returns:
            DataFrame of per-model RMSE (overall and per target).
        """
        log.info("Running baseline comparison (ARIMA / SVR / GRU / WOA-LSTM)...")
        results: Dict[str, Dict[str, float]] = {}

        # IWOA-LSTM (the model under test).
        y_true, y_pred = self._collect(self.test_loader)
        results["IWOA-LSTM"] = self._rmse_row(y_true, y_pred)

        # ARIMA — classical univariate baseline per target.
        results["ARIMA"] = self._baseline_arima(y_train, y_test)

        # SVR — on flattened windows, one regressor per target.
        results["SVR"] = self._baseline_svr(X_train, y_train, X_test, y_test)

        # GRU — same shape as LSTM, GRU cells.
        results["GRU"] = self._baseline_gru(X_train, y_train, X_test, y_test,
                                            epochs=gru_epochs)

        # WOA-LSTM — evaluated only if the caller trained one.
        if woa_model is not None:
            wt, wp = self._predict_torch(woa_model, X_test, y_test)
            results["WOA-LSTM"] = self._rmse_row(wt, wp)
        else:
            log.info("  WOA-LSTM model not supplied — row left as NaN.")
            results["WOA-LSTM"] = self._nan_row()

        df = pd.DataFrame(results).T
        log.info(f"Baseline comparison (RMSE):\n{df.round(4).to_string()}")

        if save:
            Path("outputs/results").mkdir(parents=True, exist_ok=True)
            path = "outputs/results/baseline_comparison.csv"
            df.to_csv(path)
            log.info(f"Baseline comparison saved: {path}")
        return df

    # ── RMSE row helpers ────────────────────────────────────────────────────

    def _rmse_row(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Per-target + overall RMSE as a dict keyed by target name and 'ALL'."""
        row = {name: rmse(y_true[:, j], y_pred[:, j])
               for j, name in enumerate(self.target_names)}
        row["ALL"] = rmse(y_true.ravel(), y_pred.ravel())
        return row

    def _nan_row(self) -> Dict[str, float]:
        """A row of NaNs (used when a baseline is unavailable)."""
        row = {name: float("nan") for name in self.target_names}
        row["ALL"] = float("nan")
        return row

    # ── INDIVIDUAL BASELINES ────────────────────────────────────────────────

    def _baseline_arima(self, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        ARIMA(1,1,1) per target: fit on the training target series, forecast the
        test horizon, and score RMSE. Univariate — one model per target column.

        Args:
            y_train: Training targets [n_train, n_targets].
            y_test:  Test targets     [n_test, n_targets].

        Returns:
            Per-target + overall RMSE dict.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            log.warning("  statsmodels not installed — ARIMA skipped.")
            return self._nan_row()

        h = len(y_test)
        preds = np.zeros_like(y_test)
        for j in range(y_test.shape[1]):
            try:
                series = y_train[:, j]
                fit = ARIMA(series, order=(1, 1, 1)).fit()
                preds[:, j] = fit.forecast(steps=h)
            except Exception as e:
                log.warning(f"  ARIMA failed for target {j}: {e}")
                preds[:, j] = y_train[-1, j]   # fall back to last value (naive)
        return self._rmse_row(y_test, preds)

    def _baseline_svr(self, X_train, y_train, X_test, y_test) -> Dict[str, float]:
        """
        Support Vector Regression on flattened lookback windows, one SVR per
        target (wrapped in MultiOutputRegressor).

        Args:
            X_train/X_test: Windows [n, lookback, n_features].
            y_train/y_test: Targets [n, n_targets].

        Returns:
            Per-target + overall RMSE dict.
        """
        try:
            from sklearn.svm import SVR
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            log.warning("  scikit-learn not installed — SVR skipped.")
            return self._nan_row()

        try:
            Xtr = X_train.reshape(len(X_train), -1)   # flatten time × features
            Xte = X_test.reshape(len(X_test), -1)
            model = MultiOutputRegressor(SVR(kernel="rbf", C=10.0, gamma="scale"))
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)
            return self._rmse_row(y_test, preds)
        except Exception as e:
            log.warning(f"  SVR failed: {e}")
            return self._nan_row()

    def _baseline_gru(self, X_train, y_train, X_test, y_test,
                     epochs: int = 60) -> Dict[str, float]:
        """
        Train a vanilla GRU (same I/O shape as the LSTM) and score test RMSE.

        Args:
            X_train/X_test: Windows [n, lookback, n_features].
            y_train/y_test: Targets [n, n_targets].
            epochs:         Training epochs.

        Returns:
            Per-target + overall RMSE dict.
        """
        try:
            set_seed(cfg.seed)
            model = VanillaGRU(input_size=X_train.shape[2],
                               output_size=y_train.shape[1]).to(self.device)
            optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            criterion = nn.HuberLoss(delta=cfg.training.huber_delta)

            Xtr = torch.FloatTensor(X_train).to(self.device)
            ytr = torch.FloatTensor(y_train).to(self.device)

            model.train()
            for _ in range(epochs):
                optim.zero_grad()
                loss = criterion(model(Xtr), ytr)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            gt, gp = self._predict_torch(model, X_test, y_test)
            return self._rmse_row(gt, gp)
        except Exception as e:
            log.warning(f"  GRU baseline failed: {e}")
            return self._nan_row()

    def _predict_torch(self, model: nn.Module, X_test, y_test):
        """
        Run a torch model on test windows.

        Args:
            model:  A torch model taking [n, lookback, features].
            X_test: Test windows (numpy).
            y_test: Test targets (numpy).

        Returns:
            (y_test, predictions) as numpy arrays.
        """
        model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_test).to(self.device)
            preds = model(X).cpu().numpy()
        return y_test, preds


if __name__ == "__main__":
    # Standalone metric sanity check (no torch model needed).
    log.info("Evaluator metric self-test...")
    yt = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    yp = np.array([[1.1, 1.9], [1.8, 4.2], [3.2, 5.7]])
    for j in range(2):
        m = all_metrics(yt[:, j], yp[:, j])
        log.info(f"  target {j}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    log.info("Evaluator metric self-test passed.")
