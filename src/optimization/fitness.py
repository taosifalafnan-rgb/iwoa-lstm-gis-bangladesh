"""
fitness.py — Fitness function for IWOA (and baseline WOA).

The IWOA searches jointly over (a) which subset of the 23 candidate features to
keep and (b) the LSTM hyperparameters. Each candidate must be scored by a number
the optimizer minimizes. We use the **validation RMSE of a fast proxy LSTM**:

    fitness = validation_RMSE  +  parsimony_penalty · (n_selected / n_total)

Why a *proxy* LSTM (small, few epochs) rather than the full model?
  - IWOA evaluates the fitness function thousands of times (n_whales × max_iter).
  - Training the full 128/64-unit model each time would take days.
  - A small 1-layer LSTM trained for ~30 epochs is a cheap, well-correlated
    surrogate for the ranking of candidates — exactly what the search needs.

The evaluator is built around the **feature matrix** (not pre-built loaders) so
it can honour BOTH search dimensions correctly:
  - feature subset → only the selected columns are windowed, and
  - the candidate `lookback` → sequences are rebuilt at that window length.

Design notes / edge cases handled:
  - Infeasible candidates (lookback too long for the data, empty val split,
    NaN loss) return +inf so IWOA discards them.
  - A tiny parsimony penalty nudges the search toward fewer features (better
    generalization on this small 200–288 row dataset), without dominating RMSE.
  - Fully deterministic given cfg.seed (each candidate reseeds its proxy init).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Optional, Dict

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.sequencer import Sequencer

log = get_logger(__name__)


# ── PROXY LSTM ──────────────────────────────────────────────────────────────

class ProxyLSTM(nn.Module):
    """
    Lightweight single-layer LSTM surrogate used only for fitness scoring.

    Deliberately simpler than `EnvironmentalLSTM` (no BatchNorm, no MC dropout,
    one recurrent layer) so thousands of candidate evaluations stay fast and
    numerically robust on tiny batches.

    Args:
        input_size:  Number of selected features.
        hidden:      Hidden units (small, e.g. 32).
        output_size: Number of target variables.
        dropout:     Dropout rate applied to the last hidden state.
    """

    def __init__(self, input_size: int, hidden: int, output_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch, lookback, input_size].

        Returns:
            Predictions [batch, output_size].
        """
        out, _ = self.lstm(x)          # [batch, seq, hidden]
        last = self.drop(out[:, -1, :])  # last time step
        return self.fc(last)


# ── FITNESS EVALUATOR ───────────────────────────────────────────────────────

class FitnessEvaluator:
    """
    Scores IWOA candidates by training a proxy LSTM and returning val RMSE.

    Usage:
        evaluator = FitnessEvaluator(feature_matrix)          # date-indexed df
        rmse = evaluator.evaluate(selected_features, hp_dict) # called by IWOA

    Then:
        iwoa = IWOA(fitness_fn=evaluator.evaluate)
    """

    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        feature_cols:   Optional[List[str]] = None,
        target_cols:    Optional[List[str]] = None,
        proxy_hidden:   Optional[int]   = None,
        proxy_epochs:   Optional[int]   = None,
        parsimony:      float = 0.01,
        device:         Optional[torch.device] = None,
    ):
        """
        Args:
            feature_matrix: Normalized, date-indexed DataFrame (preprocessor out).
            feature_cols:   Full candidate feature pool. Defaults to cfg.features.pool.
            target_cols:    LSTM targets. Defaults to cfg.features.targets.
            proxy_hidden:   Proxy LSTM hidden units. Defaults to cfg.iwoa.proxy_lstm.hidden.
            proxy_epochs:   Proxy training epochs. Defaults to cfg.iwoa.proxy_lstm.epochs.
            parsimony:      Weight of the feature-count penalty added to RMSE.
            device:         Torch device. Auto-detected if None.
        """
        self.df           = feature_matrix
        self.feature_cols = feature_cols or cfg.features.pool
        self.target_cols  = target_cols  or cfg.features.targets
        self.n_total      = len(self.feature_cols)
        self.parsimony    = parsimony

        # Proxy config (fast surrogate).
        self.proxy_hidden = proxy_hidden or getattr(cfg.iwoa.proxy_lstm, "hidden", 32)
        self.proxy_epochs = proxy_epochs or getattr(cfg.iwoa.proxy_lstm, "epochs", 30)
        self.device       = device or get_device()

        self._eval_count = 0
        log.info(f"FitnessEvaluator ready: {self.n_total} candidate features, "
                 f"targets={self.target_cols}, proxy hidden={self.proxy_hidden}, "
                 f"epochs={self.proxy_epochs}, device={self.device}")

    # ── PUBLIC API (the function IWOA minimizes) ────────────────────────────

    def evaluate(self, features: List[str], hp: Dict) -> float:
        """
        Score one IWOA candidate.

        Args:
            features: Selected feature names (subset of the pool).
            hp:       Hyperparameter dict with keys hidden_1, hidden_2, dropout,
                      learning_rate, batch_size, lookback.

        Returns:
            fitness: val RMSE + parsimony penalty, or +inf if infeasible.
        """
        self._eval_count += 1

        # Guard: need at least one valid feature.
        features = [f for f in features if f in self.df.columns]
        if len(features) == 0:
            return float("inf")

        lookback   = int(hp.get("lookback", cfg.lstm.lookback))
        batch_size = int(hp.get("batch_size", cfg.lstm.batch_size))
        lr         = float(hp.get("learning_rate", 1e-3))
        dropout    = float(hp.get("dropout", 0.1))

        try:
            train_loader, val_loader = self._build_loaders(features, lookback, batch_size)
        except Exception as e:
            log.debug(f"[fitness #{self._eval_count}] sequence build failed: {e}")
            return float("inf")

        # Infeasible if either split ended up empty (e.g. lookback too long).
        if train_loader is None or val_loader is None:
            return float("inf")

        rmse = self._train_and_score(train_loader, val_loader,
                                     input_size=len(features), dropout=dropout, lr=lr)
        if not np.isfinite(rmse):
            return float("inf")

        # Parsimony: prefer fewer features (helps generalization on small data).
        penalty = self.parsimony * (len(features) / self.n_total)
        return float(rmse + penalty)

    # ── INTERNALS ───────────────────────────────────────────────────────────

    def _build_loaders(self, features: List[str], lookback: int, batch_size: int):
        """
        Build train/val DataLoaders for the selected features & lookback.

        Only the selected feature columns (plus the targets, which the Sequencer
        reads for y) are windowed. The test split is discarded here — IWOA must
        never see the test set.

        Args:
            features:   Selected feature names.
            lookback:   Candidate window length.
            batch_size: Candidate batch size.

        Returns:
            (train_loader, val_loader) or (None, None) if a split is empty.
        """
        # Columns the sequencer needs: selected features ∪ targets.
        needed = list(dict.fromkeys(features + self.target_cols))
        sub_df = self.df[[c for c in needed if c in self.df.columns]].copy()

        seq = Sequencer(sub_df, features, self.target_cols,
                        lookback=lookback, batch_size=max(2, batch_size))
        train_loader, val_loader, _test = seq.get_dataloaders(shuffle_train=False)

        # Reject degenerate splits (Sequencer pads empty splits with 1 dummy row).
        if len(train_loader.dataset) <= 1 or len(val_loader.dataset) <= 1:
            return None, None
        return train_loader, val_loader

    def _train_and_score(self, train_loader, val_loader, input_size: int,
                         dropout: float, lr: float) -> float:
        """
        Train the proxy LSTM briefly and return validation RMSE.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            input_size:   Number of selected features.
            dropout:      Proxy dropout rate.
            lr:           Learning rate for the proxy Adam optimizer.

        Returns:
            Validation RMSE (float), or +inf on numerical failure.
        """
        # Reseed so proxy weight init is identical for identical candidates
        # → deterministic, comparable fitness values across the search.
        set_seed(cfg.seed)

        model = ProxyLSTM(input_size, self.proxy_hidden,
                          len(self.target_cols), dropout).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # ── Train ──
        model.train()
        for _epoch in range(self.proxy_epochs):
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optim.zero_grad()
                loss = criterion(model(X), y)
                if not torch.isfinite(loss):
                    return float("inf")
                loss.backward()
                optim.step()

        # ── Validate (RMSE over all target variables) ──
        model.eval()
        sq_err, n = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                sq_err += torch.sum((pred - y) ** 2).item()
                n += y.numel()

        if n == 0:
            return float("inf")
        return float(np.sqrt(sq_err / n))


if __name__ == "__main__":
    # Standalone smoke test with a synthetic normalized feature matrix.
    log.info("FitnessEvaluator smoke test...")
    set_seed(cfg.seed)

    n_months = 288
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    cols = cfg.features.pool
    fm = pd.DataFrame(np.random.rand(n_months, len(cols)), index=dates, columns=cols)

    ev = FitnessEvaluator(fm, proxy_epochs=3)   # few epochs for a fast test
    hp = {"hidden_1": 64, "hidden_2": 32, "dropout": 0.2,
          "learning_rate": 1e-3, "batch_size": 16, "lookback": 12}
    score = ev.evaluate(["pm25", "ndvi", "lst", "no2", "bod"], hp)
    log.info(f"Sample candidate fitness (val RMSE + penalty): {score:.6f}")
    log.info("FitnessEvaluator smoke test passed.")
