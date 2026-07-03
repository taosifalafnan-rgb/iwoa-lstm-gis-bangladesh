"""
trainer.py — LSTM training loop.

Features:
  - Huber loss
  - AdamW optimizer + cosine annealing LR schedule
  - Early stopping
  - W&B logging (optional, cfg.wandb.enabled)
  - Checkpoint saving (best val loss)
  - Full reproducibility via set_seed
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device

log = get_logger(__name__)


class EarlyStopping:
    """
    Monitor validation loss and stop training when improvement stalls.

    Args:
        patience:  Epochs to wait after last improvement. Defaults to cfg value.
        min_delta: Minimum improvement to count as improvement.
    """

    def __init__(self, patience: int = None, min_delta: float = 1e-6):
        self.patience    = patience or cfg.training.patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.best_epoch  = 0
        self.should_stop = False

    def step(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss.
            epoch:    Current epoch number.

        Returns:
            improved: True if this epoch was the best so far.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            return True  # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                log.info(f"Early stopping triggered at epoch {epoch}. "
                         f"Best epoch was {self.best_epoch} (val_loss={self.best_loss:.6f})")
            return False


class Trainer:
    """
    Full training manager for EnvironmentalLSTM.

    Usage:
        trainer = Trainer(model, train_loader, val_loader)
        history = trainer.train()
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        max_epochs:   Optional[int]   = None,
        lr:           Optional[float] = None,
        weight_decay: Optional[float] = None,
    ):
        """
        Args:
            model:        EnvironmentalLSTM instance.
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            max_epochs:   Max training epochs. Defaults to cfg value.
            lr:           Learning rate. Defaults to cfg value.
            weight_decay: AdamW weight decay. Defaults to cfg value.
        """
        set_seed(cfg.seed)

        self.device      = get_device()
        self.model       = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.max_epochs  = max_epochs  or cfg.training.max_epochs
        # Learning rate priority: explicit arg → config lstm.learning_rate (if
        # present) → 1e-3 default. The IWOA-tuned LR is normally passed in here
        # by the training stage (build path in run_pipeline).
        self.lr          = lr if lr is not None else getattr(
            cfg.lstm, "learning_rate", 1e-3)
        self.weight_decay = weight_decay or cfg.training.weight_decay

        # Loss
        if cfg.training.loss == "huber":
            self.criterion = nn.HuberLoss(delta=cfg.training.huber_delta)
        elif cfg.training.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()
        log.info(f"Loss function: {cfg.training.loss}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6
        )

        self.early_stopping = EarlyStopping()
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        # W&B init
        self.wandb_run = None
        if cfg.wandb.enabled:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases run."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project = cfg.wandb.project,
                entity  = cfg.wandb.entity,
                name    = cfg.wandb.run_name,
                tags    = cfg.wandb.tags,
                config  = {
                    "hidden_1":    self.model.hidden_1,
                    "hidden_2":    self.model.hidden_2,
                    "dropout":     self.model.dropout_rate,
                    "lr":          self.lr,
                    "max_epochs":  self.max_epochs,
                    "loss":        cfg.training.loss,
                }
            )
            log.info(f"W&B initialized: project={cfg.wandb.project}")
        except ImportError:
            log.warning("wandb not installed. Run: pip install wandb")
            self.wandb_run = None
        except Exception as e:
            log.warning(f"W&B init failed: {e}. Continuing without W&B.")
            self.wandb_run = None

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns mean train loss."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss  = self.criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self) -> float:
        """Run one validation epoch. Returns mean val loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds   = self.model(X_batch)
                loss    = self.criterion(preds, y_batch)
                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> str:
        """Save model checkpoint. Returns checkpoint path."""
        Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        path = cfg.training.best_model_file

        torch.save({
            "epoch":             epoch,
            "model_state_dict":  self.model.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "val_loss":          val_loss,
            "model_args": {
                "input_size":  self.model.input_size,
                "hidden_1":    self.model.hidden_1,
                "hidden_2":    self.model.hidden_2,
                "dropout":     self.model.dropout_rate,
                "output_size": self.model.output_size,
            }
        }, path)
        return path

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            history: Dict with train_loss, val_loss, lr per epoch.
        """
        log.info("=" * 60)
        log.info("LSTM TRAINING START")
        log.info(f"  Epochs: {self.max_epochs} | LR: {self.lr} | Device: {self.device}")
        log.info("=" * 60)

        best_val_loss   = float("inf")
        best_checkpoint = None

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            self.scheduler.step()

            improved = self.early_stopping.step(val_loss, epoch)

            # Log every N epochs
            if epoch % cfg.wandb.log_interval == 0 or epoch == 1:
                log.info(f"Epoch {epoch:4d}/{self.max_epochs} | "
                         f"Train: {train_loss:.6f} | "
                         f"Val: {val_loss:.6f} | "
                         f"LR: {current_lr:.6f}"
                         + (" ← best" if improved else ""))

            # W&B logging
            if self.wandb_run:
                import wandb
                wandb.log({
                    "epoch":      epoch,
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "lr":         current_lr,
                })

            # Save best checkpoint
            if improved:
                best_val_loss   = val_loss
                best_checkpoint = self._save_checkpoint(epoch, val_loss)

            if self.early_stopping.should_stop:
                break

        log.info("=" * 60)
        log.info("TRAINING COMPLETE")
        log.info(f"Best val loss: {best_val_loss:.6f} "
                 f"(epoch {self.early_stopping.best_epoch})")
        log.info(f"Checkpoint saved: {best_checkpoint}")
        log.info("=" * 60)

        # Save history CSV
        history_path = "outputs/results/training_history.csv"
        pd.DataFrame(self.history).to_csv(history_path, index=False)
        log.info(f"Training history saved: {history_path}")

        if self.wandb_run:
            import wandb
            wandb.finish()

        return self.history


if __name__ == "__main__":
    from src.models.lstm import EnvironmentalLSTM
    from src.data.sequencer import Sequencer
    import pandas as pd
    import numpy as np

    log.info("Trainer smoke test with synthetic data...")
    set_seed(42)

    # Synthetic data
    n_months = 288
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    feature_cols = cfg.features.pool[:10]
    target_cols  = cfg.features.targets

    synthetic = pd.DataFrame(
        np.random.randn(n_months, len(feature_cols)),
        index=dates,
        columns=feature_cols
    )

    seq = Sequencer(synthetic, feature_cols, target_cols, lookback=6, batch_size=8)
    train_loader, val_loader, _ = seq.get_dataloaders()

    model = EnvironmentalLSTM(
        input_size=len(feature_cols), hidden_1=32, hidden_2=16, output_size=4
    )

    trainer = Trainer(model, train_loader, val_loader, max_epochs=3)
    history = trainer.train()
    log.info(f"Smoke test passed. Final val loss: {history['val_loss'][-1]:.6f}")
