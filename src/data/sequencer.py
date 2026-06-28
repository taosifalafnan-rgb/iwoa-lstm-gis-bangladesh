"""
sequencer.py — Build sliding window LSTM sequence tensors.

Takes the normalized feature matrix and converts it to:
  - X tensor: [n_samples, lookback, n_features]
  - y tensor: [n_samples, n_targets]

Also handles temporal train/val/test split without data leakage.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


class EnvironmentalSequenceDataset(Dataset):
    """
    PyTorch Dataset for multi-output LSTM sequence prediction.

    Args:
        X:       Feature sequences [n_samples, lookback, n_features]
        y:       Target values     [n_samples, n_targets]
        dates:   Corresponding dates for each sample (for logging)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 dates: Optional[pd.DatetimeIndex] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.dates = dates

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class Sequencer:
    """
    Converts feature matrix to LSTM-ready sequences with temporal split.

    Usage:
        seq = Sequencer(feature_matrix, feature_cols, target_cols, lookback=12)
        train_loader, val_loader, test_loader = seq.get_dataloaders()
    """

    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        feature_cols: list,
        target_cols:  list,
        lookback:     int = None,
        batch_size:   int = None,
    ):
        """
        Args:
            feature_matrix: Normalized DataFrame from preprocessor.
            feature_cols:   List of input feature column names.
            target_cols:    List of target column names (LSTM outputs).
            lookback:       Number of time steps to look back. Defaults to cfg value.
            batch_size:     DataLoader batch size. Defaults to cfg value.
        """
        set_seed(cfg.seed)
        self.df          = feature_matrix
        self.feature_cols = feature_cols
        self.target_cols  = target_cols
        self.lookback     = lookback    or cfg.lstm.lookback
        self.batch_size   = batch_size  or cfg.lstm.batch_size

        # Validate columns exist
        missing_feat = [c for c in feature_cols if c not in feature_matrix.columns]
        missing_tgt  = [c for c in target_cols  if c not in feature_matrix.columns]
        if missing_feat:
            log.warning(f"Feature columns not in matrix (will be NaN): {missing_feat}")
        if missing_tgt:
            log.warning(f"Target columns not in matrix (will be NaN): {missing_tgt}")

    def build_sequences(self) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Build sliding window sequences from the feature matrix.

        For each time step t (starting at t=lookback):
            X[t] = feature_matrix[t-lookback : t]   shape: [lookback, n_features]
            y[t] = feature_matrix[t]                 shape: [n_targets]

        Returns:
            X:     Feature sequences [n_samples, lookback, n_features]
            y:     Target values     [n_samples, n_targets]
            dates: DatetimeIndex of the prediction timestamps
        """
        log.info(f"Building sequences: lookback={self.lookback}, "
                 f"features={len(self.feature_cols)}, targets={len(self.target_cols)}")

        # Handle missing feature columns gracefully
        available_features = [c for c in self.feature_cols if c in self.df.columns]
        available_targets  = [c for c in self.target_cols  if c in self.df.columns]

        # Fill missing with zeros (placeholder until real data arrives)
        X_full = np.zeros((len(self.df), len(self.feature_cols)))
        for i, col in enumerate(self.feature_cols):
            if col in self.df.columns:
                X_full[:, i] = self.df[col].values
            else:
                X_full[:, i] = 0.0

        y_full = np.zeros((len(self.df), len(self.target_cols)))
        for i, col in enumerate(self.target_cols):
            if col in self.df.columns:
                y_full[:, i] = self.df[col].values
            else:
                y_full[:, i] = 0.0

        X_seqs, y_seqs, dates = [], [], []
        for t in range(self.lookback, len(self.df)):
            X_seqs.append(X_full[t - self.lookback : t, :])
            y_seqs.append(y_full[t, :])
            dates.append(self.df.index[t])

        X = np.array(X_seqs)   # [n_samples, lookback, n_features]
        y = np.array(y_seqs)   # [n_samples, n_targets]
        dates = pd.DatetimeIndex(dates)

        log.info(f"Sequences built: X={X.shape}, y={y.shape}")
        return X, y, dates

    def temporal_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> dict:
        """
        Split sequences into train/val/test by date — no shuffling.

        Train: 2000–2018 | Val: 2019–2021 | Test: 2022–2024

        Args:
            X:     Feature sequences
            y:     Target sequences
            dates: DatetimeIndex aligned with sequences

        Returns:
            splits: Dict with 'train', 'val', 'test' keys,
                    each containing (X, y, dates) tuples.
        """
        train_end = pd.to_datetime(cfg.dates.train_end)
        val_end   = pd.to_datetime(cfg.dates.val_end)

        train_mask = dates <= train_end
        val_mask   = (dates > train_end) & (dates <= val_end)
        test_mask  = dates > val_end

        splits = {
            "train": (X[train_mask], y[train_mask], dates[train_mask]),
            "val":   (X[val_mask],   y[val_mask],   dates[val_mask]),
            "test":  (X[test_mask],  y[test_mask],  dates[test_mask]),
        }

        for split, (Xs, ys, ds) in splits.items():
            pct = 100 * len(Xs) / len(X)
            log.info(f"  {split:5s}: {len(Xs):4d} samples ({pct:.1f}%) | "
                     f"{ds[0].date() if len(ds)>0 else 'N/A'} → "
                     f"{ds[-1].date() if len(ds)>0 else 'N/A'}")

        return splits

    def get_dataloaders(
        self, shuffle_train: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build and return train/val/test DataLoaders.

        Args:
            shuffle_train: Whether to shuffle training data.
                           Default False — temporal order preserved.

        Returns:
            train_loader, val_loader, test_loader: PyTorch DataLoaders.
        """
        X, y, dates = self.build_sequences()
        splits = self.temporal_split(X, y, dates)

        def make_loader(split_name: str, shuffle: bool) -> DataLoader:
            Xs, ys, ds = splits[split_name]
            if len(Xs) == 0:
                log.warning(f"{split_name} set is empty — check date ranges.")
                Xs = np.zeros((1, self.lookback, len(self.feature_cols)))
                ys = np.zeros((1, len(self.target_cols)))
            dataset = EnvironmentalSequenceDataset(Xs, ys, ds)
            return DataLoader(dataset, batch_size=self.batch_size,
                              shuffle=shuffle, drop_last=False)

        train_loader = make_loader("train", shuffle_train)
        val_loader   = make_loader("val",   False)
        test_loader  = make_loader("test",  False)

        return train_loader, val_loader, test_loader

    def save_sequences(self, output_dir: Optional[str] = None) -> None:
        """
        Save sequence arrays to disk for later use without re-running.

        Args:
            output_dir: Directory to save .npy files.
        """
        output_dir = output_dir or cfg.data.processed.sequences_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        X, y, dates = self.build_sequences()
        splits = self.temporal_split(X, y, dates)

        for split, (Xs, ys, ds) in splits.items():
            np.save(f"{output_dir}/{split}_X.npy", Xs)
            np.save(f"{output_dir}/{split}_y.npy", ys)
            ds.to_frame().to_csv(f"{output_dir}/{split}_dates.csv")

        log.info(f"Sequences saved to {output_dir}")


def load_sequences(sequences_dir: Optional[str] = None) -> dict:
    """
    Load pre-saved sequence arrays from disk.

    Args:
        sequences_dir: Directory containing .npy files.

    Returns:
        splits: Dict with 'train', 'val', 'test' keys.
    """
    sequences_dir = sequences_dir or cfg.data.processed.sequences_dir
    splits = {}

    for split in ["train", "val", "test"]:
        X_path = Path(sequences_dir) / f"{split}_X.npy"
        y_path = Path(sequences_dir) / f"{split}_y.npy"

        if X_path.exists() and y_path.exists():
            X = np.load(str(X_path))
            y = np.load(str(y_path))
            splits[split] = (X, y)
            log.info(f"Loaded {split}: X={X.shape}, y={y.shape}")
        else:
            log.warning(f"Sequence files not found for {split} in {sequences_dir}")

    return splits


if __name__ == "__main__":
    # Test with synthetic data when real data not yet available
    log.info("Running Sequencer with synthetic placeholder data...")

    n_months = 288  # 2000–2024
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")

    # Synthetic feature matrix
    feature_cols = cfg.features.pool
    target_cols  = cfg.features.targets

    synthetic = pd.DataFrame(
        np.random.randn(n_months, len(feature_cols)),
        index=dates,
        columns=feature_cols
    )

    seq = Sequencer(
        feature_matrix=synthetic,
        feature_cols=feature_cols,
        target_cols=target_cols,
        lookback=12,
        batch_size=16
    )

    train_loader, val_loader, test_loader = seq.get_dataloaders()

    for batch_X, batch_y in train_loader:
        log.info(f"Train batch: X={batch_X.shape}, y={batch_y.shape}")
        break

    log.info("Sequencer test passed.")
