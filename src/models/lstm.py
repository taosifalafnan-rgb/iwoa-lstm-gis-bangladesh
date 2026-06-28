"""
lstm.py — Multi-output stacked LSTM with Monte Carlo Dropout.

Architecture:
    Input [batch, lookback, n_features]
      → LSTM Layer 1 (hidden_1, dropout)
      → LSTM Layer 2 (hidden_2, dropout)
      → BatchNorm
      → Dense (hidden_2 // 2) + ReLU
      → Dense (n_outputs)  ← [pm25, bod, lst, ndvi]

MC Dropout: dropout remains active at inference for uncertainty estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.device import get_device

log = get_logger(__name__)


class EnvironmentalLSTM(nn.Module):
    """
    Stacked multi-output LSTM for environmental variable prediction.

    Supports Monte Carlo Dropout for uncertainty quantification.
    All hyperparameters should come from IWOA output.

    Args:
        input_size:   Number of input features (from IWOA feature selection).
        hidden_1:     Units in LSTM layer 1.
        hidden_2:     Units in LSTM layer 2.
        n_layers:     Number of LSTM layers (default 2).
        dropout:      Dropout rate applied between layers.
        output_size:  Number of prediction targets (default 4).
        bidirectional: Whether to use bidirectional LSTM (default False).
    """

    def __init__(
        self,
        input_size:    int   = None,
        hidden_1:      int   = None,
        hidden_2:      int   = None,
        n_layers:      int   = None,
        dropout:       float = None,
        output_size:   int   = None,
        bidirectional: bool  = False,
    ):
        super(EnvironmentalLSTM, self).__init__()

        # Load from config if not provided
        self.input_size    = input_size  or cfg.lstm.input_size
        self.hidden_1      = hidden_1    or cfg.lstm.hidden_1
        self.hidden_2      = hidden_2    or cfg.lstm.hidden_2
        self.n_layers      = n_layers    or cfg.lstm.n_layers
        self.dropout_rate  = dropout     or cfg.lstm.dropout
        self.output_size   = output_size or cfg.lstm.output_size
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        # LSTM Layer 1
        self.lstm1 = nn.LSTM(
            input_size   = self.input_size,
            hidden_size  = self.hidden_1,
            num_layers   = 1,
            batch_first  = True,
            bidirectional = self.bidirectional,
            dropout      = 0.0
        )
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        # LSTM Layer 2
        self.lstm2 = nn.LSTM(
            input_size   = self.hidden_1 * self.directions,
            hidden_size  = self.hidden_2,
            num_layers   = 1,
            batch_first  = True,
            bidirectional = self.bidirectional,
            dropout      = 0.0
        )
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

        # Batch normalization on final hidden state
        self.bn = nn.BatchNorm1d(self.hidden_2 * self.directions)

        # Dense output layers
        fc_in = self.hidden_2 * self.directions
        self.fc1 = nn.Linear(fc_in, fc_in // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_in // 2, self.output_size)

        self._log_architecture()

    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"EnvironmentalLSTM initialized:")
        log.info(f"  Input: [{self.input_size}] → LSTM1[{self.hidden_1}] "
                 f"→ LSTM2[{self.hidden_2}] → Output[{self.output_size}]")
        log.info(f"  Dropout: {self.dropout_rate} | Bidirectional: {self.bidirectional}")
        log.info(f"  Total params: {total_params:,} | Trainable: {trainable:,}")

    def forward(
        self,
        x: torch.Tensor,
        mc_dropout: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:          Input tensor [batch, lookback, input_size].
            mc_dropout: If True, keep dropout active even in eval mode.
                        Used for Monte Carlo uncertainty estimation.

        Returns:
            output: Predictions [batch, output_size].
        """
        # LSTM Layer 1
        out1, _ = self.lstm1(x)                        # [batch, seq, hidden_1]
        out1 = self.dropout1(out1) if (self.training or mc_dropout) \
               else out1

        # LSTM Layer 2
        out2, _ = self.lstm2(out1)                     # [batch, seq, hidden_2]
        out2 = self.dropout2(out2) if (self.training or mc_dropout) \
               else out2

        # Take the last time step
        last = out2[:, -1, :]                          # [batch, hidden_2]

        # Batch norm
        last = self.bn(last)

        # Dense layers
        out = self.relu(self.fc1(last))                # [batch, hidden_2//2]
        out = self.fc2(out)                            # [batch, output_size]

        return out

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = None,
        device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout inference for uncertainty quantification.
        Runs n_samples forward passes with dropout active.

        Args:
            x:        Input tensor [batch, lookback, input_size].
            n_samples: Number of MC samples. Defaults to cfg.mc_dropout.n_samples.
            device:   Compute device.

        Returns:
            mean:  Mean prediction [batch, output_size].
            lower: Lower confidence bound [batch, output_size].
            upper: Upper confidence bound [batch, output_size].
        """
        n_samples = n_samples or cfg.mc_dropout.n_samples
        device    = device    or get_device()
        conf      = cfg.mc_dropout.confidence  # e.g. 0.95

        x = x.to(device)
        self.eval()  # Disable train-mode batch norm, but MC dropout stays active

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, mc_dropout=True)
                predictions.append(pred.cpu().numpy())

        predictions = np.stack(predictions, axis=0)  # [n_samples, batch, output_size]
        mean  = predictions.mean(axis=0)
        alpha = (1 - conf) / 2
        lower = np.quantile(predictions, alpha,     axis=0)
        upper = np.quantile(predictions, 1 - alpha, axis=0)

        return mean, lower, upper


def build_lstm_from_iwoa(iwoa_result) -> EnvironmentalLSTM:
    """
    Build EnvironmentalLSTM using hyperparameters from IWOA result.

    Args:
        iwoa_result: IWOAResult object from iwoa.py.

    Returns:
        model: Configured EnvironmentalLSTM.
    """
    model = EnvironmentalLSTM(
        input_size  = iwoa_result.n_selected,
        hidden_1    = iwoa_result.lstm_hidden_1,
        hidden_2    = iwoa_result.lstm_hidden_2,
        n_layers    = 2,
        dropout     = iwoa_result.lstm_dropout,
        output_size = len(cfg.features.targets)
    )
    log.info(f"LSTM built from IWOA: {iwoa_result.n_selected} inputs, "
             f"h1={iwoa_result.lstm_hidden_1}, h2={iwoa_result.lstm_hidden_2}")
    return model


def load_model(checkpoint_path: str) -> EnvironmentalLSTM:
    """
    Load a saved model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.

    Returns:
        model: Loaded EnvironmentalLSTM in eval mode.
    """
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_args = checkpoint.get("model_args", {})
    model = EnvironmentalLSTM(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    log.info(f"Model loaded from {checkpoint_path}")
    return model


if __name__ == "__main__":
    device = get_device()

    # Test with synthetic batch
    model = EnvironmentalLSTM(
        input_size  = 12,
        hidden_1    = 64,
        hidden_2    = 32,
        output_size = 4
    )
    model.to(device)

    batch_x = torch.randn(16, 12, 12).to(device)   # [batch=16, lookback=12, features=12]
    output  = model(batch_x)
    log.info(f"Forward pass test: input={batch_x.shape} → output={output.shape}")

    mean, lower, upper = model.predict_with_uncertainty(batch_x, n_samples=10)
    log.info(f"MC Dropout test: mean={mean.shape}, lower={lower.shape}, upper={upper.shape}")
    log.info("LSTM module test passed.")
