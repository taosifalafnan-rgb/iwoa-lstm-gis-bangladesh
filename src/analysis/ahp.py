"""
ahp.py — Analytic Hierarchy Process weight computation.

Computes:
  - Priority vector (weights) from pairwise comparison matrix
  - Consistency Index (CI)
  - Consistency Ratio (CR)
  - Validation: CR must be < 0.10

Uses matrix from configs/config.yaml — ahp.pairwise_matrix
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


class AHP:
    """
    Analytic Hierarchy Process for HHI sub-index weight derivation.

    Usage:
        ahp = AHP()
        weights, cr = ahp.compute()
        # weights: dict mapping criterion name to weight value
        # cr: float, must be < 0.10
    """

    def __init__(self, pairwise_matrix: list = None, criteria: list = None):
        """
        Args:
            pairwise_matrix: n×n comparison matrix (list of lists).
                             Defaults to cfg.ahp.pairwise_matrix.
            criteria:        List of criterion names.
                             Defaults to cfg.ahp.criteria.
        """
        raw_matrix = pairwise_matrix or cfg.ahp.pairwise_matrix
        self.criteria = criteria or cfg.ahp.criteria
        self.n = len(self.criteria)
        self.matrix = np.array(raw_matrix, dtype=float)

        assert self.matrix.shape == (self.n, self.n), \
            f"Matrix shape {self.matrix.shape} does not match n={self.n}"

        log.info(f"AHP initialized: {self.n} criteria — {self.criteria}")

    def normalize_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the pairwise comparison matrix.
        Each cell divided by its column sum.

        Returns:
            norm_matrix: Normalized matrix [n, n].
            col_sums:    Column sums [n].
        """
        col_sums = self.matrix.sum(axis=0)
        norm_matrix = self.matrix / col_sums
        return norm_matrix, col_sums

    def priority_vector(self, norm_matrix: np.ndarray) -> np.ndarray:
        """
        Compute priority vector as row means of normalized matrix.
        This is the weight vector.

        Args:
            norm_matrix: Normalized comparison matrix [n, n].

        Returns:
            weights: Priority vector [n] — sums to 1.0.
        """
        weights = norm_matrix.mean(axis=1)
        assert abs(weights.sum() - 1.0) < 1e-6, \
            f"Weights do not sum to 1: {weights.sum()}"
        return weights

    def lambda_max(self, weights: np.ndarray) -> float:
        """
        Compute principal eigenvalue λ_max for consistency checking.
        λ_max = mean of (A·w)_i / w_i for all i.

        Args:
            weights: Priority vector [n].

        Returns:
            lam_max: Principal eigenvalue.
        """
        weighted_sum = self.matrix @ weights           # [n]
        ratios       = weighted_sum / weights          # [n]
        lam_max      = ratios.mean()
        return lam_max

    def consistency_ratio(self, lam_max: float) -> Tuple[float, float]:
        """
        Compute Consistency Index (CI) and Consistency Ratio (CR).

        CI = (λ_max - n) / (n - 1)
        CR = CI / RI

        Args:
            lam_max: Principal eigenvalue.

        Returns:
            ci: Consistency Index.
            cr: Consistency Ratio. Must be < 0.10.
        """
        # ri_values is loaded from config as a plain dict with integer keys
        # (see config loader). Fall back to the standard Saaty table if absent.
        ri_values = cfg.ahp.ri_values if isinstance(cfg.ahp.ri_values, dict) \
                    else {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                          6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

        ci = (lam_max - self.n) / (self.n - 1)
        # Accept either int or str keys for robustness.
        ri = ri_values.get(self.n, ri_values.get(str(self.n), 1.12))
        cr = ci / ri if ri > 0 else 0.0

        return ci, cr

    def compute(self) -> Tuple[dict, float]:
        """
        Run full AHP computation and return weights + CR.

        Returns:
            weights_dict: {criterion_name: weight_value} — sums to 1.0.
            cr:           Consistency Ratio. Must be < 0.10.

        Raises:
            ValueError: If CR >= 0.10 (inconsistent judgments).
        """
        log.info("Running AHP computation...")

        norm_matrix, col_sums = self.normalize_matrix()
        weights = self.priority_vector(norm_matrix)
        lam_max = self.lambda_max(weights)
        ci, cr  = self.consistency_ratio(lam_max)

        weights_dict = {name: float(w) for name, w in
                        zip(self.criteria, weights)}

        log.info(f"AHP Results:")
        log.info(f"  λ_max = {lam_max:.4f}")
        log.info(f"  CI    = {ci:.4f}")
        log.info(f"  CR    = {cr:.4f} (threshold: {cfg.ahp.max_cr})")

        for name, w in weights_dict.items():
            log.info(f"  {name:20s}: {w:.4f} ({w*100:.1f}%)")

        if cr >= cfg.ahp.max_cr:
            log.error(f"CR={cr:.4f} >= {cfg.ahp.max_cr} — judgments are INCONSISTENT.")
            log.error("Revise the pairwise comparison matrix in configs/config.yaml")
            raise ValueError(
                f"AHP Consistency Ratio {cr:.4f} exceeds maximum {cfg.ahp.max_cr}. "
                f"Revise pairwise comparison matrix."
            )
        else:
            log.info(f"CR={cr:.4f} < {cfg.ahp.max_cr} — judgments are CONSISTENT ✓")

        # Save to outputs
        self._save(weights_dict, lam_max, ci, cr)

        return weights_dict, cr

    def _save(self, weights: dict, lam_max: float, ci: float, cr: float):
        Path("outputs/results").mkdir(parents=True, exist_ok=True)
        # Cast to native Python types — numpy scalars are not JSON serializable.
        result = {
            "weights":  {k: float(v) for k, v in weights.items()},
            "lambda_max": float(lam_max),
            "CI":       float(ci),
            "CR":       float(cr),
            "consistent": bool(cr < cfg.ahp.max_cr)
        }
        with open("outputs/results/ahp_weights.json", "w") as f:
            json.dump(result, f, indent=2)
        log.info("AHP weights saved: outputs/results/ahp_weights.json")


if __name__ == "__main__":
    ahp = AHP()
    weights, cr = ahp.compute()
    print(f"\nAHP weights: {weights}")
    print(f"CR = {cr:.4f}")
