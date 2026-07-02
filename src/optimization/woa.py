"""
woa.py — Standard Whale Optimization Algorithm (baseline).

This is the *unimproved* WOA used as a comparison baseline against IWOA in the
paper (WOA-LSTM vs IWOA-LSTM). Rather than re-implement the search loop and risk
subtle divergence from IWOA, WOA is defined as `IWOA` with every improvement
switched OFF:

    ┌────────────────────┬──────────┬──────────┐
    │ Component          │  IWOA    │  WOA      │
    ├────────────────────┼──────────┼──────────┤
    │ OBL initialization │  on      │  off      │
    │ Nonlinear a(t)     │  on      │  off (lin)│
    │ Inertia weighting  │  on      │  off      │
    │ Levy-flight kick   │  on      │  off      │
    └────────────────────┴──────────┴──────────┘

Because both optimizers share the identical encoding, decoder, fitness call, and
bookkeeping, any performance gap is attributable purely to the four improvements
— which is exactly what a fair ablation requires.

Reference:
    Mirjalili & Lewis (2016), "The Whale Optimization Algorithm",
    Advances in Engineering Software.
"""

from typing import Optional, List

from src.optimization.iwoa import IWOA, IWOAResult
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


class WOA(IWOA):
    """
    Standard Whale Optimization Algorithm (baseline for IWOA).

    Usage:
        from src.optimization.fitness import FitnessEvaluator
        evaluator = FitnessEvaluator(feature_matrix)
        woa = WOA(fitness_fn=evaluator.evaluate)
        result = woa.optimize()   # returns an IWOAResult, same as IWOA
    """

    def __init__(
        self,
        fitness_fn,
        feature_names: Optional[List[str]] = None,
        n_whales:      Optional[int] = None,
        max_iter:      Optional[int] = None,
        max_features:  Optional[int] = None,
    ):
        """
        Args:
            fitness_fn:    Callable(features, hp) → float (val RMSE).
            feature_names: Feature pool. Defaults to cfg.features.pool.
            n_whales:      Population size. Defaults to cfg.iwoa.n_whales.
            max_iter:      Iterations. Defaults to cfg.iwoa.max_iter.
            max_features:  Max features to select. Defaults to cfg.iwoa.max_features.

        All four IWOA improvements are forced off, yielding the canonical WOA.
        """
        super().__init__(
            fitness_fn    = fitness_fn,
            feature_names = feature_names,
            n_whales      = n_whales,
            max_iter      = max_iter,
            max_features  = max_features,
            obl_enabled   = False,   # no Opposition-Based Learning
            use_inertia   = False,   # no inertia-weight blending
            use_levy      = False,   # no Levy-flight perturbation
            nonlinear_a   = False,   # classic linear a(t) = 2 − 2·t/T
        )
        log.info("WOA (baseline) initialized — all IWOA improvements disabled.")

    def _save_results(self, result: IWOAResult) -> None:
        """
        Save baseline WOA results under distinct filenames so they never
        overwrite the IWOA outputs (needed for the side-by-side comparison).

        Args:
            result: The WOA optimization result.
        """
        import json
        import pandas as pd
        from pathlib import Path
        from dataclasses import asdict

        Path("outputs/results").mkdir(parents=True, exist_ok=True)

        cfg_dict = {k: v for k, v in asdict(result).items()
                    if k != "convergence_history"}
        with open("outputs/results/woa_best_config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

        with open("outputs/results/woa_best_features.json", "w") as f:
            json.dump({"selected_features": result.selected_features,
                       "selected_indices":  result.selected_indices,
                       "n_selected":        result.n_selected}, f, indent=2)

        pd.DataFrame({
            "iteration": range(len(result.convergence_history)),
            "best_rmse": result.convergence_history,
        }).to_csv("outputs/results/woa_convergence.csv", index=False)

        log.info("WOA baseline results saved (woa_*.json / woa_convergence.csv)")


if __name__ == "__main__":
    import numpy as np
    log.info("WOA baseline smoke test with random fitness...")

    def dummy_fitness(features, hp):
        return np.random.uniform(0.1, 1.0) + 0.01 * (len(features) - 10) ** 2

    woa = WOA(fitness_fn=dummy_fitness, n_whales=5, max_iter=10)
    result = woa.optimize()
    log.info(f"WOA smoke test passed. Selected {result.n_selected} features, "
             f"best fitness {result.best_fitness:.4f}")
