"""
iwoa.py — Improved Whale Optimization Algorithm (IWOA)

Improvements over standard WOA:
  1. Opposition-Based Learning (OBL) initialization
  2. Nonlinear inertia weight decay
  3. Levy flight perturbation (Mantegna's algorithm)
  4. Adaptive convergence factor a(t)

Tasks:
  - Feature selection (binary encoding, selects subset of 23 variables)
  - LSTM hyperparameter tuning (continuous encoding)

Fitness function: Validation RMSE from proxy mini-LSTM
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, field, asdict

from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


# ── RESULT DATACLASS ───────────────────────────────────────────────────────

@dataclass
class IWOAResult:
    """Container for IWOA optimization output."""
    selected_features:      List[str]   = field(default_factory=list)
    selected_indices:       List[int]   = field(default_factory=list)
    n_selected:             int         = 0
    lstm_hidden_1:          int         = 128
    lstm_hidden_2:          int         = 64
    lstm_dropout:           float       = 0.3
    lstm_learning_rate:     float       = 0.001
    lstm_batch_size:        int         = 16
    lstm_lookback:          int         = 12
    best_fitness:           float       = float("inf")
    convergence_history:    List[float] = field(default_factory=list)
    n_iterations_run:       int         = 0


# ── LEVY FLIGHT (Mantegna's Algorithm) ────────────────────────────────────

def levy_flight(n: int, m: int, lam: float = 1.5) -> np.ndarray:
    """
    Generate Levy flight steps using Mantegna's algorithm.

    Args:
        n:   Number of search agents (whales).
        m:   Dimension of each agent's position vector.
        lam: Levy exponent. Must be in (1, 3]. Default 1.5.

    Returns:
        steps: Levy flight step array [n, m].
    """
    num = np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2)
    den = np.math.gamma((1 + lam) / 2) * lam * (2 ** ((lam - 1) / 2))
    sigma_u = (num / den) ** (1 / lam)

    u = np.random.normal(0, sigma_u, size=(n, m))
    v = np.random.normal(0, 1,       size=(n, m))

    steps = u / (np.abs(v) ** (1 / lam))
    return steps


# ── OPPOSITION-BASED LEARNING ─────────────────────────────────────────────

def obl_initialization(
    population: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray
) -> np.ndarray:
    """
    Generate opposite population using Opposition-Based Learning.
    For each agent x_i in [lb, ub], compute x_opp = lb + ub - x_i.
    Return the better of each pair.

    Args:
        population: Initial population [n_whales, dim].
        lb:         Lower bounds [dim].
        ub:         Upper bounds [dim].

    Returns:
        improved_pop: Population after OBL initialization [n_whales, dim].
    """
    opposite = lb + ub - population
    opposite = np.clip(opposite, lb, ub)

    # Stack and return — fitness evaluation selects the better half
    # Since we don't evaluate fitness here, return both for caller to evaluate
    combined = np.vstack([population, opposite])
    return combined


# ── IWOA CORE ──────────────────────────────────────────────────────────────

class IWOA:
    """
    Improved Whale Optimization Algorithm for joint feature selection
    and LSTM hyperparameter optimization.

    Encoding:
        Position vector dim = n_features (binary) + n_hyperparams (continuous)
        Total dim = 23 + 6 = 29

    Binary part  [0:23]:  1.0 = include feature, 0.0 = exclude
    Continuous part [23:]: normalized hyperparameter values

    Usage:
        from src.optimization.fitness import FitnessEvaluator
        evaluator = FitnessEvaluator(train_loader, val_loader)
        iwoa = IWOA(fitness_fn=evaluator.evaluate)
        result = iwoa.optimize()
    """

    # Hyperparameter bounds (normalized 0-1 internally, decoded at evaluation)
    HP_BOUNDS = {
        "hidden_1":      [32,    256],
        "hidden_2":      [16,    128],
        "dropout":       [0.1,   0.5],
        "learning_rate": [1e-4,  1e-2],
        "batch_size":    [8,     64],
        "lookback":      [6,     24],
    }
    N_HP = len(HP_BOUNDS)  # 6

    def __init__(
        self,
        fitness_fn,
        feature_names:  Optional[List[str]] = None,
        n_whales:       Optional[int]   = None,
        max_iter:       Optional[int]   = None,
        max_features:   Optional[int]   = None,
        levy_lambda:    Optional[float] = None,
        omega_max:      Optional[float] = None,
        omega_min:      Optional[float] = None,
        obl_enabled:    Optional[bool]  = None,
    ):
        """
        Args:
            fitness_fn:    Callable(features: List[str], hp: dict) → float (RMSE).
            feature_names: Names of all features in pool. Defaults to cfg.features.pool.
            n_whales:      Population size. Defaults to cfg.iwoa.n_whales.
            max_iter:      Maximum iterations. Defaults to cfg.iwoa.max_iter.
            max_features:  Max features to select. Defaults to cfg.iwoa.max_features.
            levy_lambda:   Levy exponent. Defaults to cfg.iwoa.levy_lambda.
            omega_max:     Max inertia weight. Defaults to cfg.iwoa.omega_max.
            omega_min:     Min inertia weight. Defaults to cfg.iwoa.omega_min.
            obl_enabled:   Use OBL initialization. Defaults to cfg.iwoa.obl_enabled.
        """
        set_seed(cfg.seed)

        self.fitness_fn    = fitness_fn
        self.feature_names = feature_names or cfg.features.pool
        self.n_features    = len(self.feature_names)
        self.dim           = self.n_features + self.N_HP

        self.n_whales    = n_whales    or cfg.iwoa.n_whales
        self.max_iter    = max_iter    or cfg.iwoa.max_iter
        self.max_features = max_features or cfg.iwoa.max_features
        self.levy_lambda = levy_lambda or cfg.iwoa.levy_lambda
        self.omega_max   = omega_max   or cfg.iwoa.omega_max
        self.omega_min   = omega_min   or cfg.iwoa.omega_min
        self.obl_enabled = obl_enabled if obl_enabled is not None else cfg.iwoa.obl_enabled

        # Bounds
        self.lb = np.zeros(self.dim)   # all 0
        self.ub = np.ones(self.dim)    # all 1

        log.info(f"IWOA initialized: {self.n_whales} whales, "
                 f"{self.max_iter} iterations, dim={self.dim}")
        log.info(f"  OBL: {self.obl_enabled} | Levy λ={self.levy_lambda} | "
                 f"ω=[{self.omega_min},{self.omega_max}]")

    def _decode_position(self, pos: np.ndarray) -> Tuple[List[str], dict]:
        """
        Decode a position vector into feature list and hyperparameter dict.

        Args:
            pos: Position vector [dim].

        Returns:
            selected_features: List of feature names.
            hp:                 LSTM hyperparameter dict.
        """
        # Binary part: sigmoid threshold at 0.5
        binary = pos[:self.n_features]
        selected_idx = np.where(binary >= 0.5)[0]

        # Enforce max_features constraint
        if len(selected_idx) > self.max_features:
            # Keep the ones closest to 1.0 (strongest selection signal)
            top_idx = np.argsort(binary)[::-1][:self.max_features]
            selected_idx = np.sort(top_idx)

        # Ensure at least 3 features always selected
        if len(selected_idx) < 3:
            top3 = np.argsort(binary)[::-1][:3]
            selected_idx = np.sort(top3)

        selected_features = [self.feature_names[i] for i in selected_idx]

        # Continuous part: decode hyperparameters
        hp_vals = pos[self.n_features:]
        hp_keys = list(self.HP_BOUNDS.keys())
        hp = {}
        for i, key in enumerate(hp_keys):
            lb, ub = self.HP_BOUNDS[key]
            val = lb + hp_vals[i] * (ub - lb)
            # Round discrete HPs
            if key in ["hidden_1", "hidden_2", "batch_size", "lookback"]:
                val = int(round(val))
                # Ensure even numbers for hidden units
                if key in ["hidden_1", "hidden_2"]:
                    val = max(16, (val // 16) * 16)
            hp[key] = val

        return selected_features, hp

    def _inertia_weight(self, t: int) -> float:
        """
        Nonlinear inertia weight decay (Improvement 1).
        ω(t) = ω_max - (ω_max - ω_min) * (t / T_max)²

        Args:
            t: Current iteration.

        Returns:
            omega: Inertia weight at iteration t.
        """
        return self.omega_max - (self.omega_max - self.omega_min) * (t / self.max_iter) ** 2

    def _adaptive_a(self, t: int) -> float:
        """
        Adaptive nonlinear convergence factor a(t) (Improvement 4).
        a(t) = 2 - 2 * (t / T_max)²

        Args:
            t: Current iteration.

        Returns:
            a: Convergence factor at iteration t.
        """
        return 2 - 2 * (t / self.max_iter) ** 2

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with optional OBL (Improvement 2).

        Returns:
            population: Initial population [n_whales, dim].
        """
        pop = np.random.uniform(0, 1, size=(self.n_whales, self.dim))

        if self.obl_enabled:
            log.info("Applying Opposition-Based Learning initialization...")
            combined = obl_initialization(pop, self.lb, self.ub)
            # Evaluate fitness for all 2*n_whales candidates
            # Select the n_whales best
            fitnesses = []
            for i in range(len(combined)):
                feats, hp = self._decode_position(combined[i])
                try:
                    f = self.fitness_fn(feats, hp)
                except Exception:
                    f = float("inf")
                fitnesses.append(f)

            fitnesses = np.array(fitnesses)
            best_idx = np.argsort(fitnesses)[:self.n_whales]
            pop = combined[best_idx]
            log.info(f"OBL: selected best {self.n_whales} from {len(combined)} candidates")

        return np.clip(pop, self.lb, self.ub)

    def optimize(self) -> IWOAResult:
        """
        Run the full IWOA optimization loop.

        Returns:
            result: IWOAResult with best features, hyperparameters, and convergence.
        """
        log.info("=" * 60)
        log.info("IWOA OPTIMIZATION START")
        log.info("=" * 60)

        # Initialize
        population = self._initialize_population()
        fitness    = np.full(self.n_whales, float("inf"))

        # Evaluate initial fitness
        log.info("Evaluating initial population fitness...")
        for i in range(self.n_whales):
            feats, hp = self._decode_position(population[i])
            try:
                fitness[i] = self.fitness_fn(feats, hp)
            except Exception as e:
                log.warning(f"Fitness eval failed for whale {i}: {e}")
                fitness[i] = float("inf")

        best_idx     = np.argmin(fitness)
        best_pos     = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        convergence  = [best_fitness]

        log.info(f"Initial best fitness (RMSE): {best_fitness:.6f}")

        # ── MAIN LOOP ───────────────────────────────────────
        for t in range(1, self.max_iter + 1):

            omega = self._inertia_weight(t)
            a     = self._adaptive_a(t)

            for i in range(self.n_whales):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A  = 2 * a * r1 - a                        # Coefficient vector A
                C  = 2 * r2                                 # Coefficient vector C
                l  = np.random.uniform(-1, 1, self.dim)    # Spiral parameter
                p  = np.random.random()                     # Switch probability

                if p < 0.5:
                    if np.abs(A).mean() < 1:
                        # Encircling prey (exploitation)
                        D = np.abs(C * best_pos - population[i])
                        new_pos = best_pos - A * D
                    else:
                        # Search for prey (exploration)
                        rand_idx = np.random.randint(0, self.n_whales)
                        rand_pos = population[rand_idx]
                        D = np.abs(C * rand_pos - population[i])
                        new_pos = rand_pos - A * D
                else:
                    # Bubble-net spiral attack
                    D_prime = np.abs(best_pos - population[i])
                    new_pos = (D_prime * np.exp(cfg.iwoa.b_constant * l)
                               * np.cos(2 * np.pi * l) + best_pos)

                # Inertia weight (Improvement 1)
                new_pos = omega * new_pos + (1 - omega) * population[i]

                # Levy flight perturbation (Improvement 3)
                # Apply with 30% probability to escape local optima
                if np.random.random() < 0.3:
                    levy = levy_flight(1, self.dim, self.levy_lambda)[0]
                    new_pos = new_pos + 0.01 * levy * (new_pos - best_pos)

                population[i] = np.clip(new_pos, self.lb, self.ub)

            # Evaluate updated population
            for i in range(self.n_whales):
                feats, hp = self._decode_position(population[i])
                try:
                    f = self.fitness_fn(feats, hp)
                except Exception as e:
                    log.warning(f"Iter {t}, whale {i} fitness failed: {e}")
                    f = float("inf")

                if f < fitness[i]:
                    fitness[i] = f
                    if f < best_fitness:
                        best_fitness = f
                        best_pos     = population[i].copy()

            convergence.append(best_fitness)

            # Log progress every 25 iterations
            if t % 25 == 0 or t == 1:
                feats_now, hp_now = self._decode_position(best_pos)
                log.info(f"Iter {t:4d}/{self.max_iter} | "
                         f"Best RMSE: {best_fitness:.6f} | "
                         f"Features: {len(feats_now)} | "
                         f"LR: {hp_now['learning_rate']:.5f}")

        # ── DECODE FINAL RESULT ─────────────────────────────
        best_features, best_hp = self._decode_position(best_pos)
        best_indices = [self.feature_names.index(f) for f in best_features
                        if f in self.feature_names]

        result = IWOAResult(
            selected_features   = best_features,
            selected_indices    = best_indices,
            n_selected          = len(best_features),
            lstm_hidden_1       = best_hp.get("hidden_1", 128),
            lstm_hidden_2       = best_hp.get("hidden_2", 64),
            lstm_dropout        = best_hp.get("dropout", 0.3),
            lstm_learning_rate  = best_hp.get("learning_rate", 0.001),
            lstm_batch_size     = best_hp.get("batch_size", 16),
            lstm_lookback       = best_hp.get("lookback", 12),
            best_fitness        = best_fitness,
            convergence_history = convergence,
            n_iterations_run    = self.max_iter,
        )

        log.info("=" * 60)
        log.info("IWOA OPTIMIZATION COMPLETE")
        log.info(f"Best RMSE:          {result.best_fitness:.6f}")
        log.info(f"Selected features:  {result.n_selected}/{self.n_features}")
        log.info(f"  Features: {result.selected_features}")
        log.info(f"LSTM config:")
        log.info(f"  hidden_1={result.lstm_hidden_1}, hidden_2={result.lstm_hidden_2}")
        log.info(f"  dropout={result.lstm_dropout:.3f}, lr={result.lstm_learning_rate:.6f}")
        log.info(f"  batch_size={result.lstm_batch_size}, lookback={result.lstm_lookback}")
        log.info("=" * 60)

        # Save results
        self._save_results(result)

        return result

    def _save_results(self, result: IWOAResult) -> None:
        """Save IWOA results to JSON and convergence to CSV."""
        Path("outputs/results").mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = cfg.iwoa.results_file
        config_dict = {k: v for k, v in asdict(result).items()
                       if k != "convergence_history"}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        log.info(f"IWOA config saved: {config_path}")

        # Save features
        features_path = cfg.iwoa.features_file
        with open(features_path, "w") as f:
            json.dump({
                "selected_features": result.selected_features,
                "selected_indices":  result.selected_indices,
                "n_selected":        result.n_selected
            }, f, indent=2)
        log.info(f"IWOA features saved: {features_path}")

        # Save convergence
        conv_path = cfg.iwoa.convergence_file
        conv_df = pd.DataFrame({
            "iteration": range(len(result.convergence_history)),
            "best_rmse": result.convergence_history
        })
        conv_df.to_csv(conv_path, index=False)
        log.info(f"Convergence curve saved: {conv_path}")


def load_iwoa_result(results_file: Optional[str] = None,
                     features_file: Optional[str] = None) -> IWOAResult:
    """
    Load previously saved IWOA results from JSON files.
    Use this to skip re-running IWOA after first optimization.

    Args:
        results_file:  Path to iwoa_best_config.json.
        features_file: Path to iwoa_best_features.json.

    Returns:
        result: Reconstructed IWOAResult object.
    """
    results_file  = results_file  or cfg.iwoa.results_file
    features_file = features_file or cfg.iwoa.features_file

    with open(results_file, "r") as f:
        config = json.load(f)
    with open(features_file, "r") as f:
        features = json.load(f)

    result = IWOAResult(**config, **features)
    log.info(f"IWOA result loaded: {result.n_selected} features, "
             f"best RMSE={result.best_fitness:.6f}")
    return result


if __name__ == "__main__":
    log.info("IWOA smoke test with random fitness function...")

    def dummy_fitness(features, hp):
        """Random fitness for testing — replace with real FitnessEvaluator."""
        return np.random.uniform(0.1, 1.0) + 0.01 * (len(features) - 10) ** 2

    iwoa = IWOA(
        fitness_fn    = dummy_fitness,
        n_whales      = 5,      # Small for fast test
        max_iter      = 10,
        obl_enabled   = False,  # Skip OBL for smoke test speed
    )

    result = iwoa.optimize()
    log.info(f"Smoke test passed. Selected {result.n_selected} features.")
