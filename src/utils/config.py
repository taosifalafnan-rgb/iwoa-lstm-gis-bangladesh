"""
config.py — YAML config loader
Loads configs/config.yaml into a nested namespace object.
All source files import cfg from here — never load YAML directly.
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert nested dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path: str = "configs/config.yaml") -> SimpleNamespace:
    """
    Load YAML config file and return as nested SimpleNamespace.

    Args:
        config_path: Path to config.yaml relative to project root.

    Returns:
        cfg: Nested SimpleNamespace with dot-access to all parameters.

    Example:
        cfg = load_config()
        print(cfg.iwoa.n_whales)      # 30
        print(cfg.study_area.bbox)    # namespace with xmin, ymin...
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            f"Run from project root directory."
        )

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = _dict_to_namespace(raw)
    return cfg


# Singleton — import this directly in other modules
cfg = load_config()


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully.")
    print(f"  Study area: {cfg.study_area.districts}")
    print(f"  BBOX: xmin={cfg.study_area.bbox.xmin}, ymin={cfg.study_area.bbox.ymin}, "
          f"xmax={cfg.study_area.bbox.xmax}, ymax={cfg.study_area.bbox.ymax}")
    print(f"  IWOA whales: {cfg.iwoa.n_whales}")
    print(f"  IWOA max_iter: {cfg.iwoa.max_iter}")
    print(f"  LSTM hidden_1: {cfg.lstm.hidden_1}")
    print(f"  Seed: {cfg.seed}")
    print(f"  W&B enabled: {cfg.wandb.enabled}")
