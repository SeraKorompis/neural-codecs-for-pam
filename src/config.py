# src/config.py
"""
Central configuration loader.

Reads config.yaml from the project root. Copy config.yaml.example to
config.yaml and fill in your paths before running any scripts.

Usage:
    from src.config import PATHS
    SOURCE_DIR    = Path(PATHS['source_dir'])
    RESULTS_DIR = Path(PATHS['results_dir'])
"""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    config_path = PROJECT_ROOT / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}.\n"
            "Copy config.yaml.example to config.yaml and fill in your paths."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


CONFIG = load_config()
PATHS  = CONFIG['paths']