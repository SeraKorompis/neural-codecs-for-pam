#!/usr/bin/env python3
"""
scripts/reconstruction/figures/plot_metrics_vs_bitrate.py
"""
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import PATHS
from src.figures import plot_metrics_vs_bitrate_whole

METRICS  = Path(PATHS['results_dir']) / 'reconstruction_quality' / 'reconstruction_metrics_agg.csv'
FIGS_DIR = Path(PATHS['figures_dir']) / 'reconstruction_quality'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading metrics from {METRICS}...")
metrics = pd.read_csv(METRICS)

print(f"  {len(metrics)} rows, datasets: {metrics['dataset'].unique().tolist()}")
print(f"  Codecs/bitrates:")
print(metrics.groupby(['codec', 'bitrate']).size().to_string())

plot_metrics_vs_bitrate_whole(
    metrics  = metrics,
    figs_dir = FIGS_DIR,
    sharey   = True,
)