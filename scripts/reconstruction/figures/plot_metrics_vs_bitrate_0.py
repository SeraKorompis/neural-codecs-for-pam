#!/usr/bin/env python3
"""
scripts/reconstruction/figures/plot_metrics_vs_bitrate.py
Driver script for plot_metrics_vs_bitrate_whole.
Loads merged metrics CSV and calls the plotting function.
Filters to: all DAC bitrates, EnCodec 24.0 kbps only,
            MP3 8, 16, 24 kbps only, Opus 8, 14, 24 kbps only
            — allowing comparison with paper baselines at comparable bitrates.
"""
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import PATHS
from src.figures import plot_metrics_vs_bitrate_whole

METRICS  = Path(PATHS['results_dir']) / 'reconstruction_quality' / 'metrics_merged.csv'
FIGS_DIR = Path(PATHS['figures_dir']) / 'reconstruction_quality'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading metrics from {METRICS}...")
metrics = pd.read_csv(METRICS)

# Keep all EnCodec and DAC bitrates
# MP3 8, 16, 24 kbps, comparable to neural codec bitrates
# Opus 8, 14, 24 kbps, matching DAC paper performance for Opus
MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]

metrics = metrics[
    ~((metrics['codec'] == 'mp3')     & (~metrics['bitrate'].isin(MP3_KEEP))) &
    ~((metrics['codec'] == 'opus')    & (~metrics['bitrate'].isin(OPUS_KEEP)))
]

print(f"  {len(metrics)} rows, datasets: {metrics['dataset'].unique().tolist()}")
print(f"  Codecs/bitrates:")
print(metrics.groupby(['codec', 'bitrate']).size().to_string())

plot_metrics_vs_bitrate_whole(
    metrics  = metrics,
    figs_dir = FIGS_DIR,
    sharey   = True,
)