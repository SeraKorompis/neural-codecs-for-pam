#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_line_macro_metrics_vs_bitrate.py
2x2 line plot of macro metrics vs bitrate for one or more datasets.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ENCODEC_COLOR = '#1f77b4'
DAC_COLOR     = '#ff7f0e'
MP3_COLOR     = '#2ca02c'
OPUS_COLOR    = '#d62728'

FONTSIZE      = 20
TICK_SIZE     = 18
LEGEND_SIZE   = 16
LINEWIDTH     = 2.5
MARKERSIZE    = 8

LINE_STYLES = ['-', '--', ':']

DAC_KBPS = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}

def load_and_normalise(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    aliases = {
        'roc_auc':   ['macro_auc', 'macro_roc_auc', 'auc'],
        'f1':        ['macro_f1', 'f1_fixed_f1', 'f1_score', 'f1_macro'],
        'precision': ['macro_precision', 'precision_fixed_f1', 'precision_macro'],
        'recall':    ['macro_recall', 'recall_fixed_f1', 'recall_macro'],
        'map':       ['macro_map', 'mean_ap', 'mAP'],
    }
    for target_col, candidate_cols in aliases.items():
        if target_col not in df.columns:
            for cand in candidate_cols:
                if cand in df.columns:
                    df = df.rename(columns={cand: target_col})
                    break
    if 'kbps' not in df.columns and 'bitrate' in df.columns:
        df['kbps'] = pd.to_numeric(df['bitrate'], errors='coerce')
    # Parse codec_name and kbps from condition if needed
    if 'codec' not in df.columns and 'condition' in df.columns:
        def get_codec(c):
            if c == 'original': return 'original'
            for x in ['encodec', 'dac', 'mp3', 'opus']:
                if c.startswith(x): return x
            return 'unknown'
        def get_kbps(row):
            c = row['condition']
            if c == 'original': return None
            for x in ['encodec', 'dac', 'mp3', 'opus']:
                if c.startswith(x):
                    br = float(c.replace(x, ''))
                    if x == 'dac':
                        return DAC_KBPS.get(int(br), br)
                    return br
            return None
        df['codec'] = df['condition'].apply(get_codec)
        df['kbps']  = df.apply(get_kbps, axis=1)
    return df


def plot_dataset(ax, df, col, label_prefix, linestyle, legend_ax, ax_idx):
    """Plot all codec lines for one dataset on one axis."""
    orig    = df[df['codec'] == 'original']
    encodec = df[df['codec'] == 'encodec'].sort_values('kbps')
    dac     = df[df['codec'] == 'dac'].sort_values('kbps')
    mp3     = df[df['codec'] == 'mp3'].sort_values('kbps')
    opus    = df[df['codec'] == 'opus'].sort_values('kbps')

    if col not in df.columns:
        return

    if not orig.empty:
        baseline = float(orig[col].iloc[0])
        ax.axhline(baseline, color='black', linestyle='--',
                   linewidth=2.0, alpha=0.7,