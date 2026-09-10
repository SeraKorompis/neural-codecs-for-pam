#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_transfer_learned_aggregated.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

RESULTS_BASE  = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'on_original'
FIGS_DIR      = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DAC_KBPS      = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}
ENCODEC_COLOR = '#1f77b4'
DAC_COLOR     = '#ff7f0e'
MP3_COLOR     = '#2ca02c'
OPUS_COLOR    = '#9467bd'
PANEL_LABELS  = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]

DATASET_CONFIG = {
    'anuraset': {
        'path'    : RESULTS_BASE / 'anuraset' / 'macro_metrics.csv',
        'roc_col' : 'macro_roc_auc',
        'prec_col': 'precision_fixed_f1',
        'rec_col' : 'recall_fixed_f1',
        'label'   : 'AnuraSet',
    },
    'lemur': {
        'path'    : RESULTS_BASE / 'lemur' / 'macro_metrics_temporal.csv',
        'roc_col' : 'roc_auc',
        'prec_col': 'precision_fixed_f1',
        'rec_col' : 'recall_fixed_f1',
        'label'   : 'Black-and-white ruffed lemur',
    },
    'northeastern': {
        'path'    : RESULTS_BASE / 'northeastern' / 'macro_metrics_ovr.csv',
        'roc_col' : 'macro_roc_auc',
        'prec_col': 'precision_fixed_f1',
        'rec_col' : 'recall_fixed_f1',
        'label'   : 'Northeastern US soundscapes',
    },
}

METRICS = [
    ('roc_col',  'ROC-AUC'),
    ('prec_col', 'Precision'),
    ('rec_col',  'Recall'),
]

FONTSIZE    = 24
TICK_SIZE   = 24
LEGEND_SIZE = 18
LINEWIDTH   = 2.5
MARKERSIZE  = 8


def filter_bitrates(df):
    if df.empty or 'codec' not in df.columns:
        return df
    df = df[~((df['codec'] == 'mp3')  & (~df['kbps'].isin(MP3_KEEP)))]
    df = df[~((df['codec'] == 'opus') & (~df['kbps'].isin(OPUS_KEEP)))]
    return df


def load_dataset(config):
    df = pd.read_csv(config['path'])
    if 'kbps' not in df.columns or df['kbps'].isna().all():
        df['kbps'] = df.apply(
            lambda r: DAC_KBPS.get(int(r['bitrate']), float(r['bitrate']))
            if r.get('codec') == 'dac' else float(r['bitrate']), axis=1)
    orig = df[df['condition'] == 'original']
    enc  = filter_bitrates(df[df['codec'] == 'encodec'].sort_values('kbps')) \
           if 'codec' in df.columns else pd.DataFrame()
    dac  = filter_bitrates(df[df['codec'] == 'dac'].sort_values('kbps')) \
           if 'codec' in df.columns else pd.DataFrame()
    mp3  = filter_bitrates(df[df['codec'] == 'mp3'].sort_values('kbps')) \
           if 'codec' in df.columns else pd.DataFrame()
    opus = filter_bitrates(df[df['codec'] == 'opus'].sort_values('kbps')) \
           if 'codec' in df.columns else pd.DataFrame()
    return orig, enc, dac, mp3, opus


def plot_metric_panel(ax, orig, enc, dac, mp3, opus,
                      metric_col, ylabel,
                      show_ylabel, show_xlabel, panel_label,
                      collect_legend=False):
    handles = []
    labels  = []

    if not orig.empty and metric_col in orig.columns:
        baseline = float(orig[metric_col].iloc[0])
        line = ax.axhline(baseline, color='black', linestyle='--',
                          linewidth=2.0, alpha=0.7)
        if collect_legend:
            handles.append(line)
            labels.append('Original')

    for data, color, marker, label in [
        (enc,  ENCODEC_COLOR, 'o', 'EnCodec'),
        (dac,  DAC_COLOR,     's', 'DAC'),
        (mp3,  MP3_COLOR,     '^', 'MP3'),
        (opus, OPUS_COLOR,    'D', 'Opus'),
    ]:
        if not data.empty and metric_col in data.columns:
            line, = ax.plot(data['kbps'], data[metric_col],
                            color=color, marker=marker,
                            linewidth=LINEWIDTH, markersize=MARKERSIZE,
                            label=label)
            if collect_legend:
                handles.append(line)
                labels.append(label)

    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=12)
    if show_xlabel:
        ax.set_xlabel('Bitrate (kbps)', fontsize=FONTSIZE, labelpad=12)

    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=FONTSIZE, va='top', ha='left')

    return handles, labels


def main():
    plt.rcParams['font.family'] = 'Arial'

    datasets = {}
    for key, config in DATASET_CONFIG.items():
        if not config['path'].exists():
            print(f"WARNING: {config['path']} not found — skipping {key}")
            continue
        try:
            orig, enc, dac, mp3, opus = load_dataset(config)
            datasets[key] = (orig, enc, dac, mp3, opus, config)
            print(f"Loaded: {key}")
        except Exception as e:
            print(f"WARNING: could not load {key}: {e} — skipping")

    if not datasets:
        print("ERROR: no datasets loaded")
        return

    n_rows = len(METRICS)
    n_cols = len(datasets)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8 * n_cols, 6 * n_rows),
        gridspec_kw={'hspace': 0.3, 'wspace': 0.22})

    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    y_ranges = {}
    for row_idx, (metric_key, metric_label) in enumerate(METRICS):
        all_vals = []
        for key, (orig, enc, dac, mp3, opus, config) in datasets.items():
            col = config[metric_key]
            for df in [orig, enc, dac, mp3, opus]:
                if not df.empty and col in df.columns:
                    all_vals.extend(df[col].dropna().tolist())
        if all_vals:
            y_ranges[row_idx] = (
                max(0, min(all_vals) - 0.05),
                min(1, max(all_vals) + 0.05))

    panel_idx      = 0
    legend_handles = []
    legend_labels  = []

    for col_idx, (key, (orig, enc, dac, mp3, opus, config)) in enumerate(datasets.items()):
        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            metric_col  = config[metric_key]
            show_ylabel = (col_idx == 0)
            show_xlabel = (row_idx == n_rows - 1)
            panel_label = PANEL_LABELS[panel_idx]
            collect     = (row_idx == 0 and col_idx == n_cols - 1)

            handles, labels = plot_metric_panel(
                ax, orig, enc, dac, mp3, opus,
                metric_col, metric_label,
                show_ylabel, show_xlabel, panel_label,
                collect_legend=collect)

            if collect:
                legend_handles = handles
                legend_labels  = labels

            if row_idx in y_ranges:
                ax.set_ylim(y_ranges[row_idx])
            if row_idx == 0:
                ax.set_title(config['label'], fontsize=FONTSIZE, pad=16)

            panel_idx += 1

    if legend_handles:
        axes[0, n_cols - 1].legend(
            legend_handles, legend_labels,
            fontsize=LEGEND_SIZE, frameon=False, loc='best')

    plt.tight_layout()
    out = FIGS_DIR / 'transfer_learned_aggregated.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()