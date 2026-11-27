#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_fine_tuned_on_og_auc.py
Fine-tuned classifier macro ROC-AUC against bitrate, one panel per
dataset: (a) AnuraSet, (b) black-and-white ruffed lemur,
(c) Northeastern US soundscapes.
"""
import sys
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
# Okabe-Ito colour-blind-safe palette (same as the pre-trained figure)
ENCODEC_COLOR = '#0072B2'
DAC_COLOR     = '#E69F00'
MP3_COLOR     = '#009E73'
OPUS_COLOR    = '#CC79A7'
PANEL_LABELS  = ['(a)', '(b)', '(c)']

MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]

DATASET_CONFIG = {
    'northeastern': {
        'path'   : RESULTS_BASE / 'northeastern' / 'macro_metrics_ovr.csv',
        'roc_col': 'macro_roc_auc',
        'label'  : 'Northeastern birds',
    },
    'anuraset': {
        'path'   : RESULTS_BASE / 'anuraset' / 'macro_metrics.csv',
        'roc_col': 'macro_roc_auc',
        'label'  : 'AnuraSet',
    },
    'lemur': {
        'path'   : RESULTS_BASE / 'lemur' / 'macro_metrics_temporal.csv',
        'roc_col': 'roc_auc',
        'label'  : 'Lemur',
    },
}

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
    out = [orig]
    for codec in ['encodec', 'dac', 'mp3', 'opus']:
        sub = df[df['codec'] == codec].sort_values('kbps') \
              if 'codec' in df.columns else pd.DataFrame()
        out.append(filter_bitrates(sub))
    return out


def plot_panel(ax, orig, enc, dac, mp3, opus, metric_col,
               show_ylabel, panel_label, collect_legend=False):
    handles, labels = [], []

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
    ax.set_xlabel('Bitrate (kbps)', fontsize=FONTSIZE, labelpad=12)
    if show_ylabel:
        ax.set_ylabel('Macro ROC-AUC', fontsize=FONTSIZE, labelpad=12)
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=FONTSIZE, va='top', ha='left')
    return handles, labels


def main():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    datasets = {}
    for key, config in DATASET_CONFIG.items():
        if not config['path'].exists():
            print(f"WARNING: {config['path']} not found, skipping {key}")
            continue
        try:
            datasets[key] = (*load_dataset(config), config)
            print(f"Loaded: {key}")
        except Exception as e:
            print(f"WARNING: could not load {key}: {e}, skipping")

    if not datasets:
        print("ERROR: no datasets loaded")
        return

    n_cols = len(datasets)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6.5),
                             gridspec_kw={'wspace': 0.22}, squeeze=False)
    axes = axes[0]

    # shared y range across panels
    all_vals = []
    for orig, enc, dac, mp3, opus, config in datasets.values():
        for df in [orig, enc, dac, mp3, opus]:
            if not df.empty and config['roc_col'] in df.columns:
                all_vals.extend(df[config['roc_col']].dropna().tolist())
    y_range = (max(0, min(all_vals) - 0.05), min(1, max(all_vals) + 0.05))

    legend_handles, legend_labels = [], []
    for col_idx, (key, (orig, enc, dac, mp3, opus, config)) in enumerate(datasets.items()):
        ax = axes[col_idx]
        collect = (col_idx == n_cols - 1)
        handles, labels = plot_panel(
            ax, orig, enc, dac, mp3, opus, config['roc_col'],
            show_ylabel=(col_idx == 0), panel_label=PANEL_LABELS[col_idx],
            collect_legend=collect)
        if collect:
            legend_handles, legend_labels = handles, labels
        ax.set_ylim(y_range)
        ax.set_title(config['label'], fontsize=FONTSIZE, pad=16)

    if legend_handles:
        axes[-1].legend(legend_handles, legend_labels,
                        fontsize=LEGEND_SIZE, frameon=False, loc='best')

    plt.tight_layout()
    out = FIGS_DIR / 'fine_tuned_on_og_auc'
    plt.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}.png and {out}.pdf")


if __name__ == '__main__':
    main()