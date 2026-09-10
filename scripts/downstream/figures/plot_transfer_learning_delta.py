#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_transfer_learning_delta.py

Bar chart showing ROC-AUC improvement from transfer learning
on compressed data vs trained on original, per codec/bitrate.
For codecs with more than 2 bitrates, only lowest and highest are shown.
Layout: 1 row x N columns (one per dataset).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

RESULTS_BASE  = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned'
FIGS_DIR      = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DAC_KBPS      = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}
ENCODEC_COLOR = '#1f77b4'
DAC_COLOR     = '#ff7f0e'
MP3_COLOR     = '#2ca02c'
OPUS_COLOR    = '#9467bd'
PANEL_LABELS  = ['(a)', '(b)', '(c)']

MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]

CODEC_COLORS = {
    'encodec': ENCODEC_COLOR,
    'dac'    : DAC_COLOR,
    'mp3'    : MP3_COLOR,
    'opus'   : OPUS_COLOR,
}

CODEC_LABELS = {
    'encodec': 'EnCodec',
    'dac'    : 'DAC',
    'mp3'    : 'MP3',
    'opus'   : 'Opus',
}

DATASET_CONFIG = {
    'anuraset': {
        'orig_path'    : RESULTS_BASE / 'on_original'   / 'anuraset' / 'macro_metrics.csv',
        'comp_path'    : RESULTS_BASE / 'on_compressed' / 'anuraset' / 'macro_metrics.csv',
        'roc_col_orig' : 'macro_roc_auc',
        'roc_col_comp' : 'macro_roc_auc',
        'label'        : 'AnuraSet',
    },
    'lemur': {
        'orig_path'    : RESULTS_BASE / 'on_original'   / 'lemur' / 'macro_metrics_temporal.csv',
        'comp_path'    : RESULTS_BASE / 'on_compressed' / 'lemur' / 'macro_metrics_temporal.csv',
        'roc_col_orig' : 'roc_auc',
        'roc_col_comp' : 'roc_auc',
        'label'        : 'Black-and-white ruffed lemur',
    },
    'northeastern': {
        'orig_path'    : RESULTS_BASE / 'on_original'   / 'northeastern' / 'macro_metrics_ovr.csv',
        'comp_path'    : RESULTS_BASE / 'on_compressed' / 'northeastern' / 'macro_metrics_ovr.csv',
        'roc_col_orig' : 'macro_roc_auc',
        'roc_col_comp' : 'macro_roc_auc',
        'label'        : 'Northeastern US soundscapes',
    },
}

def filter_bitrates(df):
    if df.empty or 'codec' not in df.columns:
        return df
    df = df[~((df['codec'] == 'mp3')  & (~df['kbps'].isin(MP3_KEEP)))]
    df = df[~((df['codec'] == 'opus') & (~df['kbps'].isin(OPUS_KEEP)))]
    return df

def normalise_kbps(df):
    if 'kbps' not in df.columns or df['kbps'].isna().all():
        df = df.copy()
        df['kbps'] = df.apply(
            lambda r: DAC_KBPS.get(int(r['bitrate']), float(r['bitrate']))
            if r.get('codec') == 'dac' else float(r['bitrate']), axis=1)
    return df


def load_delta(config):
    if not config['orig_path'].exists() or not config['comp_path'].exists():
        print(f"  WARNING: missing file for {config['label']}")
        return None
    try:
        orig = normalise_kbps(pd.read_csv(config['orig_path']))
        comp = normalise_kbps(pd.read_csv(config['comp_path']))
    except Exception as e:
        print(f"  WARNING: could not load {config['label']}: {e}")
        return None

    roc_orig = config['roc_col_orig']
    roc_comp = config['roc_col_comp']

    orig_dict = dict(zip(orig['condition'], orig[roc_orig]))
    rows = []
    for _, row in comp.iterrows():
        cond  = row['condition']
        codec = row.get('codec', None)
        kbps  = row.get('kbps', None)
        if cond in orig_dict and codec not in ('original', None):
            delta = float(row[roc_comp]) - float(orig_dict[cond])
            rows.append({'condition': cond, 'codec': codec,
                         'kbps': kbps, 'delta': delta})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = filter_bitrates(df)

    filtered = []
    for codec, grp in df.groupby('codec'):
        grp = grp.sort_values('kbps')
        if len(grp) > 2:
            filtered.append(grp.iloc[[0, -1]])
        else:
            filtered.append(grp)

    return pd.concat(filtered, ignore_index=True) if filtered else None


def plot_delta_panel(ax, delta_df, panel_label, show_ylabel):
    if delta_df is None or delta_df.empty:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        return

    codecs = ['encodec', 'dac', 'mp3', 'opus']
    x_positions = []
    x_labels    = []
    bar_colors  = []
    deltas      = []

    x = 0
    for codec in codecs:
        sub = delta_df[delta_df['codec'] == codec].sort_values('kbps')
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            x_positions.append(x)
            x_labels.append(f"{row['kbps']:.4g}")
            bar_colors.append(CODEC_COLORS.get(codec, 'grey'))
            deltas.append(row['delta'])
            x += 1
        x += 0.8  # gap between codec groups

    ax.bar(x_positions, deltas, color=bar_colors,
           width=0.7, edgecolor='white', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=1.0, alpha=0.6)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)

    ax.tick_params(labelsize=13, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)

    if show_ylabel:
        ax.set_ylabel(
            '$\\Delta$ ROC-AUC\n(transfer learning $-$ trained on original)',
            fontsize=15, labelpad=10)
    ax.set_xlabel('Bitrate (kbps)', fontsize=15)
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=16, va='top', ha='left')


def main():
    plt.rcParams['font.family'] = 'Arial'

    active = {k: v for k, v in DATASET_CONFIG.items()
              if v['orig_path'].exists() and v['comp_path'].exists()}

    if not active:
        print("ERROR: no datasets found")
        return

    n_cols = len(active)
    # Give plenty of vertical space at the bottom for the legend
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(7 * n_cols, 7.2),
                             gridspec_kw={'wspace': 0.28})
    if n_cols == 1:
        axes = [axes]

    all_deltas = []
    delta_dfs  = {}
    for key, config in active.items():
        delta_df = load_delta(config)
        delta_dfs[key] = delta_df
        if delta_df is not None:
            all_deltas.extend(delta_df['delta'].tolist())

    if all_deltas:
        y_pad = (max(all_deltas) - min(all_deltas)) * 0.15
        y_min = min(all_deltas) - y_pad
        y_max = max(all_deltas) + y_pad

    for col_idx, (key, config) in enumerate(active.items()):
        ax = axes[col_idx]
        plot_delta_panel(ax, delta_dfs[key],
                         panel_label=PANEL_LABELS[col_idx],
                         show_ylabel=(col_idx == 0))
        ax.set_title(config['label'], fontsize=17, pad=10)
        if all_deltas:
            ax.set_ylim(y_min, y_max)

    legend_elements = [Patch(facecolor=CODEC_COLORS[c], label=CODEC_LABELS[c])
                   for c in ['encodec', 'dac', 'mp3', 'opus']]

    axes[-1].legend(handles=legend_elements,
                    loc='center left',
                    ncol=1,
                    fontsize=12,
                    framealpha=0.95,
                    bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    out = FIGS_DIR / 'transfer_learning_delta.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()