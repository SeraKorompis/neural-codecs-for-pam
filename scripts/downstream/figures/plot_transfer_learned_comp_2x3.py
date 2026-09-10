#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_roc_auc_transfer_learned_compressed_agg.py
ROC-AUC vs bitrate comparing two training strategies across datasets.
2x3 grid: Row 1 = neural codecs, Row 2 = conventional codecs.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
PANEL_LABELS  = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]

FONTSIZE    = 22
TICK_SIZE   = 20
LEGEND_SIZE = 16
LINEWIDTH   = 2.5
MARKERSIZE  = 8

CODEC_GROUPS = {
    'neural':      ['encodec', 'dac'],
    'conventional': ['mp3', 'opus'],
}

CODEC_STYLE = {
    'encodec': (ENCODEC_COLOR, 'o'),
    'dac'    : (DAC_COLOR,     's'),
    'mp3'    : (MP3_COLOR,     '^'),
    'opus'   : (OPUS_COLOR,    'D'),
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


def load_metrics(path, roc_col):
    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None, {}
    df       = normalise_kbps(pd.read_csv(path))
    orig_rows = df[df['condition'] == 'original']
    baseline  = float(orig_rows[roc_col].iloc[0]) if not orig_rows.empty else None
    codec_dfs = {}
    if 'codec' in df.columns:
        for codec in ['encodec', 'dac', 'mp3', 'opus']:
            sub = filter_bitrates(df[df['codec'] == codec].sort_values('kbps'))
            if not sub.empty:
                codec_dfs[codec] = sub
    return baseline, codec_dfs


def plot_panel(ax, orig_path, comp_path, roc_col_orig, roc_col_comp,
               codec_group, panel_label, show_ylabel, show_xlabel, row_label=None):
    baseline, codec_dfs_orig = load_metrics(orig_path, roc_col_orig)
    _,         codec_dfs_comp = load_metrics(comp_path, roc_col_comp)

    if baseline is not None:
        ax.axhline(baseline, color='black', linestyle='--',
                   linewidth=2.0, alpha=0.7)

    for codec in codec_group:
        color, marker = CODEC_STYLE[codec]
        for codec_dfs, roc_col, linestyle in [
            (codec_dfs_orig, roc_col_orig, '-'),
            (codec_dfs_comp, roc_col_comp, '--'),
        ]:
            if codec in codec_dfs and roc_col in codec_dfs[codec].columns:
                data = codec_dfs[codec]
                ax.plot(data['kbps'], data[roc_col],
                        color=color, marker=marker,
                        linewidth=LINEWIDTH, markersize=MARKERSIZE,
                        linestyle=linestyle)

    ax.set_xlim(left=0)
    ax.set_ylim(0.60, 1.00)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ylabel:
        ax.set_ylabel('ROC-AUC', fontsize=FONTSIZE, labelpad=10)
    if show_xlabel:
        ax.set_xlabel('Bitrate (kbps)', fontsize=FONTSIZE, labelpad=10)

    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=FONTSIZE, va='top', ha='left')


def main():
    plt.rcParams['font.family'] = 'Arial'

    datasets = list(DATASET_CONFIG.keys())
    n_cols   = len(datasets)
    n_rows   = 2  # neural / conventional

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8 * n_cols, 6 * n_rows),
        gridspec_kw={'hspace': 0.30, 'wspace': 0.22})

    row_labels    = ['Neural codecs', 'Conventional codecs']
    codec_groups  = [CODEC_GROUPS['neural'], CODEC_GROUPS['conventional']]
    panel_idx     = 0

    for row_idx, (codec_group, row_label) in enumerate(zip(codec_groups, row_labels)):
        for col_idx, key in enumerate(datasets):
            config = DATASET_CONFIG[key]
            ax     = axes[row_idx, col_idx]

            plot_panel(
                ax,
                config['orig_path'], config['comp_path'],
                config['roc_col_orig'], config['roc_col_comp'],
                codec_group=codec_group,
                panel_label=PANEL_LABELS[panel_idx],
                show_ylabel=(col_idx == 0),
                show_xlabel=(row_idx == n_rows - 1))

            if row_idx == 0:
                ax.set_title(config['label'], fontsize=FONTSIZE, pad=12)

            # Row label on leftmost panel
            if col_idx == 0:
                ax.set_ylabel(f'ROC-AUC\n({row_label})',
                              fontsize=FONTSIZE, labelpad=10)

            panel_idx += 1

    # Legend
    codec_handles = [
        Patch(color=CODEC_STYLE[c][0], label=CODEC_LABELS[c])
        for c in ['encodec', 'dac', 'mp3', 'opus']
    ]
    style_handles = [
        Line2D([0], [0], color='gray', linewidth=LINEWIDTH,
               linestyle='-',  label='Trained on original'),
        Line2D([0], [0], color='gray', linewidth=LINEWIDTH,
               linestyle='--', label='Domain-matched'),
        Line2D([0], [0], color='black', linewidth=2.0,
               linestyle='--', alpha=0.7, label='Original (uncompressed)'),
    ]

    leg1 = fig.legend(
        handles=codec_handles,
        loc='lower center', ncol=4,
        fontsize=LEGEND_SIZE, frameon=False,
        bbox_to_anchor=(0.5, 0.06))

    leg2 = fig.legend(
        handles=style_handles,
        loc='lower center', ncol=3,
        fontsize=LEGEND_SIZE, frameon=False,
        bbox_to_anchor=(0.5, 0.01))

    fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    out = FIGS_DIR / 'roc_auc_transfer_learned_compressed.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()