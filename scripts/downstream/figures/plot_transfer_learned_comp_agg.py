#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_roc_auc_transfer_learned_compressed_agg.py
ROC-AUC vs bitrate comparing two training strategies across datasets.
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
PANEL_LABELS  = ['(a)', '(b)', '(c)', '(d)']

MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]

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
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), roc_col
    df        = normalise_kbps(pd.read_csv(path))
    orig_rows = df[df['condition'] == 'original']
    baseline  = float(orig_rows[roc_col].iloc[0]) if not orig_rows.empty else None
    enc  = filter_bitrates(df[df['codec'] == 'encodec'].sort_values('kbps')) if 'codec' in df.columns else pd.DataFrame()
    dac  = filter_bitrates(df[df['codec'] == 'dac'].sort_values('kbps'))     if 'codec' in df.columns else pd.DataFrame()
    mp3  = filter_bitrates(df[df['codec'] == 'mp3'].sort_values('kbps'))     if 'codec' in df.columns else pd.DataFrame()
    opus = filter_bitrates(df[df['codec'] == 'opus'].sort_values('kbps'))    if 'codec' in df.columns else pd.DataFrame()
    return baseline, enc, dac, mp3, opus, roc_col


def plot_dataset_panel(ax, orig_path, comp_path, roc_col_orig, roc_col_comp,
                       panel_label, show_ylabel):
    baseline_orig, enc_orig, dac_orig, mp3_orig, opus_orig, roc_orig = \
        load_metrics(orig_path, roc_col_orig)
    _, enc_comp, dac_comp, mp3_comp, opus_comp, roc_comp = \
        load_metrics(comp_path, roc_col_comp)

    if baseline_orig is not None:
        ax.axhline(baseline_orig, color='black', linestyle='--',
                   linewidth=2.0, alpha=0.7)

    for enc, dac, mp3, opus, roc_col, linestyle in [
        (enc_orig, dac_orig, mp3_orig, opus_orig, roc_orig, '-'),
        (enc_comp, dac_comp, mp3_comp, opus_comp, roc_comp, '--'),
    ]:
        for data, color, marker in [
            (enc,  ENCODEC_COLOR, 'o'),
            (dac,  DAC_COLOR,     's'),
            (mp3,  MP3_COLOR,     '^'),
            (opus, OPUS_COLOR,    'D'),
        ]:
            if not data.empty and roc_col in data.columns:
                ax.plot(data['kbps'], data[roc_col],
                        color=color, marker=marker,
                        linewidth=LINEWIDTH, markersize=MARKERSIZE,
                        linestyle=linestyle)

    if show_ylabel:
        ax.set_ylabel('ROC-AUC', fontsize=FONTSIZE)
    ax.set_xlabel('Bitrate (kbps)', fontsize=FONTSIZE)
    ax.set_xlim(left=0)
    ax.set_ylim(0.65, 1.00)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=FONTSIZE, va='top', ha='left')


def main():
    plt.rcParams['font.family'] = 'Arial'

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

    active = {}
    for key, config in DATASET_CONFIG.items():
        if config['orig_path'].exists() or config['comp_path'].exists():
            active[key] = config
            print(f"Loaded: {key}")
        else:
            print(f"WARNING: skipping {key} — no files found")

    if not active:
        print("ERROR: no datasets found")
        return

    n_cols = len(active)

    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7),
                             gridspec_kw={'wspace': 0.22})
    if n_cols == 1:
        axes = [axes]

    for col_idx, (key, config) in enumerate(active.items()):
        ax = axes[col_idx]
        plot_dataset_panel(
            ax,
            config['orig_path'], config['comp_path'],
            config['roc_col_orig'], config['roc_col_comp'],
            panel_label=PANEL_LABELS[col_idx],
            show_ylabel=(col_idx == 0))
        ax.set_title(config['label'], fontsize=FONTSIZE, pad=10)

    # Codec handles (without title)
    codec_handles = [
        Patch(color=ENCODEC_COLOR, label='EnCodec'),
        Patch(color=DAC_COLOR,     label='DAC'),
        Patch(color=MP3_COLOR,     label='MP3'),
        Patch(color=OPUS_COLOR,    label='Opus'),
    ]

    # Style handles (without title)
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
        bbox_to_anchor=(0.5, 0.08))

    leg2 = fig.legend(
        handles=style_handles,
        loc='lower center', ncol=3,
        fontsize=LEGEND_SIZE, frameon=False,
        bbox_to_anchor=(0.5, 0.01))

    fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0.28, 1, 1])
    out = FIGS_DIR / 'roc_auc_transfer_learned_compressed.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()