#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_birdnet_macro_metrics.py
BirdNET pre-trained species detection performance.
1 row x 3 columns: ROC-AUC, Precision, Recall
Uses binary evaluation results.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import PATHS

RESULTS_DIR = Path(PATHS['results_dir']) / 'downstream' / 'pretrained' / 'birdnet'
FIGS_DIR    = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

ENCODEC_COLOR = '#1f77b4'
DAC_COLOR     = '#ff7f0e'
MP3_COLOR     = '#2ca02c'
OPUS_COLOR    = '#9467bd'

PANEL_LABELS = ['(a)', '(b)', '(c)']
METRICS = [
    ('macro_roc_auc',        'Macro ROC-AUC'),
    ('precision_optimal_f1', 'Macro Precision'),
    ('recall_optimal_f1',    'Macro Recall'),
]

FONTSIZE   = 20
LINEWIDTH  = 2
MARKERSIZE = 8

DAC_KBPS = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}

def main():
    plt.rcParams['font.family'] = 'Arial'

    path = RESULTS_DIR / 'macro_metrics_ovr.csv'
    if not path.exists():
        print(f"ERROR: not found: {path}")
        return

    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['condition'], keep='last')

    def get_kbps(row):
        cond = row['condition']
        if cond == 'original':
            return None
        for codec in ['encodec', 'dac', 'mp3', 'opus']:
            if cond.startswith(codec):
                br = float(cond.replace(codec, ''))
                if codec == 'dac':
                    return DAC_KBPS.get(int(br), br)
                return br
        return None

    df['kbps'] = df.apply(get_kbps, axis=1)
    df['codec_name'] = df['condition'].apply(lambda c:
        'original' if c == 'original' else
        next((x for x in ['encodec', 'dac', 'mp3', 'opus'] if c.startswith(x)), 'unknown'))

    orig     = df[df['condition'] == 'original']
    baseline = {m: float(orig[m].iloc[0]) for m, _ in METRICS}

    enc  = df[df['codec_name'] == 'encodec'].sort_values('kbps')
    dac  = df[df['codec_name'] == 'dac'].sort_values('kbps')
    mp3  = df[df['codec_name'] == 'mp3'].sort_values('kbps')
    opus = df[df['codec_name'] == 'opus'].sort_values('kbps')

    fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                              gridspec_kw={'wspace': 0.3})

    legend_handles = []
    legend_labels  = []

    for col_idx, (metric_col, metric_label) in enumerate(METRICS):
        ax = axes[col_idx]

        bl_line = ax.axhline(baseline[metric_col], color='black', linestyle='--',
                             linewidth=2.0, alpha=0.7,
                             label=f'Original')
        if col_idx == 0:
            legend_handles.append(bl_line)
            legend_labels.append(f'Original')

        for data, color, marker, label in [
            (enc,  ENCODEC_COLOR, 'o', 'EnCodec'),
            (dac,  DAC_COLOR,     's', 'DAC'),
            (mp3,  MP3_COLOR,     '^', 'MP3'),
            (opus, OPUS_COLOR,    'D', 'Opus'),
        ]:
            if not data.empty:
                line, = ax.plot(data['kbps'], data[metric_col],
                                color=color, marker=marker,
                                linewidth=LINEWIDTH, markersize=MARKERSIZE,
                                linestyle='-', label=label)
                if col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)

        ax.set_xlabel('Bitrate (kbps)', fontsize=FONTSIZE)
        ax.set_ylabel(metric_label, fontsize=FONTSIZE)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONTSIZE, direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(y=0.15)

        ax.text(0.02, 0.98, PANEL_LABELS[col_idx],
                transform=ax.transAxes,
                fontsize=FONTSIZE, va='top', ha='left')

        # Legend in middle panel
        if col_idx == 1:
            ax.legend(handles=legend_handles, labels=legend_labels,
                      fontsize=FONTSIZE, frameon=False, loc='best')

    plt.tight_layout()
    out = FIGS_DIR / 'birdnet_pretrained_metrics.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()