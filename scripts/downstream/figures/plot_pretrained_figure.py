#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_pretrained_figure.py
Three-panel figure for the pre-trained BirdNET evaluation:
  (a) macro ROC-AUC against bitrate for every codec
  (b) per-species ROC-AUC, original vs EnCodec
  (c) per-species ROC-AUC, original vs DAC
The scatter panels reuse make_scatter and load_and_pivot_data from
plot_auc_scatter_all.py.

Usage:
    python scripts/downstream/figures/plot_pretrained_figure.py
    python scripts/downstream/figures/plot_pretrained_figure.py --set best

--set comparable (default): EnCodec 6.0 kbps and DAC 5.33 kbps
--set best:                 EnCodec 24.0 kbps and DAC 8.0 kbps
"""
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import PATHS
from plot_auc_scatter_all import (load_and_pivot_data, make_scatter,
                                  BIRDNET_RES_DIR, N_LABEL)

FIGS_DIR = Path(PATHS['figures_dir']) / 'downstream' / 'birdnet'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

COLORS  = {'encodec': '#1f77b4', 'dac': '#ff7f0e',
           'mp3': '#2ca02c', 'opus': '#9467bd'}
MARKERS = {'encodec': 'o', 'dac': 's', 'mp3': '^', 'opus': 'D'}
NAMES   = {'encodec': 'EnCodec', 'dac': 'DAC', 'mp3': 'MP3', 'opus': 'Opus'}
DAC_KBPS = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}

FONTSIZE   = 15
LINEWIDTH  = 2
MARKERSIZE = 8
AUC_MIN, AUC_MAX = 0.50, 1.01

# (label used by make_scatter, codec, kbps) for panels (b) and (c)
SCATTER_SETS = {
    'comparable': [('EnCodec 6.0 kbps', 'encodec', 6.0),
                   ('DAC 5.33 kbps',    'dac',     5.33)],
    'best':       [('EnCodec 24.0 kbps', 'encodec', 24.0),
                   ('DAC 8.0 kbps',      'dac',     8.0)],
}


def first_existing(directory, names):
    for name in names:
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"None of {names} found in {directory}")


def split_condition(cond):
    """Return (codec, kbps) from a condition tag such as 'mp38' or 'dac3'."""
    if cond == 'original':
        return 'original', None
    for codec in COLORS:
        if cond.startswith(codec):
            br = float(cond[len(codec):])
            kbps = DAC_KBPS.get(int(br), br) if codec == 'dac' else br
            return codec, kbps
    return 'unknown', None


def plot_macro_panel(ax, macro_path):
    df = pd.read_csv(macro_path)
    df = df.drop_duplicates(subset=['condition'], keep='last')
    parsed = df['condition'].apply(split_condition)
    df['codec_name'] = [p[0] for p in parsed]
    df['kbps_plot']  = [p[1] for p in parsed]

    baseline = float(df.loc[df['condition'] == 'original',
                            'macro_roc_auc'].iloc[0])
    ax.axhline(baseline, color='black', linestyle='--', linewidth=2.0,
               alpha=0.7, label='Original')
    for codec in COLORS:
        sub = df[df['codec_name'] == codec].sort_values('kbps_plot')
        if sub.empty:
            continue
        ax.plot(sub['kbps_plot'], sub['macro_roc_auc'],
                color=COLORS[codec], marker=MARKERS[codec],
                linewidth=LINEWIDTH, markersize=MARKERSIZE,
                linestyle='-', label=NAMES[codec])

    ax.set_title('(a) Macro ROC-AUC', fontsize=FONTSIZE, pad=8, loc='left')
    ax.set_xlabel('Bitrate (kbps)', fontsize=FONTSIZE)
    ax.set_ylabel('Macro ROC-AUC', fontsize=FONTSIZE)
    ax.set_xlim(left=0)
    ax.margins(y=0.15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=13, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=13, frameon=False, loc='best')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', default='comparable',
                        choices=list(SCATTER_SETS))
    args = parser.parse_args()

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    macro_path = first_existing(BIRDNET_RES_DIR,
                                ['macro_metrics_ovr.csv', 'macro_metrics.csv'])
    species_path = first_existing(BIRDNET_RES_DIR,
                                  ['per_species_ovr.csv', 'per_species.csv'])
    print(f"Macro metrics: {macro_path}")
    print(f"Per species:   {species_path}")

    summary = load_and_pivot_data(species_path, 'birdnet')

    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    plot_macro_panel(ax_a, macro_path)

    for ax, tag, show_y, (label, codec, kbps) in zip(
            [ax_b, ax_c], ['(b)', '(c)'], [True, False],
            SCATTER_SETS[args.set]):
        make_scatter(ax, summary,
                     f'auc_{codec}_{kbps}', f'drop_{codec}_{kbps}',
                     label, tag, AUC_MIN, AUC_MAX,
                     macro_path, codec, kbps, show_ylabel=show_y)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=9, markeredgecolor='black', markeredgewidth=0.7,
               label=f'Top {N_LABEL} least degraded (upper quartile, i-v)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=9, markeredgecolor='black', markeredgewidth=0.7,
               label=f'Top {N_LABEL} most degraded (upper quartile, vi-x)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
               markersize=8, markeredgewidth=0, label='Other species'),
        mpatches.Patch(facecolor='#ffcccc', alpha=0.6,
                       label='Degradation region (y < x)'),
        Line2D([0], [0], color='#444444', linestyle='--', linewidth=1.1,
               label='No change (y = x)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=FONTSIZE - 2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.17,
                        wspace=0.18)

    out = FIGS_DIR / f'pretrained_three_panel_{args.set}'
    plt.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}.png and {out}.pdf")


if __name__ == '__main__':
    main()