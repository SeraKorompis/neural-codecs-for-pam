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
--font_scale X:            multiply most font sizes by X (default 1.3)
                           key boxes use KEY_SCALE, bottom legend LEGEND_FONTSIZE
"""
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import PATHS
from plot_auc_scatter_all import (load_and_pivot_data, make_scatter,
                                  BIRDNET_RES_DIR, N_LABEL)

# Colour-blind-safe colours for the scatter panels: blue = least degraded,
# vermillion = most degraded. recolour_scatter() swaps the colours that
# make_scatter hard-codes (green, red, pink) for these after the panels are drawn.
GOOD_COLOR, GOOD_TEXT = '#0072B2', '#00436b'
BAD_COLOR,  BAD_TEXT  = '#D55E00', '#8a3b00'
SHADE_COLOR = '#ffcccc'
COLOUR_SWAP = {
    '#2ca02c': GOOD_COLOR,   # least degraded dots and numeral box edges
    '#1b691b': GOOD_TEXT,    # least degraded numerals
    '#d62728': BAD_COLOR,    # most degraded dots and numeral box edges
    '#a81c1c': BAD_TEXT,     # most degraded numerals
    '#ffcccc': SHADE_COLOR,  # degradation region
}

FIGS_DIR = Path(PATHS['figures_dir']) / 'downstream' / 'birdnet'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind-safe palette: blue, orange, bluish green, reddish purple
COLORS  = {'encodec': '#0072B2', 'dac': '#E69F00',
           'mp3': '#009E73', 'opus': '#CC79A7'}
MARKERS = {'encodec': 'o', 'dac': 's', 'mp3': '^', 'opus': 'D'}
NAMES   = {'encodec': 'EnCodec', 'dac': 'DAC', 'mp3': 'MP3', 'opus': 'Opus'}
DAC_KBPS = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}

FONTSIZE   = 15
LINEWIDTH  = 2
MARKERSIZE = 8
AUC_MIN, AUC_MAX = 0.50, 1.01
FONT_SCALE = 1.3        # all text in the figure
KEY_SCALE = 1.3        # 'Least/Most degraded' key boxes (relative to original size)
LEGEND_FONTSIZE = 22    # bottom legend (absolute size, after scaling)

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


def recolour_scatter(axes):
    """Replace the colours hard-coded in make_scatter with the ones above."""
    def swapped(colour):
        return COLOUR_SWAP.get(to_hex(colour))

    for ax in axes:
        for coll in ax.collections:          # dots and the shaded region
            for get, put in ((coll.get_facecolor, coll.set_facecolor),
                             (coll.get_edgecolor, coll.set_edgecolor)):
                colours = get()
                if len(colours) and swapped(colours[0]):
                    put(swapped(colours[0]))
        for text in ax.texts:                # numerals and their boxes
            if swapped(text.get_color()):
                text.set_color(swapped(text.get_color()))
            box = text.get_bbox_patch()
            if box is not None and swapped(box.get_edgecolor()):
                box.set_edgecolor(swapped(box.get_edgecolor()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', default='comparable',
                        choices=list(SCATTER_SETS))
    parser.add_argument('--font_scale', type=float, default=FONT_SCALE)
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

    fig = plt.figure(figsize=(22, 9.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    plot_macro_panel(ax_a, macro_path)
    ax_a.set_box_aspect(1)   # same square box as (b) and (c), so the axes align

    for ax, tag, show_y, (label, codec, kbps) in zip(
            [ax_b, ax_c], ['(b)', '(c)'], [True, False],
            SCATTER_SETS[args.set]):
        make_scatter(ax, summary,
                     f'auc_{codec}_{kbps}', f'drop_{codec}_{kbps}',
                     label, tag, AUC_MIN, AUC_MAX,
                     macro_path, codec, kbps, show_ylabel=show_y)
    recolour_scatter([ax_b, ax_c])

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GOOD_COLOR,
               markersize=9, markeredgecolor='black', markeredgewidth=0.7,
               label=f'Top {N_LABEL} least degraded (upper quartile, i-v)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BAD_COLOR,
               markersize=9, markeredgecolor='black', markeredgewidth=0.7,
               label=f'Top {N_LABEL} most degraded (upper quartile, vi-x)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
               markersize=8, markeredgewidth=0, label='Other species'),
        mpatches.Patch(facecolor=SHADE_COLOR, alpha=0.6,
                       label='Degradation region (y < x)'),
        Line2D([0], [0], color='#444444', linestyle='--', linewidth=1.1,
               label='No change (y = x)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=FONTSIZE - 2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.22,
                        wspace=0.18)

    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().startswith('Least degraded'):
            t.set_fontsize(t.get_fontsize() * KEY_SCALE)
        else:
            t.set_fontsize(t.get_fontsize() * args.font_scale)
    for t in fig.legends[0].get_texts():
        t.set_fontsize(LEGEND_FONTSIZE)

    out = FIGS_DIR / f'pretrained_three_panel_{args.set}'
    plt.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}.png and {out}.pdf")


if __name__ == '__main__':
    main()