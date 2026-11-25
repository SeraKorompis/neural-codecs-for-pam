#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_confidence_violins.py
Per-species confidence change for the annotated positive test windows of the
Northeastern Birds dataset, with a shared species axis:
  (a) pre-trained BirdNET:                   compressed - original
  (b) transfer-learned, trained on raw audio: compressed - original
  (c) transfer-learned, trained on compressed audio (tested on compressed):
      minus the raw-trained classifier on original audio
A window with no BirdNET detection counts as confidence 0.

Usage:
    python scripts/downstream/figures/plot_confidence_violins.py
    python scripts/downstream/figures/plot_confidence_violins.py --codec encodec --kbps 6.0
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import PATHS
from plot_auc_scatter_all import load_and_pivot_data, BIRDNET_RES_DIR

RES  = Path(PATHS['results_dir']) / 'downstream'
FIGS_DIR = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

RAW_BIRDNET = RES / 'pretrained' / 'birdnet' / 'raw'
EMB         = RES / 'transfer_learned' / 'embeddings' / 'northeastern'
TL_ORIG     = RES / 'transfer_learned' / 'on_original'   / 'northeastern' / 'raw'
TL_COMP     = RES / 'transfer_learned' / 'on_compressed' / 'northeastern' / 'raw'

VIOLIN_COLOR = '#0072B2'     # Okabe-Ito blue, as for EnCodec elsewhere
MIN_ORIG_CONF = 0.05
FS, FS_TICK = 28, 24
FS_SPECIES = 26
XLIM = (-1.05, 1.05)


def evaluated_species(fallback):
    """The species used in the AUC analysis (at least 10 test positives)."""
    path = next(BIRDNET_RES_DIR / n for n in ['per_species_ovr.csv', 'per_species.csv']
                if (BIRDNET_RES_DIR / n).exists())
    sp = load_and_pivot_data(path, 'birdnet')
    if 'species' in sp.columns:
        return list(sp['species'])
    if sp.index.dtype == object:
        return list(sp.index)
    print('WARNING: species names not found in the per-species table; using all species')
    return fallback


def window_conf(df, meta):
    df = df.merge(meta[['filename', 'window_start', 'window_idx']],
                  left_on=['filename', 'start_sec'],
                  right_on=['filename', 'window_start'], how='inner')
    return df.groupby(['common_name', 'window_idx'], as_index=False)['confidence'].max()


def build_table(codec, kbps):
    tag = f'{codec}{kbps}'
    meta = pd.read_csv(EMB / 'northeastern_all_metadata.csv')
    meta['window_idx'] = range(len(meta))
    pos_idx = json.load(open(EMB / 'northeastern_positive_test_indices.json'))
    species_cols = json.load(open(EMB / 'northeastern_species_cols.json'))
    species = [s for s in evaluated_species(species_cols) if s in species_cols]
    pos = pd.DataFrame([(sp, i) for sp in species for i in pos_idx.get(sp, [])],
                       columns=['species', 'window_idx'])

    # (a) pre-trained BirdNET detections
    orig = window_conf(pd.read_csv(RAW_BIRDNET / 'detections_original_lowthresh.csv',
                       usecols=['filename', 'start_sec', 'common_name', 'confidence']), meta)
    parts = []
    for ch in pd.read_csv(RAW_BIRDNET / 'detections_reconstructed_lowthresh.csv',
                          usecols=['filename', 'codec', 'bitrate', 'start_sec',
                                   'common_name', 'confidence'], chunksize=2_000_000):
        parts.append(ch[(ch.codec.astype(str) == codec) & (ch.bitrate.astype(float) == kbps)])
    enc = window_conf(pd.concat(parts), meta)
    d = (pos.merge(orig.rename(columns={'common_name': 'species', 'confidence': 'a_orig'}),
                   on=['species', 'window_idx'], how='left')
            .merge(enc.rename(columns={'common_name': 'species', 'confidence': 'a_comp'}),
                   on=['species', 'window_idx'], how='left')
            .fillna({'a_orig': 0.0, 'a_comp': 0.0}))
    d['shift_a'] = d['a_comp'] - d['a_orig']

    # (b) and (c) transfer-learned scores, long format restricted to the positives
    s_oo = pd.read_csv(TL_ORIG / 'scores_original_on_original.csv')
    s_oc = pd.read_csv(TL_ORIG / f'scores_original_on_{tag}.csv')
    s_cc = pd.read_csv(TL_COMP / f'scores_{tag}_on_{tag}.csv')

    def long(scores, name):
        x = scores[['clip_idx'] + species].melt(id_vars='clip_idx', var_name='species',
                                                 value_name=name)
        return x.rename(columns={'clip_idx': 'window_idx'})

    for name, s in [('b_orig', s_oo), ('b_comp', s_oc), ('c_comp', s_cc)]:
        d = d.merge(long(s, name), on=['species', 'window_idx'], how='left')
    d['shift_b'] = d['b_comp'] - d['b_orig']
    d['shift_c'] = d['c_comp'] - d['b_orig']
    return d


def draw_violins(ax, d, order, col, title, show_ylabels, n_by_species):
    for pos, sp in enumerate(order):
        v = d.loc[d.species == sp, col].dropna().values
        if len(v) >= 2 and np.std(v) > 1e-9:
            parts = ax.violinplot([v], positions=[pos], orientation='horizontal',
                                  showmedians=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(VIOLIN_COLOR)
                pc.set_alpha(0.55)
                pc.set_edgecolor('white')
                pc.set_linewidth(0.4)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.4)
            parts['cmedians'].set_zorder(5)
        elif len(v):                      # constant values: no density to draw
            ax.plot([np.median(v)] * 2, [pos - 0.3, pos + 0.3], color='black', lw=1.4, zorder=5)
    ax.axvline(0, color='black', lw=0.9, alpha=0.4)
    ax.axvspan(XLIM[0], 0, alpha=0.03, color='red')
    ax.axvspan(0, XLIM[1], alpha=0.03, color='green')
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlabel('Confidence change', fontsize=FS)
    ax.set_title(title, fontsize=FS, loc='left', pad=10)
    ax.set_yticks(range(len(order)))
    if show_ylabels:
        ax.set_yticklabels([f'{sp}  (n={n_by_species[sp]})' for sp in order], fontsize=FS_SPECIES)
    else:
        ax.tick_params(axis='y', labelleft=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codec', default='encodec')
    ap.add_argument('--kbps', type=float, default=6.0)
    args = ap.parse_args()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    d = build_table(args.codec, args.kbps)
    out_csv = RES / f'confidence_shift_per_window_{args.codec}{args.kbps}.csv'
    d.to_csv(out_csv, index=False)

    n = d.groupby('species').size()
    for col in ['shift_a', 'shift_b', 'shift_c']:
        miss = d[col].isna().sum()
        if miss:
            print(f'WARNING: {miss} windows have no value for {col}')
    order = list(d.groupby('species')['shift_a'].median().sort_values().index)
    print(f'{args.codec} {args.kbps} kbps: {len(d):,} positive windows, {len(order)} species')

    for t in [0.0, MIN_ORIG_CONF]:
        line = []
        for col, oc in [('shift_a', 'a_orig'), ('shift_b', 'b_orig'), ('shift_c', 'b_orig')]:
            s = d[d[oc] >= t].groupby('species')[col].agg(['count', 'median'])
            s = s[s['count'] >= 10]
            line.append(f'{col[-1]}: {(s["median"] < 0).sum()}/{len(s)}')
        print(f'negative median shift (original confidence >= {t}): ' + ', '.join(line))

    height = max(8.0, 0.42 * len(order) + 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(28, height), sharey=True,
                             gridspec_kw={'wspace': 0.08})
    titles = ['(a) Pre-trained\n     BirdNET',
              '(b) Transfer-learned classifier\n     (raw audio)',
              '(c) Transfer-learned classifier\n     (compressed audio)']
    for ax, col, title, first in zip(axes, ['shift_a', 'shift_b', 'shift_c'], titles,
                                     [True, False, False]):
        draw_violins(ax, d, order, col, title, first, n)

    out = FIGS_DIR / f'confidence_violins_{args.codec}{args.kbps}'
    plt.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}.png and {out}.pdf; per-window table: {out_csv}')


if __name__ == '__main__':
    main()
