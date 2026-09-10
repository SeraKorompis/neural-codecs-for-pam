#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_per_species_tl_on_comp.py
Combined figure:
  (a) Left:  Box plot of per-species Δ ROC-AUC across all EnCodec bitrates
             (domain-matched minus trained-on-original), sorted by median
  (b) Right top:    Scatter EnCodec 1.5 kbps
  (c) Right bottom: Scatter EnCodec 24.0 kbps
Usage:
    python scripts/downstream/figures/plot_per_species_tl_on_comp.py \
        --dataset anuraset
    python scripts/downstream/figures/plot_per_species_tl_on_comp.py \
        --dataset northeastern
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS_BASE = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned'
FIGS_DIR     = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

MIN_POS_TEST  = 10
ENCODEC_CONDS = ['encodec1.5', 'encodec3.0', 'encodec6.0',
                 'encodec12.0', 'encodec24.0']
SCATTER_CONDS = {
    'low' : ('encodec1.5',  'EnCodec 1.5 kbps'),
    'high': ('encodec24.0', 'EnCodec 24.0 kbps'),
}

FS             = 22
FS_SMALL       = 18
GREY_COLOR     = '#999999'
BOX_COLOR      = '#AEC6E8'

DATASET_CONFIG = {
    'anuraset': {
        'orig_path': RESULTS_BASE / 'on_original'   / 'anuraset' / 'per_species.csv',
        'comp_path': RESULTS_BASE / 'on_compressed' / 'anuraset' / 'per_species.csv',
        'label'    : 'AnuraSet',
    },
    'northeastern': {
        'orig_path': RESULTS_BASE / 'on_original'   / 'northeastern' / 'per_species_ovr.csv',
        'comp_path': RESULTS_BASE / 'on_compressed' / 'northeastern' / 'per_species_ovr.csv',
        'label'    : 'Northeastern US Soundscapes',
    },
}

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_data(config):
    orig = pd.read_csv(config['orig_path'])
    comp = pd.read_csv(config['comp_path'])

    orig_enc = orig[orig['condition'].isin(ENCODEC_CONDS)].copy()

    valid_species = (orig_enc.groupby('species')['n_pos_test']
                     .min()
                     .pipe(lambda s: s[s >= MIN_POS_TEST])
                     .index.tolist())
    print(f"  Species with n_pos_test >= {MIN_POS_TEST}: {len(valid_species)}")

    orig_enc = orig_enc[orig_enc['species'].isin(valid_species)]
    comp_enc = comp[comp['condition'].isin(ENCODEC_CONDS) &
                    comp['species'].isin(valid_species)].copy()

    orig_enc = orig_enc[['species', 'condition', 'roc_auc']].rename(
        columns={'roc_auc': 'roc_orig'})
    comp_enc = comp_enc[['species', 'condition', 'roc_auc']].rename(
        columns={'roc_auc': 'roc_comp'})

    delta_df = orig_enc.merge(comp_enc, on=['species', 'condition'], how='inner')
    delta_df['delta'] = delta_df['roc_comp'] - delta_df['roc_orig']

    scatter_data = {}
    for key, (cond, label) in SCATTER_CONDS.items():
        o = orig[orig['condition'] == cond][
            ['species', 'roc_auc', 'n_pos_test']].rename(
            columns={'roc_auc': 'roc_orig'})
        c = comp[comp['condition'] == cond][
            ['species', 'roc_auc']].rename(
            columns={'roc_auc': 'roc_comp'})
        merged = o.merge(c, on='species', how='inner')
        merged = merged[merged['n_pos_test'] >= MIN_POS_TEST]
        merged['delta'] = merged['roc_comp'] - merged['roc_orig']
        scatter_data[key] = merged
        print(f"  Scatter {label}: {len(merged)} species")

    return delta_df, scatter_data, valid_species


# ---------------------------------------------------------------------------
# BOX PLOT PANEL
# ---------------------------------------------------------------------------
def plot_boxplot(ax, delta_df):
    order = (delta_df.groupby('species')['delta']
             .median()
             .sort_values()
             .index.tolist())

    positions = list(range(len(order)))
    data      = [delta_df[delta_df['species'] == sp]['delta'].values
                 for sp in order]

    ax.boxplot(data, positions=positions, vert=False,
               patch_artist=True, widths=0.6,
               showfliers=True,
               medianprops=dict(color='black', linewidth=2.0),
               boxprops=dict(facecolor=BOX_COLOR, linewidth=1.2),
               whiskerprops=dict(linewidth=1.2),
               capprops=dict(linewidth=1.2),
               flierprops=dict(marker='o', markersize=3,
                               markerfacecolor=GREY_COLOR,
                               markeredgecolor=GREY_COLOR, alpha=0.5))

    all_deltas = delta_df['delta'].values
    x_pad = (all_deltas.max() - all_deltas.min()) * 0.05
    ax.set_xlim(all_deltas.min() - x_pad, all_deltas.max() + x_pad)

    ax.axvline(0, color='black', linestyle='--',
               linewidth=1.2, alpha=0.6, zorder=0)
    ax.axvspan(0, ax.get_xlim()[1], alpha=0.04, color='green')
    ax.axvspan(ax.get_xlim()[0], 0,  alpha=0.04, color='red')

    ax.set_yticks(positions)
    ax.set_yticklabels(order, fontsize=FS_SMALL)
    ax.set_xlabel('Per-species $\\Delta$ ROC-AUC\n (trained on compressed - on original)',
                  fontsize=FS)
    ax.tick_params(axis='x', labelsize=FS_SMALL)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.0, 1.02,
            '(a) Per-species $\\Delta$ ROC-AUC from training on compressed',
            transform=ax.transAxes,
            fontsize=FS, va='bottom', ha='left')


# ---------------------------------------------------------------------------
# SCATTER PANEL
# ---------------------------------------------------------------------------
def plot_scatter(ax, scatter_df, title, panel_tag, show_xlabel):
    auc_min, auc_max = 0.50, 1.01

    ax.fill_between([auc_min, auc_max], [auc_min, auc_max],
                    [auc_min, auc_min],
                    color='#ffcccc', alpha=0.35, zorder=1)

    ax.plot([auc_min, auc_max], [auc_min, auc_max],
            color='#444444', linestyle='--',
            linewidth=1.1, alpha=0.8, zorder=2)

    ax.scatter(scatter_df['roc_orig'], scatter_df['roc_comp'],
               s=60, color=GREY_COLOR, alpha=0.65,
               linewidths=0.5, edgecolors='black', zorder=3)

    ax.set_xlim(auc_min, auc_max)
    ax.set_ylim(auc_min, auc_max)
    ax.set_aspect('equal')
    ax.set_title(f'{panel_tag} {title}', fontsize=FS, pad=8, loc='left')
    ax.set_ylabel('ROC-AUC (domain-matched)', fontsize=FS)
    if show_xlabel:
        ax.set_xlabel('ROC-AUC (trained on original)', fontsize=FS)
    ax.tick_params(labelsize=FS, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=':', zorder=0)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['anuraset', 'northeastern'])
    args   = parser.parse_args()
    config = DATASET_CONFIG[args.dataset]

    plt.rcParams['font.family'] = 'Arial'

    print(f"Loading {args.dataset}...")
    delta_df, scatter_data, valid_species = load_data(config)

    n_sp = len(valid_species)
    # fig  = plt.figure(figsize=(24, max(10, n_sp * 0.35)))

    # outer_gs = gridspec.GridSpec(1, 2,
    #                          width_ratios=[1.2, 1.4],
    #                          wspace=0.15,
    #                          right=0.80)

    # New, fix scatter width regardless of n_sp:
    fig_height = max(10, n_sp * 0.35)
    fig_width  = max(24, n_sp * 0.3)
    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_gs = gridspec.GridSpec(1, 2,
                                width_ratios=[1.2, 1.4],
                                wspace=0.15,
                                right=0.88)

    # ---- (a) BOX PLOT ----
    ax_box = fig.add_subplot(outer_gs[0, 0])
    plot_boxplot(ax_box, delta_df)

    # ---- (b) & (c) SCATTERS ----
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1],
        hspace=0.35)

    for row_idx, (key, (cond, label)) in enumerate(SCATTER_CONDS.items()):
        ax          = fig.add_subplot(right_gs[row_idx, 0])
        panel_tag   = '(b)' if row_idx == 0 else '(c)'
        show_xlabel = (row_idx == 1)
        plot_scatter(ax, scatter_data[key], label, panel_tag, show_xlabel)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#ffcccc', alpha=0.6,
                       label='Domain-matched hurt (y < x)'),
        Line2D([0], [0], color='#444444', linestyle='--',
               linewidth=1.1, label='No change (y = x)'),
    ]
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=2,
               fontsize=FS_SMALL, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIGS_DIR / f'domain_adaptation_combined_{args.dataset}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()