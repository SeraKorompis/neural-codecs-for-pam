#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_boxes_transfer_learned_delta.py
Box plot showing per-species ROC-AUC improvement from domain-matched training.
Delta = ROC-AUC(trained_on_compressed) - ROC-AUC(trained_on_original)
for each species, at lowest and highest EnCodec bitrates.

Usage:
    python scripts/downstream/figures/plot_per_species_domain_adaptation.py \
        --dataset northeastern
    python scripts/downstream/figures/plot_per_species_domain_adaptation.py \
        --dataset anuraset
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

RESULTS_BASE = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned'
EMB_BASE     = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings'
FIGS_DIR     = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

MIN_POS_TEST = 10
FONTSIZE     = 18
TICK_SIZE    = 14
LINEWIDTH    = 1.5

DATASET_CONFIG = {
    'northeastern': {
        'orig_path'   : RESULTS_BASE / 'on_original'   / 'northeastern' / 'per_species_ovr.csv',
        'comp_path'   : RESULTS_BASE / 'on_compressed' / 'northeastern' / 'per_species_ovr.csv',
        'roc_col'     : 'roc_auc',
        'species_col' : 'species',
        'label'       : 'Northeastern US Soundscapes',
        'n_pos_col'   : 'n_pos_test',   # exists in file
        'emb_dir'     : None,           # not needed
    },
    'anuraset': {
        'orig_path'   : RESULTS_BASE / 'on_original'   / 'anuraset' / 'per_species.csv',
        'comp_path'   : RESULTS_BASE / 'on_compressed' / 'anuraset' / 'per_species.csv',
        'roc_col'     : 'roc_auc',
        'species_col' : 'species',
        'label'       : 'AnuraSet',
        'n_pos_col'   : None,           # computed from label matrix
        'emb_dir'     : EMB_BASE / 'anuraset',
    },
}

CONDITIONS = {
    'lowest' : 'encodec1.5',
    'highest': 'encodec24.0',
}


def get_n_pos_test(config):
    """Return dict of {species: n_pos_test}."""
    if config['n_pos_col'] is not None:
        # Read from file directly
        df  = pd.read_csv(config['orig_path'])
        orig = df[df['condition'] == 'original']
        return dict(zip(orig[config['species_col']],
                        orig[config['n_pos_col']].astype(int)))
    else:
        # Compute from label matrix
        emb_dir = config['emb_dir']
        labels  = np.load(emb_dir / 'anuraset_test_labels.npy')
        with open(emb_dir / 'anuraset_species_cols.json') as f:
            species = json.load(f)
        n_pos = labels.sum(axis=0).astype(int)
        return dict(zip(species, n_pos))


def load_delta(config, condition, n_pos_dict):
    """
    For a given condition, compute per-species delta:
    ROC-AUC(trained_on_compressed) - ROC-AUC(trained_on_original)
    """
    orig = pd.read_csv(config['orig_path'])
    comp = pd.read_csv(config['comp_path'])

    orig_cond = orig[orig['condition'] == condition][
        [config['species_col'], config['roc_col']]
    ].rename(columns={config['roc_col']: 'roc_orig'})

    comp_cond = comp[comp['condition'] == condition][
        [config['species_col'], config['roc_col']]
    ].rename(columns={config['roc_col']: 'roc_comp'})

    merged = orig_cond.merge(comp_cond, on=config['species_col'], how='inner')
    merged['delta']      = merged['roc_comp'] - merged['roc_orig']
    merged['n_pos_test'] = merged[config['species_col']].map(n_pos_dict)

    # Filter by minimum positives
    merged = merged[merged['n_pos_test'] >= MIN_POS_TEST].copy()
    return merged


def plot_panel(ax, delta_df, species_col, condition_label,
               show_xlabel, panel_label):
    # Sort species by median delta
    order = (delta_df.groupby(species_col)['delta']
             .median()
             .sort_values()
             .index.tolist())

    positions = list(range(len(order)))
    data      = [delta_df[delta_df[species_col] == sp]['delta'].values
                 for sp in order]

    bp = ax.boxplot(data, positions=positions, vert=True,
                    patch_artist=True, widths=0.6,
                    showfliers=True,
                    medianprops=dict(color='black', linewidth=2.0),
                    boxprops=dict(facecolor='#AEC6E8', linewidth=LINEWIDTH),
                    whiskerprops=dict(linewidth=LINEWIDTH),
                    capprops=dict(linewidth=LINEWIDTH),
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor='grey',
                                    markeredgecolor='grey', alpha=0.5))

    ax.axhline(0, color='black', linestyle='--',
               linewidth=1.2, alpha=0.6, zorder=0)
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.3,
               alpha=0.03, color='green')
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.3, 0,
               alpha=0.03, color='red')

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=90, ha='center', fontsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_title(condition_label, fontsize=FONTSIZE, pad=10)
    ax.set_ylabel(
        '$\\Delta$ ROC-AUC\n(domain-matched $-$ trained on original)',
        fontsize=FONTSIZE)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.01, 0.98, panel_label, transform=ax.transAxes,
            fontsize=FONTSIZE, va='top', ha='left')

    if show_xlabel:
        ax.set_xlabel('Species', fontsize=FONTSIZE, labelpad=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['northeastern', 'anuraset'])
    args   = parser.parse_args()
    config = DATASET_CONFIG[args.dataset]

    print(f"Loading {args.dataset}...")
    n_pos_dict = get_n_pos_test(config)
    print(f"  Species with >= {MIN_POS_TEST} positives: "
          f"{sum(v >= MIN_POS_TEST for v in n_pos_dict.values())}")

    delta_low  = load_delta(config, CONDITIONS['lowest'],  n_pos_dict)
    delta_high = load_delta(config, CONDITIONS['highest'], n_pos_dict)
    print(f"  Lowest  ({CONDITIONS['lowest']}):  {len(delta_low)} species")
    print(f"  Highest ({CONDITIONS['highest']}): {len(delta_high)} species")

    # Sort order shared across both panels — by median delta at lowest bitrate
    shared_order = (delta_low.groupby(config['species_col'])['delta']
                    .median()
                    .sort_values()
                    .index.tolist())

    # Keep only species present in both conditions
    shared_species = set(delta_low[config['species_col']]) & \
                     set(delta_high[config['species_col']])
    shared_order   = [sp for sp in shared_order if sp in shared_species]
    delta_low      = delta_low[delta_low[config['species_col']].isin(shared_species)]
    delta_high     = delta_high[delta_high[config['species_col']].isin(shared_species)]

    n_sp = len(shared_order)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(16, n_sp * 0.4), 18),
        gridspec_kw={'hspace': 0.5})

    plt.rcParams['font.family'] = 'Arial'

    for ax, delta_df, cond_key, panel_label, show_xlabel in [
        (axes[0], delta_low,  'lowest',  '(a)', False),
        (axes[1], delta_high, 'highest', '(b)', True),
    ]:
        cond_label = (f'EnCodec {CONDITIONS[cond_key].replace("encodec", "")} kbps')
        # Reorder to shared_order
        delta_df = delta_df.set_index(config['species_col']).loc[shared_order].reset_index()
        plot_panel(ax, delta_df, config['species_col'],
                   cond_label, show_xlabel, panel_label)

    fig.suptitle(
        f'{config["label"]} — Per-species ROC-AUC gain from domain-matched training',
        fontsize=FONTSIZE + 2, y=1.01)

    out = FIGS_DIR / f'per_species_domain_adaptation_{args.dataset}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()