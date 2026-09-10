#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_auc_scatter_all.py
Per-species ROC-AUC: original vs compressed audio scatter plots.
MEE/BES Publication format:
  - Roman numerals (i-x) next to highlighted dots with direct leader arrows
  - Complete species names + Delta-AUC listed in a key box
  - Supports: birdnet, anuraset, northeastern
Usage:
    python scripts/downstream/figures/plot_auc_scatter_all.py --dataset birdnet
    python scripts/downstream/figures/plot_auc_scatter_all.py --dataset anuraset
    python scripts/downstream/figures/plot_auc_scatter_all.py --dataset northeastern
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from adjustText import adjust_text
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import PATHS

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS_DIR          = Path(PATHS['results_dir'])
BIRDNET_RES_DIR      = RESULTS_DIR / 'downstream' / 'pretrained' / 'birdnet'
ANURA_RES_DIR        = RESULTS_DIR / 'downstream' / 'transfer_learned' / 'on_original' / 'anuraset'
ANURA_EMB_DIR        = RESULTS_DIR / 'downstream' / 'transfer_learned' / 'embeddings' / 'anuraset'
NORTHEASTERN_RES_DIR = RESULTS_DIR / 'downstream' / 'transfer_learned' / 'on_original' / 'northeastern'
FIGS_BASE_DIR        = Path(PATHS['figures_dir']) / 'downstream'

# ---------------------------------------------------------------------------
# CONFIGS
# ---------------------------------------------------------------------------
N_LABEL        = 5
MIN_POS_TEST   = 10
TOP_QUARTILE_Q = 0.75
ROMAN_NUMS     = ['i', 'ii', 'iii', 'iv', 'v',
                  'vi', 'vii', 'viii', 'ix', 'x']
EVAL_CONFIGS = [
    ('encodec', 6.0,  'EnCodec 6.0 kbps'),
    ('encodec', 24.0, 'EnCodec 24.0 kbps'),
    ('dac',     5.33, 'DAC 5.33 kbps'),
    ('dac',     8.0,  'DAC 8.0 kbps'),
]
FIGURE_A = [('EnCodec 6.0 kbps', '#2171b5'),
            ('DAC 5.33 kbps',    '#d94801')]
FIGURE_B = [('EnCodec 24.0 kbps', '#08306b'),
            ('DAC 8.0 kbps',      '#7f2704')]

# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------
def load_n_pos_test_anuraset():
    """Compute n_pos_test per species from AnuraSet test label matrix."""
    import json
    labels  = np.load(ANURA_EMB_DIR / 'anuraset_test_labels.npy')
    with open(ANURA_EMB_DIR / 'anuraset_species_cols.json') as f:
        species = json.load(f)
    return {sp: int(labels[:, i].sum()) for i, sp in enumerate(species)}


def load_and_pivot_data(csv_path, dataset):
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    # If no codec/kbps columns, parse from condition (BirdNET format)
    if 'codec' not in df.columns or 'kbps' not in df.columns:
        DAC_KBPS = {2.0: 1.78, 3.0: 2.67, 6.0: 5.33, 9.0: 8.0}
        def parse_condition(cond):
            if cond == 'original':
                return 'original', None
            for c in ['encodec', 'dac']:
                if cond.startswith(c):
                    try:
                        bitrate = float(cond[len(c):])
                        kbps    = DAC_KBPS.get(bitrate, bitrate) if c == 'dac' \
                                else bitrate
                        return c, kbps
                    except Exception:
                        pass
            return cond, None
        parsed       = df['condition'].apply(parse_condition)
        df['codec']  = [p[0] for p in parsed]
        df['kbps']   = [p[1] for p in parsed]

    df = df.dropna(subset=['species', 'roc_auc'])
    dedup_cols = ['species', 'condition'] + \
                 (['codec'] if 'codec' in df.columns else [])
    df = df.drop_duplicates(subset=dedup_cols, keep='first')

    orig_df = df[df['condition'] == 'original'].copy()

    # Add n_pos_test
    if 'n_pos_test' in orig_df.columns:
        orig_df = orig_df.set_index('species')[['roc_auc', 'n_pos_test']].rename(
            columns={'roc_auc': 'auc_orig'})
    else:
        orig_df = orig_df.set_index('species')[['roc_auc']].rename(
            columns={'roc_auc': 'auc_orig'})
        if dataset == 'anuraset':
            n_pos_map = load_n_pos_test_anuraset()
            orig_df['n_pos_test'] = orig_df.index.map(n_pos_map)

    summary = orig_df.copy()
    for codec, kbps, label in EVAL_CONFIGS:
        codec_mask = df['codec'] == codec
        kbps_vals  = pd.to_numeric(df['kbps'], errors='coerce')
        kbps_mask  = np.isclose(kbps_vals.fillna(-1), float(kbps), atol=0.1)
        mask       = codec_mask & kbps_mask
        comp_df    = df[mask].copy()
        if comp_df.empty:
            print(f"  Warning: no rows for {label}")
            continue
        col     = f'auc_{codec}_{kbps}'
        comp_df = comp_df.set_index('species')[['roc_auc']].rename(
            columns={'roc_auc': col})
        summary = summary.join(comp_df, how='inner')
        summary[f'drop_{codec}_{kbps}'] = summary['auc_orig'] - summary[col]

    summary = summary.reset_index()

    # Filter to species with >= MIN_POS_TEST test positives
    if 'n_pos_test' in summary.columns:
        before = len(summary)
        summary = summary[summary['n_pos_test'] >= MIN_POS_TEST].copy()
        print(f"  After n_pos_test >= {MIN_POS_TEST} filter: {len(summary)} / {before} species")

    print(f"  Species in all conditions: {len(summary)}")
    return summary


def get_macro_metrics(metrics_csv_path, codec, kbps):
    if not metrics_csv_path.exists():
        return None, None
    m_df = pd.read_csv(metrics_csv_path)
    m_df['kbps_num']     = pd.to_numeric(m_df['kbps'],    errors='coerce')
    m_df['bitrate_num']  = pd.to_numeric(m_df.get('bitrate', pd.Series()), errors='coerce') \
                           if 'bitrate' in m_df.columns else pd.Series(dtype=float)
    match = m_df[(m_df['codec'] == codec) &
                 (np.isclose(m_df['kbps_num'].fillna(
                     m_df['bitrate_num'] if 'bitrate_num' in m_df.columns else -1),
                     float(kbps), atol=0.1))]
    if match.empty:
        cond_str = f"{codec}{int(kbps) if float(kbps).is_integer() else kbps}"
        match = m_df[m_df['condition'].str.contains(cond_str, case=False, na=False)]
    if not match.empty:
        orig_row   = m_df[m_df['condition'] == 'original']
        macro_orig = orig_row['macro_roc_auc'].values[0] \
                     if not orig_row.empty else m_df['macro_roc_auc'].values[0]
        macro_comp = match['macro_roc_auc'].values[0]
        return macro_orig, macro_comp
    return None, None

# ---------------------------------------------------------------------------
# SCATTER HELPER
# ---------------------------------------------------------------------------
def make_scatter(ax, summary, auc_col, drop_col, title,
                 panel_tag, auc_min, auc_max,
                 macro_metrics_path, codec_name, kbps_val,
                 show_ylabel=True):
    sub_summary = summary.copy()
    q75   = sub_summary['auc_orig'].quantile(TOP_QUARTILE_Q)
    top_q = sub_summary[sub_summary['auc_orig'] >= q75].copy()
    ld_df = top_q.nsmallest(N_LABEL, drop_col).copy().reset_index(drop=True)
    md_df = top_q.nlargest(N_LABEL,  drop_col).copy().reset_index(drop=True)
    most_deg  = md_df['species'].tolist()
    least_deg = ld_df['species'].tolist()

    # Degradation shading
    ax.fill_between([auc_min, auc_max], [auc_min, auc_max],
                    [auc_min, auc_min],
                    color='#ffcccc', alpha=0.35, zorder=1)
    # Diagonal
    ax.plot([auc_min, auc_max], [auc_min, auc_max],
            color='#444444', linestyle='--',
            linewidth=1.1, alpha=0.8, zorder=2)
    # Other species
    other = sub_summary[~sub_summary['species'].isin(most_deg + least_deg)]
    ax.scatter(other['auc_orig'], other[auc_col],
               s=60, color='#999999', alpha=0.50,
               linewidths=0, zorder=3)
    # Robust (green)
    ax.scatter(ld_df['auc_orig'], ld_df[auc_col],
               s=100, color='#2ca02c', alpha=0.95,
               linewidths=1.0, edgecolors='black', zorder=5)
    # Degraded (red)
    ax.scatter(md_df['auc_orig'], md_df[auc_col],
               s=100, color='#d62728', alpha=0.95,
               linewidths=1.0, edgecolors='black', zorder=5)

    # Roman numeral labels
    texts = []
    all_x = list(ld_df['auc_orig']) + list(md_df['auc_orig'])
    all_y = list(ld_df[auc_col])   + list(md_df[auc_col])
    for idx, r in ld_df.iterrows():
        t = ax.text(
            r['auc_orig'], r[auc_col],
            ROMAN_NUMS[idx],
            fontsize=11.0, color='#1b691b',
            ha='center', va='center', zorder=12,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#2ca02c', linewidth=0.8, alpha=0.92))
        texts.append(t)
    for idx, r in md_df.iterrows():
        t = ax.text(
            r['auc_orig'], r[auc_col],
            ROMAN_NUMS[N_LABEL + idx],
            fontsize=11.0, color='#a81c1c',
            ha='center', va='center', zorder=12,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#d62728', linewidth=0.8, alpha=0.92))
        texts.append(t)
    adjust_text(
        texts, ax=ax,
        x=all_x, y=all_y,
        expand_points=(1.5, 1.5),
        expand_text=(1.2, 1.2),
        arrowprops=dict(arrowstyle='->', color='#444444',
                        lw=0.65, shrinkA=0, shrinkB=3,
                        alpha=0.85),
        lim=300)

    # Key box with n_pos_test
    has_n = 'n_pos_test' in ld_df.columns
    key_lines = ['Least degraded (i-v):']
    for idx, r in ld_df.iterrows():
        delta   = r[auc_col] - r['auc_orig']
        n_str   = f", n={int(r['n_pos_test'])}" if has_n else ''
        key_lines.append(f"  {ROMAN_NUMS[idx]}. {r['species']} ({delta:+.3f}{n_str})")
    key_lines.append("")
    key_lines.append('Most degraded (vi-x):')
    for idx, r in md_df.iterrows():
        delta   = r[auc_col] - r['auc_orig']
        n_str   = f", n={int(r['n_pos_test'])}" if has_n else ''
        key_lines.append(f"  {ROMAN_NUMS[N_LABEL + idx]}. {r['species']} ({delta:+.3f}{n_str})")
    ax.text(0.04, 0.88, "\n".join(key_lines),
            transform=ax.transAxes,
            fontsize=11, va='top', ha='left', zorder=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffff',
                      alpha=0.92, edgecolor='#cccccc', linewidth=0.8))

    # Macro AUC badge
    macro_orig, macro_comp = get_macro_metrics(macro_metrics_path, codec_name, kbps_val)
    if macro_orig is None or macro_comp is None:
        macro_orig = sub_summary['auc_orig'].mean()
        macro_comp = sub_summary[auc_col].mean()
    ax.text(0.04, 0.97,
            f'Macro AUC: {macro_orig:.3f} → {macro_comp:.3f}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=12.0, color='#222222',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffffff',
                      alpha=0.92, edgecolor='#cccccc'))

    # Axes
    ax.set_xlim(auc_min, auc_max)
    ax.set_ylim(auc_min, auc_max)
    ax.set_aspect('equal')
    ax.set_title(f"{panel_tag} {title}", fontsize=15, pad=8, loc='left')
    ax.set_xlabel('ROC-AUC on original audio', fontsize=15)
    if show_ylabel:
        ax.set_ylabel('ROC-AUC on compressed audio', fontsize=15)
        explicit_ticks = [t for t in np.arange(auc_min, 1.01, 0.1)
                          if not np.isclose(t, auc_min, atol=0.01)]
        ax.set_xticks(explicit_ticks)
        ax.set_yticks(np.arange(auc_min, 1.01, 0.1))
    else:
        ax.tick_params(labelleft=False)
        ax.set_xticks(np.arange(auc_min, 1.01, 0.1))
        ax.set_yticks(np.arange(auc_min, 1.01, 0.1))
    ax.tick_params(labelsize=13, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=':', zorder=0)

# ---------------------------------------------------------------------------
# FIGURE BUILDER
# ---------------------------------------------------------------------------
def make_figure(fig_configs, summary, out_path, auc_min, auc_max, macro_metrics_path):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.8),
                             gridspec_kw={'wspace': 0.10})
    panel_tags = ['(a)', '(b)']
    col_map = {
        'EnCodec 6.0 kbps'  : ('auc_encodec_6.0',  'drop_encodec_6.0',  'encodec', 6.0),
        'EnCodec 24.0 kbps' : ('auc_encodec_24.0', 'drop_encodec_24.0', 'encodec', 24.0),
        'DAC 5.33 kbps'     : ('auc_dac_5.33',     'drop_dac_5.33',     'dac',     5.33),
        'DAC 8.0 kbps'      : ('auc_dac_8.0',      'drop_dac_8.0',      'dac',     8.0),
    }
    for ax, (label, _), show_y, tag in zip(
            axes, fig_configs, [True, False], panel_tags):
        auc_col, drop_col, codec_name, kbps_val = col_map[label]
        make_scatter(ax, summary, auc_col, drop_col,
                     label, tag, auc_min, auc_max,
                     macro_metrics_path, codec_name, kbps_val,
                     show_ylabel=show_y)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#2ca02c', markersize=9,
               markeredgecolor='black', markeredgewidth=0.7,
               label=f'Top {N_LABEL} least degraded (upper quartile, i–v)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#d62728', markersize=9,
               markeredgecolor='black', markeredgewidth=0.7,
               label=f'Top {N_LABEL} most degraded (upper quartile, vi–x)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#999999', markersize=8,
               markeredgewidth=0, label='Other species'),
        mpatches.Patch(facecolor='#ffcccc', alpha=0.6,
                       label='Degradation region (y < x)'),
        Line2D([0], [0], color='#444444', linestyle='--',
               linewidth=1.1, label='No change (y = x)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=15, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['birdnet', 'anuraset', 'northeastern'])
    args = parser.parse_args()

    out_dir = FIGS_BASE_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'birdnet':
        csv_file           = BIRDNET_RES_DIR / 'per_species_ovr.csv'
        macro_metrics_path = BIRDNET_RES_DIR / 'macro_metrics_ovr.csv'
        auc_min            = 0.50
        auc_max            = 1.01
    elif args.dataset == 'northeastern':
        csv_file           = NORTHEASTERN_RES_DIR / 'per_species_ovr.csv'
        macro_metrics_path = NORTHEASTERN_RES_DIR / 'macro_metrics_ovr.csv'
        auc_min            = 0.50
        auc_max            = 1.01
    else:  # anuraset
        csv_file           = ANURA_RES_DIR / 'per_species.csv'
        macro_metrics_path = ANURA_RES_DIR / 'macro_metrics.csv'
        auc_min            = 0.50
        auc_max            = 1.01

    summary = load_and_pivot_data(csv_file, args.dataset)
    summary.to_csv(out_dir / 'per_species_auc_pivoted.csv', index=False)

    print(f"\nGenerating Figure A ({args.dataset}), comparable bitrates...")
    make_figure(FIGURE_A, summary,
                out_dir / 'auc_scatter_comparable_bitrates.png',
                auc_min, auc_max, macro_metrics_path)

    print(f"\nGenerating Figure B ({args.dataset}), highest bitrates...")
    make_figure(FIGURE_B, summary,
                out_dir / 'auc_scatter_best_quality.png',
                auc_min, auc_max, macro_metrics_path)

    print(f"\nAll done. Outputs in: {out_dir}")

if __name__ == '__main__':
    main()