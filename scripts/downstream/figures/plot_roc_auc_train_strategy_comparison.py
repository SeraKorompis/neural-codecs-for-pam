#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_roc_auc_train_strategy_comparison.py

ROC-AUC vs bitrate comparing two training strategies:
  - Trained on original audio, evaluated on compressed
  - Domain-matched: trained and evaluated on same compressed audio

Two subplots per figure: one per codec (EnCodec | DAC).
Formatted strictly according to Methods in Ecology and Evolution (MEE) guidelines.

Usage:
    python scripts/downstream/figures/plot_roc_auc_train_strategy_comparison.py \
        --dataset lemur
    python scripts/downstream/figures/plot_roc_auc_train_strategy_comparison.py \
        --dataset anuraset
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import re

EPHEMERAL    = Path(os.environ.get('EPHEMERAL', '.'))
RESULTS_BASE = EPHEMERAL / 'ai_audio_compression' / 'results' / \
               'downstream' / 'transfer_learned'
FIGS_DIR     = EPHEMERAL / 'ai_audio_compression' / 'figures' / \
               'paper' / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DAC_KBPS = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}


def get_roc_col(df):
    for col in ['macro_roc_auc', 'roc_auc', 'roc_auc_dm', 'macro_auc']:
        if col in df.columns:
            return col
    return 'roc_auc'


def parse_condition_str(cond):
    cond = str(cond).strip().lower()
    if 'original' in cond or 'uncompressed' in cond:
        return 'original', None
    
    if cond.startswith('encodec'):
        num_part = re.sub(r'[^0-9.]', '', cond.replace('encodec', ''))
        kbps = float(num_part) if num_part else None
        return 'encodec', kbps
    elif cond.startswith('dac'):
        num_part = re.sub(r'[^0-9.]', '', cond.replace('dac', ''))
        if num_part:
            val = float(num_part)
            kbps = DAC_KBPS.get(int(val), val) if val in [2, 3, 6, 9] else val
            return 'dac', kbps
        return 'dac', None
    return 'unknown', None


def load_metrics(dataset, scenario):
    if scenario == 'on_original':
        path = RESULTS_BASE / 'on_original' / dataset / 'macro_metrics.csv'
    else:
        path = RESULTS_BASE / 'on_compressed' / dataset / 'macro_metrics.csv'

    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None

    df = pd.read_csv(path)
    roc_col = get_roc_col(df)

    if roc_col != 'roc_auc':
        df = df.rename(columns={roc_col: 'roc_auc'})

    if 'codec' not in df.columns or df['codec'].isna().all():
        if 'condition' in df.columns:
            df['codec'] = df['condition'].apply(lambda x: parse_condition_str(x)[0])
            if 'kbps' not in df.columns or df['kbps'].isna().all():
                df['kbps'] = df['condition'].apply(lambda x: parse_condition_str(x)[1])

    if 'kbps' in df.columns and not df['kbps'].isna().all():
        df['kbps'] = pd.to_numeric(df['kbps'], errors='coerce')
    else:
        def compute_kbps(r):
            if str(r.get('condition', '')).lower() == 'original':
                return np.nan
            if r.get('codec') == 'dac':
                b = float(r.get('bitrate', 0))
                return DAC_KBPS.get(int(b), b)
            return float(r.get('bitrate', 0))
        df['kbps'] = df.apply(compute_kbps, axis=1)

    df['roc_auc'] = pd.to_numeric(df['roc_auc'], errors='coerce')

    orig_rows = df[(df.get('condition') == 'original') | (df.get('codec') == 'original')]
    baseline  = float(orig_rows['roc_auc'].dropna().iloc[0]) if not orig_rows.empty else None

    enc = df[(df['codec'] == 'encodec') & df['kbps'].notna() & df['roc_auc'].notna()].copy().sort_values('kbps')
    dac = df[(df['codec'] == 'dac') & df['kbps'].notna() & df['roc_auc'].notna()].copy().sort_values('kbps')

    print(f"  {scenario}: {len(enc)} EnCodec rows, {len(dac)} DAC rows, "
          f"baseline={f'{baseline:.4f}' if baseline else 'N/A'}")

    return {
        'baseline': baseline,
        'encodec' : enc,
        'dac'     : dac,
    }


def plot_codec_panel(ax, orig_data, matched_data, codec,
                     color, panel_tag, codec_name, show_ylabel=True):
    """Plot one codec panel with original and domain-matched lines."""

    # 1. Original uncompressed baseline reference line
    if orig_data and orig_data['baseline'] is not None:
        ax.axhline(orig_data['baseline'], color='#444444', linestyle=':',
                   linewidth=1.3, alpha=0.85, zorder=2)

    # 2. Trained on original, evaluated on compressed
    if orig_data is not None:
        src = orig_data[codec]
        if not src.empty and 'roc_auc' in src.columns:
            ax.plot(src['kbps'], src['roc_auc'],
                    color=color, marker='o',
                    linewidth=2.0, markersize=6.5,
                    linestyle='-', alpha=0.9, zorder=4)

    # 3. Domain-matched: trained and evaluated on compressed
    if matched_data is not None:
        src_m = matched_data[codec]
        if not src_m.empty and 'roc_auc' in src_m.columns:
            ax.plot(src_m['kbps'], src_m['roc_auc'],
                    color=color, marker='s',
                    linewidth=2.0, markersize=6.5,
                    linestyle='--', alpha=0.9,
                    markerfacecolor='white', markeredgewidth=1.6, zorder=4)

    # Clean MEE Title placed OUTSIDE the plot box
    ax.set_title(f'{panel_tag} {codec_name}', loc='left',
                 fontsize=11.5, fontweight='bold', pad=10)

    ax.set_xlabel('Bitrate (kbps)', fontsize=10.5, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('ROC-AUC', fontsize=10.5, fontweight='bold')
    else:
        ax.tick_params(labelleft=False)

    ax.grid(True, alpha=0.25, linestyle=':', zorder=0)
    ax.set_xlim(left=0)

    # Calculate unified Y-axis range
    all_vals = []
    for d in [orig_data, matched_data]:
        if d is None:
            continue
        s = d[codec]
        if not s.empty and 'roc_auc' in s.columns:
            all_vals.extend(s['roc_auc'].dropna().tolist())
        if d['baseline'] is not None:
            all_vals.append(d['baseline'])

    if all_vals:
        y_min = max(0.0, min(all_vals) - 0.04)
        y_max = min(1.01, max(all_vals) + 0.03)
        ax.set_ylim(y_min, y_max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['lemur', 'anuraset'])
    args = parser.parse_args()

    print(f"\nLoading data for {args.dataset}...")
    orig_data    = load_metrics(args.dataset, 'on_original')
    matched_data = load_metrics(args.dataset, 'on_compressed')

    if orig_data is None and matched_data is None:
        print("ERROR: No evaluation results found for either regime.")
        return

    codecs_to_plot = []
    if (orig_data and not orig_data['encodec'].empty) or (matched_data and not matched_data['encodec'].empty):
        codecs_to_plot.append(('encodec', '#1f77b4', 'EnCodec'))
    if (orig_data and not orig_data['dac'].empty) or (matched_data and not matched_data['dac'].empty):
        codecs_to_plot.append(('dac', '#d94801', 'DAC'))

    n_plots = len(codecs_to_plot)
    if n_plots == 0:
        print("ERROR: No valid codec rows found.")
        return

    # Sized for standard double-column print layout
    fig, axes = plt.subplots(1, n_plots, figsize=(5.0 * n_plots, 4.4),
                             gridspec_kw={'wspace': 0.14})
    if n_plots == 1:
        axes = [axes]

    panel_tags = ['(a)', '(b)']

    for idx, (codec_key, color, codec_label) in enumerate(codecs_to_plot):
        plot_codec_panel(
            ax=axes[idx],
            orig_data=orig_data,
            matched_data=matched_data,
            codec=codec_key,
            color=color,
            panel_tag=panel_tags[idx],
            codec_name=codec_label,
            show_ylabel=(idx == 0)
        )

    # Consolidated figure-level legend with spacious placement
    baseline_val = orig_data['baseline'] if (orig_data and orig_data['baseline']) else 0.0
    legend_elements = [
        Line2D([0], [0], color='#444444', linestyle=':', linewidth=1.3,
               label=f'Original audio baseline ({baseline_val:.3f})'),
        Line2D([0], [0], color='#555555', marker='o', linestyle='-', linewidth=1.8,
               markersize=6, label='Trained on uncompressed (zero-shot)'),
        Line2D([0], [0], color='#555555', marker='s', linestyle='--', linewidth=1.8,
               markersize=6, markerfacecolor='white', markeredgewidth=1.5,
               label='Domain-matched (retrained on compressed)')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9.2, framealpha=0.95, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    
    out = FIGS_DIR / f'{args.dataset}_roc_auc_train_strategy_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved publication figure successfully to: {out}")


if __name__ == '__main__':
    main()