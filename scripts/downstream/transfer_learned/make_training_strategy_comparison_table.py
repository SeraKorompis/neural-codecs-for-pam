#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/make_training_strategy_comparison_table.py

Produces a comparison table of ROC-AUC for two training strategies:
    1. Trained on original audio, evaluated on compressed
    2. Domain-matched: trained and evaluated on same compressed audio

Saves to:
    results/downstream/transfer_learned/
        training_strategy_comparison_{dataset}.csv

Usage:
    python scripts/downstream/transfer_learned/make_training_strategy_comparison_table.py \
        --dataset lemur
    python scripts/downstream/transfer_learned/make_training_strategy_comparison_table.py \
        --dataset anuraset
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

EPHEMERAL    = Path(os.environ.get('EPHEMERAL', '.'))
RESULTS_BASE = EPHEMERAL / 'ai_audio_compression' / 'results' / \
               'downstream' / 'transfer_learned'

DAC_KBPS = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}


def get_roc_col(df):
    if 'macro_roc_auc' in df.columns:
        return 'macro_roc_auc'
    return 'roc_auc'


def load_and_normalise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    roc_col = get_roc_col(df)
    if roc_col != 'roc_auc':
        df = df.rename(columns={roc_col: 'roc_auc'})
    if 'kbps' not in df.columns:
        df['kbps'] = df.apply(
            lambda r: DAC_KBPS.get(int(r['bitrate']), float(r['bitrate']))
            if r['codec'] == 'dac' else float(r['bitrate']),
            axis=1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['lemur', 'anuraset'])
    args = parser.parse_args()

    # Load on_original results
    orig_path = RESULTS_BASE / 'on_original' / args.dataset / 'macro_metrics.csv'
    if not orig_path.exists():
        print(f"ERROR: not found: {orig_path}")
        return
    orig_df = load_and_normalise(orig_path)

    # Original baseline
    baseline = float(orig_df[orig_df['condition'] == 'original']['roc_auc'].iloc[0])
    print(f"  Original baseline ROC-AUC: {baseline:.4f}")

    # Load on_compressed results
    comp_path = RESULTS_BASE / 'on_compressed' / args.dataset / 'macro_metrics.csv'
    comp_df = None
    if comp_path.exists():
        comp_df = load_and_normalise(comp_path)
    else:
        print(f"  WARNING: no domain-matched results found: {comp_path}")

    # Build comparison table
    rows = []

    # Original baseline row
    rows.append({
        'dataset'              : args.dataset,
        'codec'                : 'original',
        'kbps'                 : None,
        'roc_auc_on_original'  : baseline,
        'roc_auc_domain_matched': None,
        'delta'                : None,
        'note'                 : 'uncompressed baseline',
    })

    # Compressed conditions
    compressed = orig_df[orig_df['condition'] != 'original'].copy()
    for _, row in compressed.iterrows():
        codec   = row['codec']
        kbps    = row['kbps']
        roc_orig = row['roc_auc']

        # Find matching domain-matched result
        roc_matched = None
        if comp_df is not None:
            mask = (comp_df['codec'] == codec) & \
                   (np.isclose(comp_df['kbps'].astype(float),
                               float(kbps), atol=0.1))
            if 'train_source' in comp_df.columns:
                mask = mask & (comp_df['train_source'] == codec)
            matches = comp_df[mask]
            if not matches.empty:
                roc_matched = float(matches['roc_auc'].iloc[0])

        delta = (roc_matched - roc_orig) if roc_matched is not None else None

        rows.append({
            'dataset'               : args.dataset,
            'codec'                 : codec,
            'kbps'                  : kbps,
            'roc_auc_on_original'   : round(roc_orig, 4),
            'roc_auc_domain_matched': round(roc_matched, 4) if roc_matched else None,
            'delta'                 : round(delta, 4) if delta is not None else None,
            'note'                  : 'domain-matched available' if roc_matched
                                      else 'domain-matched not yet available',
        })

    table = pd.DataFrame(rows)

    # Sort by codec then kbps
    codec_order = {'original': 0, 'encodec': 1, 'dac': 2}
    table['codec_order'] = table['codec'].map(codec_order)
    table = table.sort_values(['codec_order', 'kbps']).drop(
        columns='codec_order').reset_index(drop=True)

    out_path = RESULTS_BASE / f'training_strategy_comparison_{args.dataset}.csv'
    table.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"\n{table.to_string(index=False)}")


if __name__ == '__main__':
    main()