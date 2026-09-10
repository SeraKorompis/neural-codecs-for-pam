#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/create_temporal_split.py
Creates temporal 80/20 train/test split indices for Northeastern and Lemur.

Northeastern (stratified_temporal):
    - Per-species: earliest 80% of annotated windows -> train, latest 20% -> test
    - Background windows split temporally too
    Saves:
        northeastern_positive_train_indices.json
        northeastern_positive_test_indices.json
        northeastern_background_train_indices.npy
        northeastern_background_test_indices.npy
        northeastern_shared_species.json

Lemur (temporal):
    - All recorders combined, sorted chronologically
    - Roar and no-roar windows split separately (80/20)
    Saves:
        lemur_temporal_train_indices.npy
        lemur_temporal_test_indices.npy
        lemur_temporal_split_summary.csv

Usage:
    python scripts/downstream/transfer_learned/create_temporal_split.py \\
        --dataset northeastern --test_ratio 0.2 --min_pos_train 10

    python scripts/downstream/transfer_learned/create_temporal_split.py \\
        --dataset lemur --test_ratio 0.2
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

EMB_BASE = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings'


# ---------------------------------------------------------------------------
# NORTHEASTERN
# ---------------------------------------------------------------------------
def create_northeastern_split(test_ratio: float, min_pos_train: int):
    emb_dir = EMB_BASE / 'northeastern'
    meta    = pd.read_csv(emb_dir / 'northeastern_all_metadata.csv').reset_index(drop=True)
    labels  = np.load(emb_dir / 'northeastern_all_labels.npy')

    with open(emb_dir / 'northeastern_species_cols.json') as f:
        species = json.load(f)

    print(f"  Total windows: {len(meta):,}")
    print(f"  Labels shape:  {labels.shape}")
    print(f"  Species:       {len(species)}")

    # Sort chronologically
    dates     = meta['date'].astype(int).values
    sort_keys = np.argsort(dates, kind='stable')

    # Per-species positive split
    positive_train_idx = {}
    positive_test_idx  = {}
    shared_species     = []
    skipped_species    = []

    for sp_idx, sp in enumerate(species):
        sp_mask    = labels[:, sp_idx] > 0
        sp_windows = sort_keys[sp_mask[sort_keys]]
        if len(sp_windows) == 0:
            skipped_species.append((sp, 0, 0))
            continue
        # Split latest test_ratio fraction for testing, earliest for training
        n_test  = max(1, int(len(sp_windows) * test_ratio)) 
        n_train = len(sp_windows) - n_test 
        train_idx = sp_windows[:n_train].tolist() 
        test_idx  = sp_windows[n_train:].tolist() 
        positive_train_idx[sp] = train_idx
        positive_test_idx[sp]  = test_idx
        if len(train_idx) >= min_pos_train:
            shared_species.append(sp)
        else:
            skipped_species.append((sp, len(train_idx), len(test_idx)))

    print(f"  Species with >= {min_pos_train} train positives: {len(shared_species)}")
    print(f"  Species skipped: {len(skipped_species)}")

    # Background split
    bg_mask    = labels.sum(axis=1) == 0
    bg_windows = sort_keys[bg_mask[sort_keys]]
    n_bg_test  = max(1, int(len(bg_windows) * test_ratio))
    bg_train   = bg_windows[:-n_bg_test]
    bg_test    = bg_windows[-n_bg_test:]

    print(f"  Background train: {len(bg_train):,}")
    print(f"  Background test:  {len(bg_test):,}")

    # Save positive and background train and test indices
    with open(emb_dir / 'northeastern_positive_train_indices.json', 'w') as f:
        json.dump(positive_train_idx, f)
    with open(emb_dir / 'northeastern_positive_test_indices.json', 'w') as f:
        json.dump(positive_test_idx, f)
    np.save(emb_dir / 'northeastern_background_train_indices.npy', bg_train)
    np.save(emb_dir / 'northeastern_background_test_indices.npy',  bg_test)
    with open(emb_dir / 'northeastern_shared_species.json', 'w') as f:
        json.dump(shared_species, f, indent=2)

    print(f"  Saved all index files to {emb_dir}")


# ---------------------------------------------------------------------------
# LEMUR
# ---------------------------------------------------------------------------
def extract_timestamp(filename):
    parts = Path(filename).stem.split('_')
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            date = p
            time = parts[i+1] if (i+1 < len(parts) and
                                   len(parts[i+1]) == 6 and
                                   parts[i+1].isdigit()) else '000000'
            return pd.Timestamp(
                f'{date[:4]}-{date[4:6]}-{date[6:8]} '
                f'{time[:2]}:{time[2:4]}:{time[4:6]}')
    return pd.NaT


def create_lemur_split(test_ratio: float):
    emb_dir = EMB_BASE / 'lemur'
    meta    = pd.read_csv(emb_dir / 'lemur_all_metadata.csv')

    print(f"  Total windows: {len(meta):,}")
    print(f"  Roar:    {(meta['label']==1).sum():,}")
    print(f"  No-roar: {(meta['label']==0).sum():,}")
    print(f"  Recorders: {meta['recorder'].value_counts().to_dict()}")

    # Extract timestamps and sort chronologically
    meta['timestamp'] = meta['filename'].apply(extract_timestamp)
    meta['datetime']  = meta['timestamp'] + pd.to_timedelta(
        meta['window_start'], unit='s')
    meta = meta.sort_values('datetime').reset_index(drop=True)

    print(f"\n  Date range: {meta['datetime'].min()} to {meta['datetime'].max()}")
    print('\n  Recorder date ranges:')
    print(meta.groupby('recorder')['datetime'].agg(['min','max']).to_string())

    # Split roar and no-roar separately
    roar_idx    = meta[meta['label'] == 1].index.values
    no_roar_idx = meta[meta['label'] == 0].index.values

    n_roar_train    = int(len(roar_idx)    * (1 - test_ratio))
    n_no_roar_train = int(len(no_roar_idx) * (1 - test_ratio))

    roar_train    = roar_idx[:n_roar_train]
    roar_test     = roar_idx[n_roar_train:]
    no_roar_train = no_roar_idx[:n_no_roar_train]
    no_roar_test  = no_roar_idx[n_no_roar_train:]

    train_idx = np.sort(np.concatenate([roar_train, no_roar_train]))
    test_idx  = np.sort(np.concatenate([roar_test,  no_roar_test]))

    print(f"\n  Train: {len(train_idx):,} "
          f"({len(roar_train):,} roar, {len(no_roar_train):,} no-roar)")
    print(f"  Test:  {len(test_idx):,} "
          f"({len(roar_test):,} roar, {len(no_roar_test):,} no-roar)")

    print('\n  Recorder distribution in train:')
    print(meta.loc[train_idx, 'recorder'].value_counts().to_string())
    print('\n  Recorder distribution in test:')
    print(meta.loc[test_idx, 'recorder'].value_counts().to_string())

    # Save indices
    np.save(emb_dir / 'lemur_temporal_train_indices.npy', train_idx)
    np.save(emb_dir / 'lemur_temporal_test_indices.npy',  test_idx)

    # Save summary for recorder distribution analysis
    meta['split'] = 'unassigned'
    meta.loc[train_idx, 'split'] = 'train'
    meta.loc[test_idx,  'split'] = 'test'
    meta.to_csv(emb_dir / 'lemur_temporal_split_summary.csv', index=False)

    print(f"\n  Saved index files and summary to {emb_dir}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Create temporal train/test split for Northeastern or Lemur')
    parser.add_argument('--dataset', required=True,
                        choices=['northeastern', 'lemur'])
    parser.add_argument('--test_ratio', default=0.2, type=float,
                        help='Test fraction (default 0.2)')
    parser.add_argument('--min_pos_train', default=10, type=int,
                        help='Northeastern only: min positive train windows '
                             'to include species (default 10)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Creating temporal split: {args.dataset}")
    print(f"  test_ratio: {args.test_ratio}")
    if args.dataset == 'northeastern':
        print(f"  min_pos_train: {args.min_pos_train}")
    print(f"{'='*60}\n")

    if args.dataset == 'northeastern':
        create_northeastern_split(args.test_ratio, args.min_pos_train)
    else:
        create_lemur_split(args.test_ratio)

    print('\nDone.')


if __name__ == '__main__':
    main()