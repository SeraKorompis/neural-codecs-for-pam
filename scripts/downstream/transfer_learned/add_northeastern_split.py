#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/add_northeastern_split.py

Adds a train/test split to northeastern embeddings metadata and generates
per-species positive and background window indices for binary classification.

Supports four split strategies:
    stratified_temporal -- window-level per-species temporal split (recommended)
                           for each species, earliest 80% of annotated windows
                           → train, latest 20% → test
                           background windows split temporally too
    temporal            -- simple date cutoff
    blocked             -- interleaved date blocks
    random              -- random recording-level split

For stratified_temporal, saves:
    northeastern_positive_train_indices.json
    northeastern_positive_test_indices.json
    northeastern_background_train_indices.npy
    northeastern_background_test_indices.npy
    northeastern_shared_species.json  (species with >= min_pos_train windows)

Usage:
    python scripts/downstream/transfer_learned/add_northeastern_split.py \\
        --strategy stratified_temporal --test_ratio 0.2 --min_pos_train 10
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

EPHEMERAL    = Path(os.environ.get('EPHEMERAL', '.'))
EMB_DIR      = EPHEMERAL / 'northeastern_classifier' / 'embeddings'
META_PATH    = EMB_DIR / 'northeastern_all_metadata.csv'
LABELS_PATH  = EMB_DIR / 'northeastern_all_labels.npy'
SPECIES_PATH = EMB_DIR / 'northeastern_species_cols.json'


def split_temporal(meta, cutoff):
    return meta['date'].astype(int).apply(
        lambda d: 'train' if d <= cutoff else 'test')


def split_blocked(meta, test_every):
    unique_dates = sorted(meta['date'].astype(int).unique().tolist())
    date_split   = {
        date: 'test' if (i % test_every == 1) else 'train'
        for i, date in enumerate(unique_dates)
    }
    print(f"\n  Blocked split (every {test_every}nd date → test):")
    for date, split in date_split.items():
        n = (meta['date'].astype(int) == date).sum()
        print(f"    {date}: {split} ({n} windows)")
    return meta['date'].astype(int).map(date_split)


def split_random(meta, test_ratio, seed=42):
    rng       = np.random.default_rng(seed)
    filenames = meta['filename'].unique()
    n_test    = int(len(filenames) * test_ratio)
    test_files = set(rng.choice(filenames, size=n_test, replace=False))
    return meta['filename'].apply(
        lambda f: 'test' if f in test_files else 'train')


def split_stratified_temporal(meta, labels, species, test_ratio, min_pos_train):
    """
    Window-level per-species temporal stratification.

    For each species:
      - Find all windows where species is annotated
      - Sort by (date, window position)
      - Earliest (1-test_ratio) → train positives
      - Latest test_ratio → test positives

    Background windows (no annotations at all):
      - Sort by (date, window position)
      - Earliest (1-test_ratio) → train background
      - Latest test_ratio → test background

    Returns:
        positive_train_idx: dict {species: list of int}
        positive_test_idx:  dict {species: list of int}
        bg_train_idx:       np.ndarray of int
        bg_test_idx:        np.ndarray of int
        shared_species:     list of species with >= min_pos_train train windows
    """
    n_windows = len(meta)
    print(f"  Total windows: {n_windows:,}")

    # Build sort key: (date, row_index) for chronological ordering
    dates     = meta['date'].astype(int).values
    sort_keys = np.argsort(dates, kind='stable')  # stable sort preserves file order

    # Per-species positive indices
    positive_train_idx = {}
    positive_test_idx  = {}
    shared_species     = []
    skipped_species    = []

    for sp_idx, sp in enumerate(species):
        # All windows annotated for this species, in chronological order
        sp_mask    = labels[:, sp_idx] > 0
        sp_windows = sort_keys[sp_mask[sort_keys]]  # chronologically sorted indices

        if len(sp_windows) == 0:
            skipped_species.append((sp, 0, 0))
            continue

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
    print(f"  Species skipped (too few train positives): {len(skipped_species)}")
    if skipped_species:
        for sp, n_tr, n_te in skipped_species:
            print(f"    {sp}: train={n_tr}, test={n_te}")

    # Background indices — windows with no annotations at all
    bg_mask    = labels.sum(axis=1) == 0
    bg_windows = sort_keys[bg_mask[sort_keys]]

    n_bg_test  = max(1, int(len(bg_windows) * test_ratio))
    n_bg_train = len(bg_windows) - n_bg_test

    bg_train_idx = bg_windows[:n_bg_train]
    bg_test_idx  = bg_windows[n_bg_train:]

    print(f"  Background train windows: {len(bg_train_idx):,}")
    print(f"  Background test windows:  {len(bg_test_idx):,}")

    return (positive_train_idx, positive_test_idx,
            bg_train_idx, bg_test_idx, shared_species)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='stratified_temporal',
                        choices=['temporal', 'blocked',
                                 'stratified_temporal', 'random'])
    parser.add_argument('--cutoff',       default='20170520',
                        help='Cutoff date for temporal split (YYYYMMDD)')
    parser.add_argument('--test_every',   default=2, type=int,
                        help='For blocked: every Nth date → test')
    parser.add_argument('--test_ratio',   default=0.2, type=float,
                        help='Test fraction (default 0.2)')
    parser.add_argument('--min_pos_train', default=10, type=int,
                        help='Min positive train windows to include species')
    args = parser.parse_args()

    print(f"Loading metadata: {META_PATH}")
    meta = pd.read_csv(META_PATH)
    meta = meta.reset_index(drop=True)
    print(f"  Total windows: {len(meta):,}")
    print(f"  Strategy: {args.strategy}")

    if args.strategy == 'stratified_temporal':
        if not LABELS_PATH.exists() or not SPECIES_PATH.exists():
            print("ERROR: labels or species file not found")
            return

        labels = np.load(LABELS_PATH)
        with open(SPECIES_PATH) as f:
            species = json.load(f)

        print(f"  Labels shape: {labels.shape}")
        print(f"  Species: {len(species)}")

        (pos_train, pos_test,
         bg_train, bg_test,
         shared_species) = split_stratified_temporal(
            meta, labels, species, args.test_ratio, args.min_pos_train)

        # Save indices
        out_pos_train = EMB_DIR / 'northeastern_positive_train_indices.json'
        out_pos_test  = EMB_DIR / 'northeastern_positive_test_indices.json'
        out_bg_train  = EMB_DIR / 'northeastern_background_train_indices.npy'
        out_bg_test   = EMB_DIR / 'northeastern_background_test_indices.npy'
        out_species   = EMB_DIR / 'northeastern_shared_species.json'

        with open(out_pos_train, 'w') as f:
            json.dump(pos_train, f)
        print(f"  Saved: {out_pos_train.name}")

        with open(out_pos_test, 'w') as f:
            json.dump(pos_test, f)
        print(f"  Saved: {out_pos_test.name}")

        np.save(out_bg_train, bg_train)
        print(f"  Saved: {out_bg_train.name} ({len(bg_train):,} windows)")

        np.save(out_bg_test, bg_test)
        print(f"  Saved: {out_bg_test.name} ({len(bg_test):,} windows)")

        with open(out_species, 'w') as f:
            json.dump(shared_species, f, indent=2)
        print(f"  Saved: {out_species.name} ({len(shared_species)} species)")

        # Print per-species summary for shared species
        print(f"\n  Per-species summary (shared species only):")
        print(f"  {'Species':40s} {'Train+':>8} {'Test+':>7} {'BG_tr':>8} {'BG_te':>7}")
        print(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")
        for sp in shared_species:
            print(f"  {sp:40s} {len(pos_train[sp]):>8} "
                  f"{len(pos_test[sp]):>7} "
                  f"{len(bg_train):>8} "
                  f"{len(bg_test):>7}")

    else:
        # Simple splits — just update metadata split column
        if args.strategy == 'temporal':
            meta['split'] = split_temporal(meta, int(args.cutoff))
        elif args.strategy == 'blocked':
            meta['split'] = split_blocked(meta, args.test_every)
        else:  # random
            meta['split'] = split_random(meta, args.test_ratio)

        train_w = (meta['split'] == 'train').sum()
        test_w  = (meta['split'] == 'test').sum()
        print(f"  Train: {train_w:,} windows")
        print(f"  Test:  {test_w:,} windows")
        print(f"  Ratio: {train_w/(train_w+test_w):.2f} / "
              f"{test_w/(train_w+test_w):.2f}")

        meta.to_csv(META_PATH, index=False)
        print(f"  Saved: {META_PATH}")

    print("\nDone.")


if __name__ == '__main__':
    main()