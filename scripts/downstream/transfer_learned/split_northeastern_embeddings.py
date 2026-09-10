#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/split_northeastern_embeddings.py

Splits northeastern_all_*.npy embedding files into train and test sets
based on the split column in northeastern_all_metadata.csv.

Saves:
    northeastern_train_original_embeddings.npy
    northeastern_test_original_embeddings.npy
    northeastern_train_{source}{bitrate}_embeddings.npy  (for each condition)
    northeastern_test_{source}{bitrate}_embeddings.npy
    northeastern_train_labels.npy
    northeastern_test_labels.npy
    northeastern_train_metadata.csv
    northeastern_test_metadata.csv

Usage:
    python scripts/downstream/transfer_learned/split_northeastern_embeddings.py
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

EPHEMERAL = Path(os.environ.get('EPHEMERAL', '.'))
EMB_DIR   = EPHEMERAL / 'northeastern_classifier' / 'embeddings'


def main():
    # Load metadata with split column
    meta_path = EMB_DIR / 'northeastern_all_metadata.csv'
    print(f"Loading metadata: {meta_path}")
    meta = pd.read_csv(meta_path)

    if 'split' not in meta.columns:
        print("ERROR: 'split' column not found — run add_northeastern_split.py first")
        return

    train_idx = meta[meta['split'] == 'train'].index.tolist()
    test_idx  = meta[meta['split'] == 'test'].index.tolist()

    print(f"  Train windows: {len(train_idx):,}")
    print(f"  Test windows:  {len(test_idx):,}")

    # Save split metadata
    meta.iloc[train_idx].reset_index(drop=True).to_csv(
        EMB_DIR / 'northeastern_train_metadata.csv', index=False)
    meta.iloc[test_idx].reset_index(drop=True).to_csv(
        EMB_DIR / 'northeastern_test_metadata.csv', index=False)
    print(f"  Saved: northeastern_train_metadata.csv")
    print(f"  Saved: northeastern_test_metadata.csv")

    # Split labels (same for all conditions)
    labels = np.load(EMB_DIR / 'northeastern_all_labels.npy')
    np.save(EMB_DIR / 'northeastern_train_labels.npy', labels[train_idx])
    np.save(EMB_DIR / 'northeastern_test_labels.npy',  labels[test_idx])
    print(f"  Saved: northeastern_train_labels.npy {labels[train_idx].shape}")
    print(f"  Saved: northeastern_test_labels.npy  {labels[test_idx].shape}")

    # Split all embedding files
    emb_files = sorted(EMB_DIR.glob('northeastern_all_*_embeddings.npy'))
    print(f"\n  Found {len(emb_files)} embedding files to split:")

    for emb_path in emb_files:
        # e.g. northeastern_all_original_embeddings.npy
        # → northeastern_train_original_embeddings.npy
        #   northeastern_test_original_embeddings.npy
        stem      = emb_path.stem  # northeastern_all_original_embeddings
        condition = stem.replace('northeastern_all_', '').replace('_embeddings', '')
        train_out = EMB_DIR / f'northeastern_train_{condition}_embeddings.npy'
        test_out  = EMB_DIR / f'northeastern_test_{condition}_embeddings.npy'

        if train_out.exists() and test_out.exists():
            print(f"  Already split: {condition} — skipping")
            continue

        print(f"  Splitting: {condition}...")
        emb = np.load(emb_path)
        np.save(train_out, emb[train_idx])
        np.save(test_out,  emb[test_idx])
        print(f"    Train: {emb[train_idx].shape} → {train_out.name}")
        print(f"    Test:  {emb[test_idx].shape}  → {test_out.name}")
        del emb

    print(f"\nDone.")


if __name__ == '__main__':
    main()