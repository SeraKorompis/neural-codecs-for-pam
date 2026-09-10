#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/train_northeastern_ovr_classifiers.py
Trains per-species binary classifiers for Northeastern US Soundscapes
using sklearn OneVsRestClassifier:
    - Positives: annotated windows for each species (temporal train split)
    - Negatives: ALL non-positive training windows (including other species)
    - OneVsRestClassifier(LogisticRegression) trained jointly across all species
"""
import argparse
import json
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import PATHS

RESULTS_BASE = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned'
EMB_DIR      = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'northeastern'

LOGREG_PARAMS = dict(max_iter=1000, C=1.0, solver='lbfgs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_source', default='original',
                        choices=['original', 'encodec', 'dac', 'mp3', 'opus'])
    parser.add_argument('--train_bitrate', default=None)
    args = parser.parse_args()

    if args.train_source != 'original' and args.train_bitrate is None:
        parser.error('--train_bitrate required when --train_source is not original')

    if args.train_source == 'original':
        source_tag = 'original'
        clf_dir    = RESULTS_BASE / 'on_original' / 'northeastern' / \
                     'classifiers' / f'binary_classifiers_ovr_{source_tag}'
    else:
        train_bitrate = (float(args.train_bitrate) if '.' in args.train_bitrate
                         else int(args.train_bitrate))
        source_tag = f'{args.train_source}{train_bitrate}'
        clf_dir    = RESULTS_BASE / 'on_compressed' / 'northeastern' / \
                     'classifiers' / f'binary_classifiers_ovr_{source_tag}'

    clf_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training OneVsRestClassifier: northeastern")
    print(f"  train source: {source_tag}")
    print(f"  clf_dir:      {clf_dir}")
    print(f"{'='*60}")

    # Load embeddings
    if args.train_source == 'original':
        emb_path = EMB_DIR / 'northeastern_all_original_embeddings_fixed.npy'
    else:
        emb_path = EMB_DIR / f'northeastern_all_{source_tag}_embeddings.npy'

    if not emb_path.exists():
        print(f"ERROR: embeddings not found: {emb_path}")
        sys.exit(1)

    print(f"\nLoading embeddings: {emb_path.name}")
    embeddings = np.load(emb_path)
    print(f"  Shape: {embeddings.shape}")

    # Load indices and species
    with open(EMB_DIR / 'northeastern_positive_train_indices.json') as f:
        pos_train_idx = json.load(f)

    bg_train_idx = np.load(EMB_DIR / 'northeastern_background_train_indices.npy')

    with open(EMB_DIR / 'northeastern_species_cols.json') as f:
        species = json.load(f)

    # Build full train index set
    all_pos_idx   = np.array(list({idx for idxs in pos_train_idx.values()
                                   for idx in idxs}), dtype=int)
    all_train_idx = np.unique(np.concatenate([all_pos_idx, bg_train_idx]))

    print(f"  Species: {len(species)}")
    print(f"  Total train windows: {len(all_train_idx):,}")

    # Build multilabel matrix for all train windows
    print(f"\nBuilding multilabel label matrix...")
    n_train   = len(all_train_idx)
    n_species = len(species)
    idx_map   = {idx: i for i, idx in enumerate(all_train_idx)}
    y_train   = np.zeros((n_train, n_species), dtype=np.float32)

    for j, sp in enumerate(species):
        for idx in pos_train_idx.get(sp, []):
            if idx in idx_map:
                y_train[idx_map[idx], j] = 1.0

    print(f"  Label matrix: {y_train.shape}")
    print(f"  Positive rate: {y_train.mean():.4f}")

    # Scale embeddings
    print(f"\nFitting scaler...")
    X_train = embeddings[all_train_idx]
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, clf_dir / 'scaler.joblib')
    print(f"  Scaler fit on {n_train:,} windows")

    # Train OneVsRestClassifier
    print(f"\nTraining OneVsRestClassifier...")
    clf = OneVsRestClassifier(
        LogisticRegression(**LOGREG_PARAMS),
        n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"  Done.")

    # Save classifier and scaler
    joblib.dump(clf, clf_dir / 'classifier.joblib')
    print(f"  Saved classifier to {clf_dir / 'classifier.joblib'}")

    # Quick train evaluation per species
    print(f"\nEvaluating on train set...")
    y_scores = clf.predict_proba(X_train)
    results  = {}

    for j, sp in enumerate(species):
        if y_train[:, j].sum() == 0:
            continue
        auc = float(roc_auc_score(y_train[:, j], y_scores[:, j]))

        best_f1, best_thr_f1 = 0.0, 0.5
        for thr in np.arange(0.1, 0.91, 0.05):
            preds = (y_scores[:, j] >= thr).astype(int)
            f1    = f1_score(y_train[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr_f1 = f1, thr

        fpr, tpr, thr_roc = roc_curve(y_train[:, j], y_scores[:, j])
        dist         = np.sqrt((1 - tpr)**2 + fpr**2)
        best_thr_roc = float(thr_roc[np.argmin(dist)])

        results[sp] = {
            'train_roc_auc' : auc,
            'n_pos_train'   : int(y_train[:, j].sum()),
            'threshold_f1'  : float(best_thr_f1),
            'threshold_roc' : float(best_thr_roc),
        }
        print(f"  {sp:40s} n_pos={int(y_train[:,j].sum()):5d}  "
              f"train_auc={auc:.3f}")

    # Save thresholds and results
    thresholds_roc = np.array([results[sp]['threshold_roc']
                                for sp in species if sp in results])
    thresholds_f1  = np.array([results[sp]['threshold_f1']
                                for sp in species if sp in results])
    np.save(clf_dir / 'optimal_thresholds_roc.npy', thresholds_roc)
    np.save(clf_dir / 'optimal_thresholds_f1.npy',  thresholds_f1)

    with open(clf_dir / 'train_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {clf_dir}")


if __name__ == '__main__':
    main()