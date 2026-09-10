#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/train_classifier.py
Unified classifier training script for downstream evaluation of
BirdNET embeddings across PAM datasets.
Trains a StandardScaler + OneVsRestClassifier(LogisticRegression) on
BirdNET embeddings. Supports training on original or compressed audio
embeddings to evaluate domain-matched training as a mitigation strategy.
Supported datasets:
    anuraset          -- multilabel, 42 anuran species, clip-level
    lemur             -- binary, roar vs no-roar, window-level
    lemur_temporal    -- binary, roar vs no-roar, temporal train/test split
    northeastern      -- multilabel, 81 bird species, window-level (all species)
    northeastern_ovr  -- multilabel, per-species OneVsRest with background
                          negatives (all non-positive train windows),
                          evaluated on the train set
Saves to:
    on_original:   results/downstream/transfer_learned/on_original/{dataset}/classifiers/
    on_compressed: results/downstream/transfer_learned/on_compressed/{dataset}/classifiers/
Usage:
    # Train on original audio
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset anuraset
    # Train on original audio (Northeastern, shared species only)
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset northeastern_shared
    # Train on compressed audio (domain-matched)
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur --train_source encodec --train_bitrate 6.0
    # Train the Northeastern OneVsRest classifier (background negatives)
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset northeastern_ovr --train_source original
"""
import argparse
import json
import os
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

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS_BASE = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned'

DATASET_CONFIG = {
    'anuraset': {
        'emb_dir'   : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'anuraset',
        'train_emb' : 'anuraset_train_original_embeddings.npy',
        'train_lbl' : 'anuraset_train_labels.npy',
        'test_emb'  : 'anuraset_test_original_embeddings.npy',
        'test_lbl'  : 'anuraset_test_labels.npy',
        'species'   : 'anuraset_species_cols.json',
        'multilabel': True,
    },
    'lemur': {
        'emb_dir'   : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'lemur',
        'train_emb' : 'lemur_train_original_embeddings.npy',
        'train_lbl' : 'lemur_train_labels.npy',
        'test_emb'  : 'lemur_test_original_embeddings.npy',
        'test_lbl'  : 'lemur_test_labels.npy',
        'species'   : None,
        'multilabel': False,
    },
    'northeastern': {
        'emb_dir'   : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'northeastern',
        'train_emb' : 'northeastern_train_original_embeddings.npy',
        'train_lbl' : 'northeastern_train_labels.npy',
        'test_emb'  : 'northeastern_test_original_embeddings.npy',
        'test_lbl'  : 'northeastern_test_labels.npy',
        'species'   : 'northeastern_species_cols.json',
        'multilabel': True,
    },
    'lemur_temporal': {
        'emb_dir'   : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'lemur',
        'train_emb' : 'lemur_all_original_embeddings.npy',
        'train_lbl' : 'lemur_all_labels.npy',
        'test_emb'  : 'lemur_all_original_embeddings.npy',
        'test_lbl'  : 'lemur_all_labels.npy',
        'species'   : None,
        'multilabel': False,
        'train_idx' : 'lemur_temporal_train_indices.npy',
        'test_idx'  : 'lemur_temporal_test_indices.npy',
    },
    # 'northeastern_ovr' reuses the same embedding directory as 'northeastern'
    # but is trained/evaluated via a fully separate code path (see
    # run_northeastern_ovr below) using positive/background index sets
    # rather than pre-split train/test label arrays.
    'northeastern_ovr': {
        'emb_dir'   : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'northeastern',
        'species'   : 'northeastern_species_cols.json',
        'multilabel': True,
    },
}

LOGREG_PARAMS = dict(max_iter=1000, C=1.0, solver='lbfgs')

def resolve_source_tag(train_source, train_bitrate):
    """Return (source_tag, train_bitrate_parsed) from CLI args."""
    if train_source == 'original':
        return 'original', None
    train_bitrate = float(train_bitrate) if '.' in train_bitrate else int(train_bitrate)
    return f'{train_source}{train_bitrate}', train_bitrate


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------
def evaluate_multilabel(y_true, y_scores, species=None):
    ap_per_species = average_precision_score(
        y_true, y_scores, average=None)
    macro_map = float(ap_per_species.mean())
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_thresholds_f1  = np.zeros(y_true.shape[1])
    best_thresholds_roc = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        best_f1, best_thr_f1 = 0.0, 0.5
        for thr in thresholds:
            preds = (y_scores[:, i] >= thr).astype(int)
            f1    = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr_f1 = f1, thr
        best_thresholds_f1[i] = best_thr_f1
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            fpr, tpr, thr_roc = roc_curve(y_true[:, i], y_scores[:, i])
            dist = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
            best_thresholds_roc[i] = thr_roc[np.argmin(dist)]
        else:
            best_thresholds_roc[i] = best_thr_f1
    best_thresholds = best_thresholds_roc
    y_pred   = (y_scores >= best_thresholds).astype(int)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    results = {
        'macro_map': macro_map,
        'macro_f1' : macro_f1,
        'micro_f1' : micro_f1,
    }
    if species is not None:
        results['per_species'] = {
            sp: {'ap': float(ap), 'threshold': float(thr)}
            for sp, ap, thr in zip(species, ap_per_species, best_thresholds)
        }
    return results, best_thresholds_f1, best_thresholds_roc


def evaluate_binary(y_true, y_scores):
    from sklearn.metrics import precision_score, recall_score
    roc_auc    = float(roc_auc_score(y_true, y_scores))
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_f1, best_thr_f1 = 0.0, 0.5
    for thr in thresholds:
        preds = (y_scores >= thr).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, thr
    fpr, tpr, thr_roc = roc_curve(y_true, y_scores)
    dist = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    best_thr_roc = float(thr_roc[np.argmin(dist)])
    best_thr     = best_thr_roc
    y_pred       = (y_scores >= best_thr).astype(int)
    precision    = float(precision_score(y_true, y_pred, zero_division=0))
    recall       = float(recall_score(y_true, y_pred, zero_division=0))
    return {
        'roc_auc'               : roc_auc,
        'f1'                    : float(best_f1),
        'precision'             : precision,
        'recall'                : recall,
        'optimal_threshold'     : best_thr_roc,
        'optimal_threshold_roc' : best_thr_roc,
        'optimal_threshold_f1'  : float(best_thr_f1),
    }, np.array([best_thr_f1]), np.array([best_thr_roc])


# ---------------------------------------------------------------------------
# STANDARD TRAIN/TEST PIPELINE
# (anuraset, lemur, lemur_temporal)
# ---------------------------------------------------------------------------
def run_standard(args):
    results_dataset = 'lemur' if args.dataset == 'lemur_temporal' \
                  else args.dataset
    config  = DATASET_CONFIG[args.dataset]
    emb_dir = config['emb_dir']

    source_tag, train_bitrate = resolve_source_tag(args.train_source, args.train_bitrate)

    # Resolve classifier output directory
    if args.train_source == 'original':
        clf_dir = RESULTS_BASE / 'on_original' / results_dataset / 'classifiers' / \
          ('classifier_temporal' if args.dataset == 'lemur_temporal' else
           'classifier')
    else:
        clf_dir = RESULTS_BASE / 'on_compressed' / results_dataset / 'classifiers' / \
                    (f'classifier_temporal_{source_tag}' if args.dataset == 'lemur_temporal'
                    else f'classifier_{source_tag}')
    clf_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training classifier: {args.dataset}")
    print(f"  train source: {source_tag}")
    print(f"  clf_dir:      {clf_dir}")
    print(f"{'='*60}")

    # Resolve train embedding path
    if args.train_source == 'original':
        train_emb_path = emb_dir / config['train_emb']
    else:
        if args.dataset == 'lemur_temporal':
            train_emb_path = emb_dir / f'lemur_all_{source_tag}_embeddings.npy'
        else:
            train_emb_path = emb_dir / \
                f'{results_dataset}_train_{source_tag}_embeddings.npy'

    if not train_emb_path.exists():
        print(f'ERROR: train embeddings not found: {train_emb_path}')
        sys.exit(1)

    print(f"\nLoading embeddings...")
    X_all_train = np.load(train_emb_path)
    y_all_train = np.load(emb_dir / config['train_lbl'])
    X_all_test  = np.load(emb_dir / config['test_emb'])
    y_all_test  = np.load(emb_dir / config['test_lbl'])

    # Apply index subsetting if specified (e.g. lemur_temporal)
    if 'train_idx' in config:
        train_idx = np.load(emb_dir / config['train_idx'])
        test_idx  = np.load(emb_dir / config['test_idx'])
        X_train = X_all_train[train_idx]
        y_train = y_all_train[train_idx]
        X_test  = X_all_test[test_idx]
        y_test  = y_all_test[test_idx]
        print(f"  Applied temporal split indices")
    else:
        X_train = X_all_train
        y_train = y_all_train
        X_test  = X_all_test
        y_test  = y_all_test

    print(f"  Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"  Test:  {X_test.shape},  labels: {y_test.shape}")

    species = None
    if config['species'] is not None:
        with open(emb_dir / config['species']) as f:
            species = json.load(f)
        print(f"  Species: {len(species)}")

    print(f"\nFitting scaler...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\nTraining classifier...")
    clf = OneVsRestClassifier(
        LogisticRegression(**LOGREG_PARAMS),
        n_jobs=-1)
    clf.fit(X_train, y_train)
    print("  Done.")

    print(f"\nEvaluating on original test embeddings...")
    y_scores = clf.predict_proba(X_test)
    if config['multilabel']:
        results, best_thresholds_f1, best_thresholds_roc = \
            evaluate_multilabel(y_test, y_scores, species)
        print(f"  Macro mAP: {results['macro_map']:.4f}")
        print(f"  Macro F1:  {results['macro_f1']:.4f}")
        print(f"  Micro F1:  {results['micro_f1']:.4f}")
    else:
        if y_scores.ndim == 2:
            y_scores = y_scores[:, 1]
        results, best_thresholds_f1, best_thresholds_roc = \
            evaluate_binary(y_test, y_scores)
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"  F1:        {results['f1']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  Threshold: {results['optimal_threshold']:.2f}")

    # Save classifier, scaler, thresholds
    joblib.dump(clf,    clf_dir / 'classifier.joblib')
    joblib.dump(scaler, clf_dir / 'scaler.joblib')
    np.save(clf_dir / 'optimal_thresholds.npy',     best_thresholds_roc)
    np.save(clf_dir / 'optimal_thresholds_roc.npy', best_thresholds_roc)
    np.save(clf_dir / 'optimal_thresholds_f1.npy',  best_thresholds_f1)

    results['condition']    = source_tag
    results['dataset']      = args.dataset
    results['train_source'] = source_tag
    with open(clf_dir / 'results_original.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {clf_dir}")


# ---------------------------------------------------------------------------
# NORTHEASTERN ONE-VS-REST PIPELINE (not target species: background + other species positives, train-set eval)
# ---------------------------------------------------------------------------
def run_northeastern_ovr(args):
    emb_dir = DATASET_CONFIG['northeastern_ovr']['emb_dir']
    source_tag, train_bitrate = resolve_source_tag(args.train_source, args.train_bitrate)

    if args.train_source == 'original':
        clf_dir = RESULTS_BASE / 'on_original' / 'northeastern' / \
                     'classifiers' / f'binary_classifiers_ovr_balanced_{source_tag}'
    else:
        clf_dir = RESULTS_BASE / 'on_compressed' / 'northeastern' / \
                     'classifiers' / f'binary_classifiers_ovr_balanced_{source_tag}'
    clf_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Training OneVsRestClassifier: northeastern")
    print(f"  train source: {source_tag}")
    print(f"  clf_dir:      {clf_dir}")
    print(f"{'='*60}")

    # Load embeddings
    if args.train_source == 'original':
        emb_path = emb_dir / 'northeastern_all_original_embeddings_fixed.npy'
    else:
        emb_path = emb_dir / f'northeastern_all_{source_tag}_embeddings.npy'

    if not emb_path.exists():
        print(f"ERROR: embeddings not found: {emb_path}")
        sys.exit(1)

    print(f"\nLoading embeddings: {emb_path.name}")
    embeddings = np.load(emb_path)
    print(f"  Shape: {embeddings.shape}")

    # Load indices and species
    with open(emb_dir / 'northeastern_positive_train_indices.json') as f:
        pos_train_idx = json.load(f)
    bg_train_idx = np.load(emb_dir / 'northeastern_background_train_indices.npy')
    with open(emb_dir / 'northeastern_species_cols.json') as f:
        species = json.load(f)

    # Build full train index set
    all_pos_idx   = np.array(list({idx for idxs in pos_train_idx.values()
                                   for idx in idxs}), dtype=int)
    # Train set = all positive train windows + background
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
        # If idx is in the positive train indices for this species, set the corresponding
        #  row in y_train to 1, else leave it as 0 (background + other species positives)
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
    LOGREG_PARAMS_OVR = dict(max_iter=1000, C=1.0, solver='lbfgs', class_weight='balanced')
    clf = OneVsRestClassifier(
        LogisticRegression(**LOGREG_PARAMS_OVR),
        n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"  Done.")

    # Save classifier
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


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train downstream classifier on BirdNET embeddings')
    parser.add_argument('--dataset', required=True,
                        choices=['anuraset', 'lemur', 'northeastern',
                                 'lemur_temporal',
                                 'northeastern_ovr'],)
    parser.add_argument('--train_source', default='original',
                        choices=['original', 'encodec', 'dac', 'mp3', 'opus'],
                        help='Audio source for train embeddings')
    parser.add_argument('--train_bitrate', default=None,
                        help='Codec bitrate for train embeddings e.g. 6.0 or 9')
    args = parser.parse_args()

    if args.train_source != 'original' and args.train_bitrate is None:
        parser.error('--train_bitrate required when --train_source is not original')

    if args.dataset == 'northeastern_ovr':
        run_northeastern_ovr(args)
    else:
        run_standard(args)


if __name__ == '__main__':
    main()