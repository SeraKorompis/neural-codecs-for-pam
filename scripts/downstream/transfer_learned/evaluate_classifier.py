#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/evaluate_classifier.py
Evaluates transfer-learned BirdNET classifiers on original and compressed
audio embeddings.
Supports two evaluation scenarios:
    1. Trained on original, evaluated on compressed (default)
       --dataset anuraset --source encodec --bitrate 6.0
    2. Domain-matched: trained on compressed, evaluated on same compressed
       --dataset lemur --source dac --bitrate 9 \
       --train_source dac --train_bitrate 9
Supported datasets: anuraset, lemur, lemur_temporal,
                    northeastern, northeastern_ovr
Saves to:
    On original classifier:
        results/downstream/transfer_learned/on_original/{dataset}/
    Domain-matched classifier:
        results/downstream/transfer_learned/on_compressed/{dataset}/
    northeastern_ovr per-condition files (to avoid race conditions):
        results/downstream/transfer_learned/on_original/northeastern/raw/per_condition/
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score, precision_score, recall_score,
                             roc_curve)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS
# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS_BASE = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned'
DAC_KBPS = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}
DATASET_CONFIG = {
    'anuraset': {
        'emb_dir'    : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'anuraset',
        'test_lbl'   : 'anuraset_test_labels.npy',
        'species'    : 'anuraset_species_cols.json',
        'multilabel' : True,
        'ovr'        : False,
        'temporal'   : False,
    },
    'lemur': {
        'emb_dir'    : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'lemur',
        'test_lbl'   : 'lemur_test_labels.npy',
        'species'    : None,
        'multilabel' : False,
        'ovr'        : False,
        'temporal'   : False,
    },
    'lemur_temporal': {
        'emb_dir'    : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'lemur',
        'all_emb'    : 'lemur_all_original_embeddings.npy',
        'all_lbl'    : 'lemur_all_labels.npy',
        'test_idx'   : 'lemur_temporal_test_indices.npy',
        'test_lbl'   : None,
        'species'    : None,
        'multilabel' : False,
        'ovr'        : False,
        'temporal'   : True,
    },
    'northeastern': {
        'emb_dir'    : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'northeastern',
        'test_lbl'   : 'northeastern_test_labels.npy',
        'species'    : 'northeastern_species_cols.json',
        'multilabel' : True,
        'ovr'        : False,
        'temporal'   : False,
    },
    'northeastern_ovr': {
        'emb_dir'    : Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'northeastern',
        'test_lbl'   : None,
        'species'    : 'northeastern_species_cols.json',
        'multilabel' : True,
        'ovr'        : True,
        'temporal'   : False,
    },
}
THRESHOLD_SCAN = np.arange(0.1, 0.91, 0.05)
# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def get_emb_path(emb_dir, dataset, source, bitrate):
    if dataset == 'northeastern_ovr':
        if source == 'original':
            return emb_dir / 'northeastern_all_original_embeddings_fixed.npy'
        return emb_dir / f'northeastern_all_{source}{bitrate}_embeddings.npy'
    if dataset == 'lemur_temporal':
        if source == 'original':
            return emb_dir / 'lemur_all_original_embeddings.npy'
        return emb_dir / f'lemur_all_{source}{bitrate}_embeddings.npy'
    base = dataset
    if source == 'original':
        return emb_dir / f'{base}_test_original_embeddings.npy'
    return emb_dir / f'{base}_test_{source}{bitrate}_embeddings.npy'

def get_kbps(source, bitrate):
    if source == 'original':
        return None
    if source == 'dac':
        return DAC_KBPS.get(int(bitrate), float(bitrate))
    return float(bitrate)

def get_clf_dir(dataset, train_source, train_bitrate):
    if dataset == 'northeastern_ovr':
        if train_source == 'original':
            return RESULTS_BASE / 'on_original' / 'northeastern' / \
                   'classifiers' / 'binary_classifiers_ovr_original'
        tag = f'{train_source}{train_bitrate}'
        return RESULTS_BASE / 'on_compressed' / 'northeastern' / \
               'classifiers' / f'binary_classifiers_ovr_{tag}'
    if dataset == 'lemur_temporal':
        if train_source == 'original':
            return RESULTS_BASE / 'on_original' / 'lemur' / \
                   'classifiers' / 'classifier_temporal'
        tag = f'{train_source}{train_bitrate}'
        return RESULTS_BASE / 'on_compressed' / 'lemur' / \
               'classifiers' / f'classifier_temporal_{tag}'
    if train_source == 'original':
        return RESULTS_BASE / 'on_original' / dataset / 'classifiers'
    train_tag = f'{train_source}{train_bitrate}'
    return RESULTS_BASE / 'on_compressed' / dataset / 'classifiers' / \
           f'classifier_{train_tag}'

def get_results_dir(dataset, train_source, train_bitrate):
    results_dataset = {
        'northeastern_ovr' : 'northeastern',
        'lemur_temporal'   : 'lemur',
    }.get(dataset, dataset)
    if train_source == 'original':
        return RESULTS_BASE / 'on_original' / results_dataset
    return RESULTS_BASE / 'on_compressed' / results_dataset

def get_metrics_filename(dataset):
    if dataset == 'northeastern_ovr':
        return 'macro_metrics_ovr.csv'
    if dataset == 'lemur_temporal':
        return 'macro_metrics_temporal.csv'
    return 'macro_metrics.csv'

def get_per_species_filename(dataset):
    if dataset == 'northeastern_ovr':
        return 'per_species_ovr.csv'
    if dataset == 'lemur_temporal':
        return 'per_species_temporal.csv'
    return 'per_species.csv'

def append_csv(path: Path, row: dict):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

# ---------------------------------------------------------------------------
# MULTILABEL EVALUATION (AnuraSet, Northeastern)
# ---------------------------------------------------------------------------
def evaluate_multilabel(y_true, y_scores, fixed_thr_f1, fixed_thr_roc,
                        species=None, min_pos_test=10):
    ap_per_species = average_precision_score(y_true, y_scores, average=None)
    if min_pos_test > 0:
        valid = [i for i in range(y_true.shape[1])
                 if y_true[:, i].sum() >= min_pos_test]
    else:
        valid = [i for i in range(y_true.shape[1])
                 if 0 < y_true[:, i].sum() < len(y_true)]
    macro_map     = float(ap_per_species[valid].mean())
    macro_roc_auc = float(roc_auc_score(
        y_true[:, valid], y_scores[:, valid], average='macro'))
    def prf(y_t, y_p):
        return (float(f1_score(y_t, y_p, average='macro', zero_division=0)),
                float(precision_score(y_t, y_p, average='macro', zero_division=0)),
                float(recall_score(y_t, y_p, average='macro', zero_division=0)))
    f1_ff1,  pr_ff1,  re_ff1  = prf(y_true, (y_scores >= fixed_thr_f1).astype(int))
    f1_froc, pr_froc, re_froc = prf(y_true, (y_scores >= fixed_thr_roc).astype(int))
    opt_thr_f1 = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        best_f1, best_thr = 0.0, 0.5
        for thr in THRESHOLD_SCAN:
            f1 = f1_score(y_true[:, i], (y_scores[:, i] >= thr).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        opt_thr_f1[i] = best_thr
    f1_of1, pr_of1, re_of1 = prf(y_true, (y_scores >= opt_thr_f1).astype(int))
    mean_shift_f1 = float(np.nanmean(opt_thr_f1 - fixed_thr_f1))
    opt_thr_roc = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        if 0 < y_true[:, i].sum() < len(y_true):
            fpr, tpr, thr_roc = roc_curve(y_true[:, i], y_scores[:, i])
            opt_thr_roc[i] = thr_roc[np.argmin(np.sqrt((1-tpr)**2 + fpr**2))]
        else:
            opt_thr_roc[i] = opt_thr_f1[i]
    f1_oroc, pr_oroc, re_oroc = prf(y_true, (y_scores >= opt_thr_roc).astype(int))
    mean_shift_roc = float(np.nanmean(opt_thr_roc - fixed_thr_roc))
    macro_row = {
        'macro_roc_auc'            : macro_roc_auc,
        'macro_map'                : macro_map,
        'f1_fixed_f1'              : f1_ff1,
        'precision_fixed_f1'       : pr_ff1,
        'recall_fixed_f1'          : re_ff1,
        'f1_fixed_roc'             : f1_froc,
        'precision_fixed_roc'      : pr_froc,
        'recall_fixed_roc'         : re_froc,
        'f1_optimal_f1'            : f1_of1,
        'precision_optimal_f1'     : pr_of1,
        'recall_optimal_f1'        : re_of1,
        'mean_threshold_shift_f1'  : mean_shift_f1,
        'f1_optimal_roc'           : f1_oroc,
        'precision_optimal_roc'    : pr_oroc,
        'recall_optimal_roc'       : re_oroc,
        'mean_threshold_shift_roc' : mean_shift_roc,
    }
    per_species_rows = []
    if species is not None:
        for i, sp in enumerate(species):
            sp_true   = y_true[:, i]
            sp_scores = y_scores[:, i]
            ap        = float(ap_per_species[i])
            sp_roc_auc = float(roc_auc_score(sp_true, sp_scores)) \
                if 0 < sp_true.sum() < len(sp_true) else float('nan')
            per_species_rows.append({
                'species'              : sp,
                'ap'                   : ap,
                'roc_auc'              : sp_roc_auc,
                'n_pos_test'           : int(sp_true.sum()),
                'f1_fixed_f1'          : float(f1_score(sp_true, (sp_scores >= fixed_thr_f1[i]).astype(int),  zero_division=0)),
                'precision_fixed_f1'   : float(precision_score(sp_true, (sp_scores >= fixed_thr_f1[i]).astype(int),  zero_division=0)),
                'recall_fixed_f1'      : float(recall_score(sp_true, (sp_scores >= fixed_thr_f1[i]).astype(int),  zero_division=0)),
                'f1_fixed_roc'         : float(f1_score(sp_true, (sp_scores >= fixed_thr_roc[i]).astype(int), zero_division=0)),
                'precision_fixed_roc'  : float(precision_score(sp_true, (sp_scores >= fixed_thr_roc[i]).astype(int), zero_division=0)),
                'recall_fixed_roc'     : float(recall_score(sp_true, (sp_scores >= fixed_thr_roc[i]).astype(int), zero_division=0)),
                'f1_optimal_f1'        : float(f1_score(sp_true, (sp_scores >= opt_thr_f1[i]).astype(int),  zero_division=0)),
                'precision_optimal_f1' : float(precision_score(sp_true, (sp_scores >= opt_thr_f1[i]).astype(int),  zero_division=0)),
                'recall_optimal_f1'    : float(recall_score(sp_true, (sp_scores >= opt_thr_f1[i]).astype(int),  zero_division=0)),
                'f1_optimal_roc'       : float(f1_score(sp_true, (sp_scores >= opt_thr_roc[i]).astype(int), zero_division=0)),
                'precision_optimal_roc': float(precision_score(sp_true, (sp_scores >= opt_thr_roc[i]).astype(int), zero_division=0)),
                'recall_optimal_roc'   : float(recall_score(sp_true, (sp_scores >= opt_thr_roc[i]).astype(int), zero_division=0)),
            })
    return macro_row, per_species_rows

# ---------------------------------------------------------------------------
# BINARY EVALUATION (Lemur, Lemur temporal)
# ---------------------------------------------------------------------------
def evaluate_binary(y_true, y_scores, fixed_thr_f1, fixed_thr_roc):
    roc_auc   = float(roc_auc_score(y_true, y_scores))
    map_score = float(average_precision_score(y_true, y_scores))
    def prf_bin(y_t, y_p):
        return (float(f1_score(y_t, y_p, zero_division=0)),
                float(precision_score(y_t, y_p, zero_division=0)),
                float(recall_score(y_t, y_p, zero_division=0)))
    f1_ff1,  pr_ff1,  re_ff1  = prf_bin(y_true, (y_scores >= fixed_thr_f1).astype(int))
    f1_froc, pr_froc, re_froc = prf_bin(y_true, (y_scores >= fixed_thr_roc).astype(int))
    best_f1, best_thr_f1 = 0.0, float(fixed_thr_f1)
    for thr in THRESHOLD_SCAN:
        f1 = f1_score(y_true, (y_scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, thr
    f1_of1, pr_of1, re_of1 = prf_bin(y_true, (y_scores >= best_thr_f1).astype(int))
    fpr, tpr, thr_roc = roc_curve(y_true, y_scores)
    best_thr_roc        = float(thr_roc[np.argmin(np.sqrt((1-tpr)**2 + fpr**2))])
    f1_oroc, pr_oroc, re_oroc = prf_bin(y_true, (y_scores >= best_thr_roc).astype(int))
    macro_row = {
        'roc_auc'              : roc_auc,
        'map'                  : map_score,
        'f1_fixed_f1'          : f1_ff1,
        'precision_fixed_f1'   : pr_ff1,
        'recall_fixed_f1'      : re_ff1,
        'fixed_threshold_f1'   : float(fixed_thr_f1),
        'f1_fixed_roc'         : f1_froc,
        'precision_fixed_roc'  : pr_froc,
        'recall_fixed_roc'     : re_froc,
        'fixed_threshold_roc'  : float(fixed_thr_roc),
        'f1_optimal_f1'        : float(best_f1),
        'precision_optimal_f1' : pr_of1,
        'recall_optimal_f1'    : re_of1,
        'optimal_threshold_f1' : float(best_thr_f1),
        'threshold_shift_f1'   : float(best_thr_f1 - fixed_thr_f1),
        'f1_optimal_roc'       : f1_oroc,
        'precision_optimal_roc': pr_oroc,
        'recall_optimal_roc'   : re_oroc,
        'optimal_threshold_roc': best_thr_roc,
        'threshold_shift_roc'  : float(best_thr_roc - fixed_thr_roc),
    }
    return macro_row, []

# ---------------------------------------------------------------------------
# NORTHEASTERN OVR EVALUATION
# ---------------------------------------------------------------------------
def evaluate_northeastern_ovr(embeddings, clf, scaler, fixed_thr_f1,
                               fixed_thr_roc, emb_dir, species,
                               min_pos_test, condition, source,
                               bitrate, kbps, results_dir, raw_dir,
                               train_source_tag):
    with open(emb_dir / 'northeastern_positive_test_indices.json') as f:
        pos_test_idx = json.load(f)
    bg_test_idx = np.load(emb_dir / 'northeastern_background_test_indices.npy')
    all_pos_idx  = np.array(list({idx for idxs in pos_test_idx.values()
                                  for idx in idxs}), dtype=int)
    all_test_idx = np.unique(np.concatenate([all_pos_idx, bg_test_idx]))
    print(f"  Total test windows: {len(all_test_idx):,}")

    X_test   = scaler.transform(embeddings[all_test_idx])
    y_scores = clf.predict_proba(X_test)
    idx_map  = {idx: i for i, idx in enumerate(all_test_idx)}

    # Save raw scores
    scores_df = pd.DataFrame(y_scores, columns=species)
    scores_df.insert(0, 'clip_idx', all_test_idx)
    scores_df['condition']     = condition
    scores_df['codec']         = source
    scores_df['bitrate']       = bitrate
    scores_df['kbps']          = kbps
    scores_df['train_source']  = train_source_tag
    scores_df['train_bitrate'] = bitrate
    scores_df.to_csv(
        raw_dir / f'scores_{train_source_tag}_on_{condition}.csv',
        index=False)
    print(f"  Scores saved")

    per_species_rows = []
    for sp_idx, sp in enumerate(species):
        pos_idx = np.array(pos_test_idx.get(sp, []), dtype=int)
        if len(pos_idx) == 0:
            continue
        neg_idx    = np.setdiff1d(all_test_idx, pos_idx)
        sp_all_idx = np.concatenate([pos_idx, neg_idx]).astype(int)
        local_idx  = np.array([idx_map[i] for i in sp_all_idx if i in idx_map])
        y_true     = np.array([1]*len(pos_idx) + [0]*len(neg_idx))
        sp_scores  = y_scores[local_idx, sp_idx]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        thr_f1  = float(fixed_thr_f1[sp_idx])  if sp_idx < len(fixed_thr_f1)  else 0.5
        thr_roc = float(fixed_thr_roc[sp_idx]) if sp_idx < len(fixed_thr_roc) else 0.5
        sp_roc_auc = float(roc_auc_score(y_true, sp_scores))
        ap         = float(average_precision_score(y_true, sp_scores))
        best_f1, best_thr_f1 = 0.0, 0.5
        for thr in THRESHOLD_SCAN:
            f1 = f1_score(y_true, (sp_scores >= thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr_f1 = f1, thr
        fpr, tpr, thr_roc_arr = roc_curve(y_true, sp_scores)
        best_thr_roc = float(thr_roc_arr[np.argmin(np.sqrt((1-tpr)**2 + fpr**2))])
        def prf(thr):
            y_pred = (sp_scores >= thr).astype(int)
            return (float(f1_score(y_true, y_pred, zero_division=0)),
                    float(precision_score(y_true, y_pred, zero_division=0)),
                    float(recall_score(y_true, y_pred, zero_division=0)))
        n_pos_test = int(y_true.sum())
        print(f"  {sp:40s} ROC-AUC={sp_roc_auc:.3f}  n_pos={n_pos_test}")
        per_species_rows.append({
            'species'              : sp,
            'roc_auc'              : sp_roc_auc,
            'ap'                   : ap,
            'n_pos_test'           : n_pos_test,
            'f1_fixed_f1'          : prf(thr_f1)[0],
            'precision_fixed_f1'   : prf(thr_f1)[1],
            'recall_fixed_f1'      : prf(thr_f1)[2],
            'f1_fixed_roc'         : prf(thr_roc)[0],
            'precision_fixed_roc'  : prf(thr_roc)[1],
            'recall_fixed_roc'     : prf(thr_roc)[2],
            'f1_optimal_f1'        : float(best_f1),
            'precision_optimal_f1' : prf(best_thr_f1)[1],
            'recall_optimal_f1'    : prf(best_thr_f1)[2],
            'f1_optimal_roc'       : prf(best_thr_roc)[0],
            'precision_optimal_roc': prf(best_thr_roc)[1],
            'recall_optimal_roc'   : prf(best_thr_roc)[2],
            'condition'            : condition,
            'codec'                : source,
            'bitrate'              : bitrate,
            'kbps'                 : kbps,
        })

    macro_rows = [r for r in per_species_rows if r['n_pos_test'] >= min_pos_test]
    print(f"\n  Species with >= {min_pos_test} test positives: "
          f"{len(macro_rows)} / {len(per_species_rows)}")

    macro_row = {
        'macro_roc_auc'        : float(np.mean([r['roc_auc'] for r in macro_rows])),
        'macro_ap'             : float(np.mean([r['ap'] for r in macro_rows])),
        'f1_fixed_f1'          : float(np.mean([r['f1_fixed_f1'] for r in macro_rows])),
        'precision_fixed_f1'   : float(np.mean([r['precision_fixed_f1'] for r in macro_rows])),
        'recall_fixed_f1'      : float(np.mean([r['recall_fixed_f1'] for r in macro_rows])),
        'f1_optimal_f1'        : float(np.mean([r['f1_optimal_f1'] for r in macro_rows])),
        'precision_optimal_f1' : float(np.mean([r['precision_optimal_f1'] for r in macro_rows])),
        'recall_optimal_f1'    : float(np.mean([r['recall_optimal_f1'] for r in macro_rows])),
        'f1_optimal_roc'       : float(np.mean([r['f1_optimal_roc'] for r in macro_rows])),
        'precision_optimal_roc': float(np.mean([r['precision_optimal_roc'] for r in macro_rows])),
        'recall_optimal_roc'   : float(np.mean([r['recall_optimal_roc'] for r in macro_rows])),
        'n_species'            : len(macro_rows),
        'condition'            : condition,
        'codec'                : source,
        'bitrate'              : bitrate,
        'kbps'                 : kbps,
    }
    print(f"  Macro ROC-AUC: {macro_row['macro_roc_auc']:.4f}")

    # Save per-condition files to avoid race condition with parallel jobs
    per_cond_dir = results_dir / 'raw' / 'per_condition'
    per_cond_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([macro_row]).to_csv(
        per_cond_dir / f'macro_metrics_ovr_{condition}.csv', index=False)
    print(f"  Saved to per_condition/macro_metrics_ovr_{condition}.csv")

    pd.DataFrame(per_species_rows).to_csv(
        per_cond_dir / f'per_species_ovr_{condition}.csv', index=False)
    print(f"  Saved to per_condition/per_species_ovr_{condition}.csv")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate transfer-learned classifier on compressed embeddings')
    parser.add_argument('--dataset', required=True,
                        choices=['anuraset', 'lemur', 'lemur_temporal',
                                 'northeastern', 'northeastern_ovr'])
    parser.add_argument('--source', required=True,
                        choices=['original', 'encodec', 'dac', 'mp3', 'opus'])
    parser.add_argument('--bitrate', default=None)
    parser.add_argument('--train_source', default='original',
                        choices=['original', 'encodec', 'dac', 'mp3', 'opus'])
    parser.add_argument('--train_bitrate', default=None)
    parser.add_argument('--min_pos_test', default=10, type=int)
    args = parser.parse_args()

    if args.source != 'original' and args.bitrate is None:
        parser.error('--bitrate required when --source is not original')
    if args.train_source != 'original' and args.train_bitrate is None:
        parser.error('--train_bitrate required when --train_source is not original')

    bitrate = None
    if args.bitrate is not None:
        bitrate = float(args.bitrate) if '.' in args.bitrate else int(args.bitrate)
    train_bitrate = None
    if args.train_bitrate is not None:
        train_bitrate = float(args.train_bitrate) if '.' in args.train_bitrate \
                        else int(args.train_bitrate)

    source_tag       = 'original' if args.source == 'original' else \
                       f'{args.source}{bitrate}'
    train_source_tag = 'original' if args.train_source == 'original' else \
                       f'{args.train_source}{train_bitrate}'
    condition = source_tag
    kbps      = get_kbps(args.source, bitrate)

    clf_dir     = get_clf_dir(args.dataset, args.train_source, train_bitrate)
    results_dir = get_results_dir(args.dataset, args.train_source, train_bitrate)
    raw_dir     = results_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Evaluating: {args.dataset} | test={condition} | "
          f"trained_on={train_source_tag}")
    print(f"  clf_dir:     {clf_dir}")
    print(f"  results_dir: {results_dir}")
    print(f"{'='*60}")

    if not clf_dir.exists():
        print(f"ERROR: classifier not found: {clf_dir}")
        sys.exit(1)

    config  = DATASET_CONFIG[args.dataset]
    emb_dir = config['emb_dir']

    clf           = joblib.load(clf_dir / 'classifier.joblib')
    scaler        = joblib.load(clf_dir / 'scaler.joblib')
    fixed_thr_f1  = np.load(clf_dir / 'optimal_thresholds_f1.npy')
    fixed_thr_roc = np.load(clf_dir / 'optimal_thresholds_roc.npy')

    species = None
    if config['species'] is not None:
        with open(emb_dir / config['species']) as f:
            species = json.load(f)

    # -----------------------------------------------------------------------
    # NORTHEASTERN OVR
    # -----------------------------------------------------------------------
    if config['ovr']:
        emb_path = get_emb_path(emb_dir, args.dataset, args.source, bitrate)
        if not emb_path.exists():
            print(f"ERROR: embeddings not found: {emb_path}")
            sys.exit(1)
        print(f"\nLoading embeddings: {emb_path.name}")
        embeddings = np.load(emb_path)
        print(f"  Shape: {embeddings.shape}")
        evaluate_northeastern_ovr(
            embeddings, clf, scaler, fixed_thr_f1, fixed_thr_roc,
            emb_dir, species, args.min_pos_test,
            condition, args.source, bitrate, kbps,
            results_dir, raw_dir, train_source_tag)
        return

    # -----------------------------------------------------------------------
    # LEMUR TEMPORAL
    # -----------------------------------------------------------------------
    if config['temporal']:
        emb_path = get_emb_path(emb_dir, args.dataset, args.source, bitrate)
        if not emb_path.exists():
            print(f"ERROR: embeddings not found: {emb_path}")
            sys.exit(1)
        print(f"\nLoading embeddings: {emb_path.name}")
        all_embeddings = np.load(emb_path)
        all_labels     = np.load(emb_dir / config['all_lbl'])
        test_idx       = np.load(emb_dir / config['test_idx'])
        X_test = all_embeddings[test_idx]
        y_test = all_labels[test_idx]
        print(f"  Embeddings (all): {all_embeddings.shape}")
        print(f"  Test subset:      {X_test.shape}, labels: {y_test.shape}")
        X_test_s = scaler.transform(X_test)
        y_scores = clf.predict_proba(X_test_s)
        if y_scores.ndim == 2:
            y_scores = y_scores[:, 1]
        macro_row, _ = evaluate_binary(
            y_test, y_scores,
            float(fixed_thr_f1[0]), float(fixed_thr_roc[0]))
        print(f"  ROC-AUC:     {macro_row['roc_auc']:.4f}")
        print(f"  mAP:         {macro_row['map']:.4f}")
        print(f"  F1 (opt F1): {macro_row['f1_optimal_f1']:.4f}")
        pd.DataFrame({
            'clip_idx'     : np.arange(len(y_scores)),
            'score'        : y_scores,
            'label'        : y_test,
            'condition'    : condition,
            'codec'        : args.source,
            'bitrate'      : bitrate,
            'kbps'         : kbps,
            'train_source' : args.train_source,
            'train_bitrate': train_bitrate,
        }).to_csv(raw_dir / f'scores_{train_source_tag}_on_{condition}.csv',
                  index=False)
        print(f"  Scores saved")
        macro_row['condition'] = condition
        macro_row['codec']     = args.source
        macro_row['bitrate']   = bitrate
        macro_row['kbps']      = kbps
        append_csv(results_dir / 'macro_metrics_temporal.csv', macro_row)
        print(f"  Appended to macro_metrics_temporal.csv")
        return

    # -----------------------------------------------------------------------
    # STANDARD EVALUATION (AnuraSet, Lemur, Northeastern)
    # -----------------------------------------------------------------------
    emb_path = get_emb_path(emb_dir, args.dataset, args.source, bitrate)
    if not emb_path.exists():
        print(f"ERROR: embeddings not found: {emb_path}")
        sys.exit(1)
    X_test = np.load(emb_path)
    y_test = np.load(emb_dir / config['test_lbl'])
    print(f"  Embeddings: {X_test.shape}")
    print(f"  Labels:     {y_test.shape}")
    X_test_s = scaler.transform(X_test)
    y_scores  = clf.predict_proba(X_test_s)

    if config['multilabel']:
        macro_row, per_species_rows = evaluate_multilabel(
            y_test, y_scores, fixed_thr_f1, fixed_thr_roc, species,
            min_pos_test=args.min_pos_test)
        print(f"  Macro ROC-AUC: {macro_row['macro_roc_auc']:.4f}")
        print(f"  Macro mAP:     {macro_row['macro_map']:.4f}")
        print(f"  F1 (opt F1):   {macro_row['f1_optimal_f1']:.4f}")
        scores_df = pd.DataFrame(y_scores, columns=species)
        scores_df.insert(0, 'clip_idx', np.arange(len(y_scores)))
        scores_df['condition']     = condition
        scores_df['codec']         = args.source
        scores_df['bitrate']       = bitrate
        scores_df['kbps']          = kbps
        scores_df['train_source']  = args.train_source
        scores_df['train_bitrate'] = train_bitrate
        scores_df.to_csv(
            raw_dir / f'scores_{train_source_tag}_on_{condition}.csv',
            index=False)
        print(f"  Scores saved")
    else:
        if y_scores.ndim == 2:
            y_scores = y_scores[:, 1]
        macro_row, per_species_rows = evaluate_binary(
            y_test, y_scores,
            float(fixed_thr_f1[0]), float(fixed_thr_roc[0]))
        print(f"  ROC-AUC:     {macro_row['roc_auc']:.4f}")
        print(f"  mAP:         {macro_row['map']:.4f}")
        print(f"  F1 (opt F1): {macro_row['f1_optimal_f1']:.4f}")
        pd.DataFrame({
            'clip_idx'     : np.arange(len(y_scores)),
            'score'        : y_scores,
            'label'        : y_test,
            'condition'    : condition,
            'codec'        : args.source,
            'bitrate'      : bitrate,
            'kbps'         : kbps,
            'train_source' : args.train_source,
            'train_bitrate': train_bitrate,
        }).to_csv(
            raw_dir / f'scores_{train_source_tag}_on_{condition}.csv',
            index=False)
        print(f"  Scores saved")

    macro_row['condition'] = condition
    macro_row['codec']     = args.source
    macro_row['bitrate']   = bitrate
    macro_row['kbps']      = kbps
    append_csv(results_dir / get_metrics_filename(args.dataset), macro_row)
    print(f"  Appended to {get_metrics_filename(args.dataset)}")

    if per_species_rows:
        per_sp_df = pd.DataFrame(per_species_rows)
        per_sp_df['condition']     = condition
        per_sp_df['codec']         = args.source
        per_sp_df['bitrate']       = bitrate
        per_sp_df['kbps']          = kbps
        per_sp_df['train_source']  = args.train_source
        per_sp_df['train_bitrate'] = train_bitrate
        sp_path = results_dir / get_per_species_filename(args.dataset)
        if sp_path.exists():
            per_sp_df.to_csv(sp_path, mode='a', header=False, index=False)
        else:
            per_sp_df.to_csv(sp_path, index=False)
        print(f"  Appended to {get_per_species_filename(args.dataset)}")

if __name__ == '__main__':
    main()