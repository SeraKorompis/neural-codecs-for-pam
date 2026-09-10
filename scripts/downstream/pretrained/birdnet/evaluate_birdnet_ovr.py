#!/usr/bin/env python3
"""
scripts/downstream/pretrained/birdnet/evaluate_birdnet_ovr.py
Evaluates pre-trained BirdNET on Northeastern US Soundscapes using the
same binary evaluation framework as the transfer-learned OVR classifiers,
enabling direct comparison.
Negative sampling: all non-positive test windows (OneVsRest), consistent
with the northeastern_ovr transfer-learned evaluation.
Saves to:
    results/downstream/pretrained/birdnet/
        macro_metrics_ovr.csv
        per_species_ovr.csv
Usage:
    python scripts/downstream/pretrained/birdnet/evaluate_birdnet_ovr.py
    python scripts/downstream/pretrained/birdnet/evaluate_birdnet_ovr.py \
        --detections_csv results/downstream/pretrained/birdnet/raw/detections_reconstructed_lowthresh.csv \
        --condition encodec1.5
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_curve)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import PATHS
RESULTS_BASE   = Path(PATHS['results_dir']) / 'downstream' / 'pretrained' / 'birdnet'
EMB_DIR        = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / \
                 'embeddings' / 'northeastern'
THRESHOLD_SCAN = np.arange(0.1, 0.91, 0.05)
def evaluate_species(y_true, y_scores):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    roc_auc = float(roc_auc_score(y_true, y_scores))
    ap      = float(average_precision_score(y_true, y_scores))
    def prf(thr):
        y_pred = (y_scores >= thr).astype(int)
        return (float(f1_score(y_true, y_pred, zero_division=0)),
                float(precision_score(y_true, y_pred, zero_division=0)),
                float(recall_score(y_true, y_pred, zero_division=0)))
    best_f1, best_thr_f1 = 0.0, 0.5
    for thr in THRESHOLD_SCAN:
        f1 = f1_score(y_true, (y_scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, thr
    fpr, tpr, thr_roc = roc_curve(y_true, y_scores)
    best_thr_roc        = float(thr_roc[np.argmin(np.sqrt((1-tpr)**2 + fpr**2))])
    f1_of1, pr_of1, re_of1 = prf(best_thr_f1)
    f1_or,  pr_or,  re_or  = prf(best_thr_roc)
    return {
        'roc_auc'              : roc_auc,
        'ap'                   : ap,
        'n_pos_test'           : int(y_true.sum()),
        'f1_optimal_f1'        : float(best_f1),
        'precision_optimal_f1' : pr_of1,
        'recall_optimal_f1'    : re_of1,
        'f1_optimal_roc'       : f1_or,
        'precision_optimal_roc': pr_or,
        'recall_optimal_roc'   : re_or,
    }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections_csv', default=None)
    parser.add_argument('--condition', default='original')
    parser.add_argument('--codec',    default=None)
    parser.add_argument('--bitrate',  default=None)
    parser.add_argument('--kbps',     default=None, type=float)
    parser.add_argument('--min_pos_test', default=10, type=int)
    args = parser.parse_args()
    if args.detections_csv is None:
        detections_path = RESULTS_BASE / 'raw' / 'detections_original_lowthresh.csv'
    else:
        detections_path = Path(args.detections_csv)
    if not detections_path.exists():
        print(f"ERROR: detections CSV not found: {detections_path}")
        sys.exit(1)
    print(f"{'='*60}")
    print(f"BirdNET OVR binary evaluation")
    print(f"  condition:    {args.condition}")
    print(f"  detections:   {detections_path.name}")
    print(f"  min_pos_test: {args.min_pos_test}")
    print(f"{'='*60}")
    # Load metadata
    meta = pd.read_csv(EMB_DIR / 'northeastern_all_metadata.csv').reset_index(drop=True)
    print(f"\n  Total windows: {len(meta):,}")
    window_lookup = {(row.filename, row.window_start): idx
                     for idx, row in meta.iterrows()}
    with open(EMB_DIR / 'northeastern_species_cols.json') as f:
        species = json.load(f)
    print(f"  Species: {len(species)}")
    # Load test indices
    with open(EMB_DIR / 'northeastern_positive_test_indices.json') as f:
        pos_test_idx = json.load(f)
    bg_test_idx = np.load(EMB_DIR / 'northeastern_background_test_indices.npy')
    # Build full test window set (all positive + background)
    all_pos_idx  = np.array(list({idx for idxs in pos_test_idx.values()
                                  for idx in idxs}), dtype=int)
    all_test_idx = np.unique(np.concatenate([all_pos_idx, bg_test_idx]))
    print(f"  Total test windows: {len(all_test_idx):,}")
    # Load and filter detections
    print(f"\nLoading detections...")
    det = pd.read_csv(detections_path)
    det = det[det['common_name'].isin(species)].copy()
    print(f"  After filtering to shared species: {len(det):,}")
    # Filter by codec/bitrate for reconstructed detections
    if args.codec is not None and args.codec != 'original':
        if 'codec' in det.columns and 'bitrate' in det.columns:
            det['bitrate_num'] = pd.to_numeric(det['bitrate'], errors='coerce')
            det = det[det['codec'].astype(str) == args.codec].copy()
            if args.bitrate is not None:
                det = det[np.isclose(det['bitrate_num'].fillna(-1),
                                     float(args.bitrate), atol=0.1)].copy()
            print(f"  After filtering to {args.codec} {args.bitrate}: {len(det):,}")
    det['window_idx'] = det.apply(
        lambda r: window_lookup.get((r['filename'], r['start_sec']), None),
        axis=1)
    n_unmatched = det['window_idx'].isna().sum()
    if n_unmatched > 0:
        print(f"  WARNING: {n_unmatched} detections unmatched")
    det = det.dropna(subset=['window_idx'])
    det['window_idx'] = det['window_idx'].astype(int)
    # Filter to test windows only
    test_window_set = set(all_test_idx.tolist())
    det_test        = det[det['window_idx'].isin(test_window_set)].copy()
    print(f"  Detections in test windows: {len(det_test):,}")
    # Evaluate per species — OneVsRest negatives
    per_species_rows = []
    skipped          = []
    idx_map          = {idx: i for i, idx in enumerate(all_test_idx)}
    for sp in species:
        pos_idx = np.array(pos_test_idx.get(sp, []), dtype=int)
        if len(pos_idx) == 0:
            skipped.append(sp)
            continue
        # OneVsRest: negatives = all non-positive test windows
        neg_idx    = np.setdiff1d(all_test_idx, pos_idx)
        sp_all_idx = np.concatenate([pos_idx, neg_idx]).astype(int)
        y_true     = np.array([1]*len(pos_idx) + [0]*len(neg_idx))
        # BirdNET scores — missing = 0.0
        sp_det    = det_test[
            (det_test['common_name'] == sp) &
            (det_test['window_idx'].isin(set(sp_all_idx.tolist())))
        ][['window_idx', 'confidence']]
        score_map = dict(zip(sp_det['window_idx'], sp_det['confidence']))
        y_scores  = np.array([score_map.get(idx, 0.0) for idx in sp_all_idx])
        metrics = evaluate_species(y_true, y_scores)
        if metrics is None:
            skipped.append(sp)
            continue
        print(f"  {sp:40s} ROC-AUC={metrics['roc_auc']:.3f}  "
              f"n_pos={len(pos_idx)}")
        per_species_rows.append({
            'species'  : sp,
            **metrics,
            'condition': args.condition,
            'codec'    : args.codec,
            'bitrate'  : args.bitrate,
            'kbps'     : args.kbps,
        })
    if skipped:
        print(f"\nSkipped {len(skipped)} species")
    if not per_species_rows:
        print("ERROR: no species evaluated")
        sys.exit(1)
    # Macro metrics
    macro_rows = [r for r in per_species_rows
                  if r['n_pos_test'] >= args.min_pos_test]
    print(f"\n  Species with >= {args.min_pos_test} test positives: "
          f"{len(macro_rows)} / {len(per_species_rows)}")
    macro_row = {
        'macro_roc_auc'        : float(np.mean([r['roc_auc'] for r in macro_rows])),
        'macro_ap'             : float(np.mean([r['ap'] for r in macro_rows])),
        'f1_optimal_f1'        : float(np.mean([r['f1_optimal_f1'] for r in macro_rows])),
        'precision_optimal_f1' : float(np.mean([r['precision_optimal_f1'] for r in macro_rows])),
        'recall_optimal_f1'    : float(np.mean([r['recall_optimal_f1'] for r in macro_rows])),
        'f1_optimal_roc'       : float(np.mean([r['f1_optimal_roc'] for r in macro_rows])),
        'precision_optimal_roc': float(np.mean([r['precision_optimal_roc'] for r in macro_rows])),
        'recall_optimal_roc'   : float(np.mean([r['recall_optimal_roc'] for r in macro_rows])),
        'n_species'            : len(macro_rows),
        'n_species_total'      : len(per_species_rows),
        'min_pos_test'         : args.min_pos_test,
        'condition'            : args.condition,
        'codec'                : args.codec,
        'bitrate'              : args.bitrate,
        'kbps'                 : args.kbps,
    }
    print(f"\n  Macro ROC-AUC: {macro_row['macro_roc_auc']:.4f} "
          f"(n={len(macro_rows)} species)")
    # Save
    out_dir = RESULTS_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    macro_path = out_dir / 'macro_metrics_ovr.csv'
    df_macro   = pd.DataFrame([macro_row])
    if macro_path.exists():
        df_macro.to_csv(macro_path, mode='a', header=False, index=False)
    else:
        df_macro.to_csv(macro_path, index=False)
    print(f"  Appended to {macro_path}")
    per_sp_path = out_dir / 'per_species_ovr.csv'
    per_sp_df   = pd.DataFrame(per_species_rows)
    if per_sp_path.exists():
        per_sp_df.to_csv(per_sp_path, mode='a', header=False, index=False)
    else:
        per_sp_df.to_csv(per_sp_path, index=False)
    print(f"  Appended to {per_sp_path}")
    print(f"\nDone.")
if __name__ == '__main__':
    main()
