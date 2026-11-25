#!/usr/bin/env python3
"""
Delta AUC = macro ROC-AUC of classifiers fine-tuned on compressed audio minus
that of classifiers fine-tuned on raw audio, both tested on the same compressed
condition. Writes two files to results/downstream/transfer_learned/:
  fine_tuning_delta_auc_all_conditions.csv  every dataset, codec and bitrate
  fine_tuning_delta_auc_low_high.csv        lowest and highest bitrate per codec
                                            (same MP3/Opus subsets as the figure)
Usage: python scripts/downstream/figures/delta_auc_numbers.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fine_tuning_delta import (DATASET_CONFIG, PATHS, MP3_KEEP, OPUS_KEEP,
                                    normalise_kbps)

OUT_DIR = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned'

rows = []
for key, cfg in DATASET_CONFIG.items():
    if not cfg['orig_path'].exists() or not cfg['comp_path'].exists():
        print(f'WARNING: missing file for {key}')
        continue
    raw = normalise_kbps(pd.read_csv(cfg['orig_path']))
    comp = normalise_kbps(pd.read_csv(cfg['comp_path']))
    raw_auc = dict(zip(raw['condition'], raw[cfg['roc_col_orig']]))
    for _, r in comp.iterrows():
        cond, codec = r['condition'], r.get('codec')
        if cond in raw_auc and codec not in ('original', None):
            a_raw, a_comp = float(raw_auc[cond]), float(r[cfg['roc_col_comp']])
            rows.append(dict(dataset=key, codec=codec, kbps=r['kbps'],
                             condition=cond,
                             auc_finetuned_raw=a_raw,
                             auc_finetuned_compressed=a_comp,
                             delta_auc=a_comp - a_raw))

allc = pd.DataFrame(rows).sort_values(['dataset', 'codec', 'kbps'])
allc.to_csv(OUT_DIR / 'fine_tuning_delta_auc_all_conditions.csv', index=False)

# lowest and highest bitrate per dataset and codec, same subsets as the figure
sel = allc[~((allc.codec == 'mp3') & ~allc.kbps.isin(MP3_KEEP))]
sel = sel[~((sel.codec == 'opus') & ~sel.kbps.isin(OPUS_KEEP))]
lh = []
for (ds, codec), g in sel.groupby(['dataset', 'codec']):
    g = g.sort_values('kbps')
    lh.append(g.iloc[[0]].assign(position='lowest'))
    if len(g) > 1:
        lh.append(g.iloc[[-1]].assign(position='highest'))
lh = pd.concat(lh, ignore_index=True)
lh.to_csv(OUT_DIR / 'fine_tuning_delta_auc_low_high.csv', index=False)

low, high = lh[lh.position == 'lowest'], lh[lh.position == 'highest']
print(lh.round(4).to_string(index=False))
print()
print(f"Lowest bitrate:  delta AUC {low.delta_auc.min():.3f} to {low.delta_auc.max():.3f}")
print(f"Highest bitrate: delta AUC {high.delta_auc.min():.3f} to {high.delta_auc.max():.3f}")
m = low.merge(high, on=['dataset', 'codec'], suffixes=('_low', '_high'))
print(f"Larger gain at lowest than at highest: {(m.delta_auc_low > m.delta_auc_high).sum()} of {len(m)} cases")
print()
print('All conditions: number with delta <= 0 (no improvement)')
for ds, g in allc.groupby('dataset'):
    bad = g[g.delta_auc <= 0]
    print(f'  {ds}: {len(bad)} of {len(g)}')
    for _, r in bad.iterrows():
        print(f'    {r.codec} {r.kbps:g} kbps: {r.delta_auc:.5f}')
print(f'\nSaved to {OUT_DIR}')
