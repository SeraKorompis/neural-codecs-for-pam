#!/usr/bin/env python3
"""
Number of species with a negative median (and mean) confidence shift for the
pre-trained BirdNET model on Northeastern Birds, using the same positives and
definition as plot_violin_with_spectrograms.py:
  - positives = annotated positive test windows of each species
  - shift = confidence on compressed audio - confidence on original audio
  - a window with no detection counts as confidence 0
Usage: python scripts/downstream/figures/confidence_shift_numbers.py [codec] [kbps]
"""
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import PATHS

codec = sys.argv[1] if len(sys.argv) > 1 else 'encodec'
kbps = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
MIN_WINDOWS = 10                 # positive windows per species
THRESHOLDS = [0.0, 0.05]         # minimum original confidence (0.05 = violin plot)

RES = Path(PATHS['results_dir']) / 'downstream'
RAW = RES / 'pretrained' / 'birdnet' / 'raw'
EMB = RES / 'transfer_learned' / 'embeddings' / 'northeastern'

meta = pd.read_csv(EMB / 'northeastern_all_metadata.csv')
meta['window_idx'] = range(len(meta))
pos_idx = json.load(open(EMB / 'northeastern_positive_test_indices.json'))
species_cols = json.load(open(EMB / 'northeastern_species_cols.json'))

pos = pd.DataFrame([(sp, i) for sp in species_cols for i in pos_idx.get(sp, [])],
                   columns=['common_name', 'window_idx'])


def window_conf(df):
    df = df.merge(meta[['filename', 'window_start', 'window_idx']],
                  left_on=['filename', 'start_sec'],
                  right_on=['filename', 'window_start'], how='inner')
    return df.groupby(['common_name', 'window_idx'], as_index=False)['confidence'].max()


orig = window_conf(pd.read_csv(RAW / 'detections_original_lowthresh.csv',
                               usecols=['filename', 'start_sec', 'common_name', 'confidence']))
parts = []
for ch in pd.read_csv(RAW / 'detections_reconstructed_lowthresh.csv',
                      usecols=['filename', 'codec', 'bitrate', 'start_sec',
                               'common_name', 'confidence'], chunksize=2_000_000):
    parts.append(ch[(ch.codec.astype(str) == codec) & (ch.bitrate.astype(float) == kbps)])
enc = window_conf(pd.concat(parts))

d = (pos.merge(orig.rename(columns={'confidence': 'conf_orig'}), on=['common_name', 'window_idx'], how='left')
        .merge(enc.rename(columns={'confidence': 'conf_enc'}), on=['common_name', 'window_idx'], how='left')
        .fillna({'conf_orig': 0.0, 'conf_enc': 0.0}))
d['shift'] = d['conf_enc'] - d['conf_orig']

print(f'{codec} {kbps} kbps; positive windows: {len(d):,}; species with positives: {d.common_name.nunique()}')
for t in THRESHOLDS:
    s = d[d.conf_orig >= t].groupby('common_name')['shift'].agg(['count', 'median', 'mean'])
    s = s[s['count'] >= MIN_WINDOWS]
    print(f'original confidence >= {t}: {len(s)} species with >= {MIN_WINDOWS} windows; '
          f'negative median shift: {(s["median"] < 0).sum()}; negative mean shift: {(s["mean"] < 0).sum()}')
