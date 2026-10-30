#!/usr/bin/env python3
"""
scripts/reconstruction/compute_recon_metrics_agg.py

Aggregates per-file reconstruction metrics from metrics_merged.csv
into per-codec/bitrate/dataset means, saved to reconstruction_metrics_agg.csv.

Inputs:
    {results_dir}/reconstruction_quality/metrics_merged.csv

Outputs:
    {results_dir}/reconstruction_quality/reconstruction_metrics_agg.csv
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS

MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]
LEMUR_SUBSETS = ['lemur_s4a', 'lemur_swift1', 'lemur_swift2']
DAC_KBPS_MAP = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}


def main():
    results_dir = Path(PATHS['results_dir'])

    df = pd.read_csv(results_dir / 'reconstruction_quality' / 'metrics_merged.csv')
    print(f"Loaded {len(df)} rows")

    # -----------------------------------------------------------------------
    # FILTERS: keep only evaluated bitrates
    # -----------------------------------------------------------------------
    df = df[
        ~((df['codec'] == 'mp3')  & (~df['bitrate'].isin(MP3_KEEP))) &
        ~((df['codec'] == 'opus') & (~df['bitrate'].isin(OPUS_KEEP)))
    ]
    print(f"After bitrate filter: {len(df)} rows")

    # -----------------------------------------------------------------------
    # COMBINE LEMUR SUBSETS: pool all files across recorders
    # -----------------------------------------------------------------------
    lemur_df = df[df['dataset'].isin(LEMUR_SUBSETS)].copy()
    lemur_df['dataset'] = 'lemur'
    df = pd.concat([df, lemur_df], ignore_index=True)
    print(f"After lemur combine: {len(df)} rows")

    # -----------------------------------------------------------------------
    # NOMINAL KBPS: DAC uses n_q codebooks, convert to kbps
    # -----------------------------------------------------------------------
    df['kbps'] = df.apply(
        lambda r: DAC_KBPS_MAP[int(r['bitrate'])] if r['codec'] == 'dac'
                  else float(r['bitrate']), axis=1)

    # -----------------------------------------------------------------------
    # AGGREGATE: mean and std per dataset/codec/bitrate
    # -----------------------------------------------------------------------
    agg = (
        df.groupby(['dataset', 'codec', 'bitrate', 'kbps'])
        .agg(
            si_snr_mean = ('si_snr_whole', 'mean'),
            si_snr_std  = ('si_snr_whole', 'std'),
            stft_mean   = ('stft_whole',   'mean'),
            stft_std    = ('stft_whole',   'std'),
            n_files     = ('filename',     'count'),
        )
        .round(6)
        .reset_index()
        .sort_values(['dataset', 'codec', 'kbps'])
    )

    out = results_dir / 'reconstruction_quality' / 'reconstruction_metrics_agg.csv'
    agg.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Shape: {agg.shape}")
    print("\nSample:")
    print(agg[['dataset', 'codec', 'kbps', 'si_snr_mean', 'stft_mean', 'n_files']].to_string())


if __name__ == '__main__':
    main()