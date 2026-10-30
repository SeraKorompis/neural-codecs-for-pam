#!/usr/bin/env python3
"""
scripts/reconstruction/merge_metrics.py

Merges the per-(dataset, codec, bitrate) metrics.csv files written by
run_benchmark.py into one file, used by the aggregation and RTF scripts.

Input  (written by run_benchmark.py):
    {results_dir}/{mode}/{device}/metrics/{dataset}/{codec}/{bitrate}/metrics.csv
    (neural codecs run on cuda, conventional codecs on cpu, so both
     {device} folders are read)

Output:
    {results_dir}/reconstruction_quality/metrics_merged.csv

Usage:
    python scripts/reconstruction/merge_metrics.py            # full runs
    python scripts/reconstruction/merge_metrics.py --mode test
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS

# Number of annotated files per dataset in a full run
EXPECTED_FILES = {
    'anuraset': 1206,
    'northeastern_birds': 284,
    'lemur_s4a': 7,
    'lemur_swift1': 33,
    'lemur_swift2': 31,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='full', choices=['test', 'full'])
    args = parser.parse_args()

    results_dir = Path(PATHS['results_dir'])
    files = sorted((results_dir / args.mode).glob('*/metrics/*/*/*/metrics.csv'))
    if not files:
        print(f"ERROR: no metrics.csv files found under {results_dir / args.mode}")
        sys.exit(1)

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Read {len(files)} files, {len(df)} rows")

    # Duplicate (dataset, file, codec, bitrate) rows would double-count files
    key = ['dataset', 'filename', 'codec', 'bitrate']
    n_dup = int(df.duplicated(subset=key).sum())
    if n_dup:
        print(f"WARNING: {n_dup} duplicate rows for the same file/codec/bitrate "
              f"(e.g. the same condition run on both cuda and cpu)")

    # Completeness check against the expected number of files (full runs only)
    if args.mode == 'full':
        counts = df.groupby(['dataset', 'codec', 'bitrate']).size()
        incomplete = [(idx, n) for idx, n in counts.items()
                      if n != EXPECTED_FILES.get(idx[0], n)]
        for (dataset, codec, bitrate), n in incomplete:
            print(f"WARNING: {dataset}/{codec}/{bitrate}: {n} files, "
                  f"expected {EXPECTED_FILES[dataset]}")
        if not incomplete:
            print("All dataset/codec/bitrate combinations are complete")

    out = results_dir / 'reconstruction_quality' / 'metrics_merged.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(df.groupby(['dataset', 'codec']).size().to_string())


if __name__ == '__main__':
    main()
