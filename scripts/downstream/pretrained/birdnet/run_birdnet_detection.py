#!/usr/bin/env python3
"""
scripts/downstream/pretrained/birdnet/run_birdnet_detection.py
Runs BirdNET on original and reconstructed Northeastern Birds audio
at near-zero confidence threshold (0.01) to capture continuous scores
needed for threshold-invariant AUC evaluation.
Saves to:
    results/downstream/pretrained/birdnet/raw/detections_original_lowthresh.csv
    results/downstream/pretrained/birdnet/raw/detections_reconstructed_lowthresh.csv
    results/downstream/pretrained/birdnet/raw/week_species_cache.csv
Usage:
    python scripts/downstream/pretrained/birdnet/run_birdnet_detection.py
    python scripts/downstream/pretrained/birdnet/run_birdnet_detection.py \
        --reconstructed_only --codecs mp3 --bitrates 8,16,24
"""
import sys
import re
import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import PATHS
from src.birdnet_eval import (
    NORTHEASTERN_LAT,
    NORTHEASTERN_LON,
    _date_from_filename,
    _date_to_birdnet_week,
    _build_week_species_cache,
    _parse_species_name,
    run_birdnet_on_file,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_DIR   = Path(PATHS['source_dir']) / 'northeastern_us_soundscapes' / 'soundscape_data'
RECON_DIR  = Path(PATHS['recon_dir']) / 'northeastern_birds'
OUTPUT_DIR = Path(PATHS['results_dir']) / 'downstream' / 'pretrained' / 'birdnet' / 'raw'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONF        = 0.01
WEEK_CACHE_PATH = OUTPUT_DIR / 'week_species_cache.csv'


def save_week_cache(week_cache: dict, path: Path) -> None:
    rows = []
    for week, species_set in week_cache.items():
        for species in sorted(species_set):
            rows.append({'week': week, 'species_name': species})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved week cache: {len(week_cache)} weeks, {len(rows)} entries -> {path}")


def load_week_cache(path: Path) -> dict:
    df = pd.read_csv(path)
    cache = {}
    for week, group in df.groupby('week'):
        cache[int(week)] = set(group['species_name'].tolist())
    return cache


def run_original_lowthresh(audio_files, acoustic_model, geo_model,
                            week_cache, output_csv):
    completed = set()
    if output_csv.exists():
        existing  = pd.read_csv(output_csv)
        completed = set(existing['filename'].unique())
        print(f"Found {len(completed)} already-processed files, skipping.")

    for audio_path in tqdm(audio_files, desc="Original (low thresh)", unit="file"):
        filename = audio_path.name
        if filename in completed:
            continue
        try:
            df = run_birdnet_on_file(
                audio_path     = audio_path,
                acoustic_model = acoustic_model,
                geo_model      = geo_model,
                lat            = NORTHEASTERN_LAT,
                lon            = NORTHEASTERN_LON,
                min_conf       = MIN_CONF,
                top_k          = None,
                week_cache     = week_cache,
            )
            write_header = not output_csv.exists()
            df.to_csv(output_csv, mode='a', header=write_header, index=False)
            completed.add(filename)
            tqdm.write(f"  {filename}: {len(df)} detections")
        except Exception as e:
            tqdm.write(f"  Skipping {filename}: {type(e).__name__}: {e}")
    print(f"Done. Saved to {output_csv}")


def run_reconstructed_lowthresh(audio_files, acoustic_model, geo_model,
                                 week_cache, output_csv,
                                 filter_codecs=None, filter_bitrates=None):
    completed = set()
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        for _, row in existing.iterrows():
            completed.add((row['filename'], row['codec'], float(row['bitrate'])))
        print(f"Found {len(completed)} already-processed combinations, skipping.")

    # Discover codec/bitrate subdirectories with optional filtering
    codec_bitrate_dirs = []
    for codec_dir in sorted(RECON_DIR.iterdir()):
        if not codec_dir.is_dir():
            continue
        codec = codec_dir.name
        if filter_codecs and codec not in filter_codecs:
            continue
        for bitrate_dir in sorted(codec_dir.iterdir()):
            if not bitrate_dir.is_dir():
                continue
            try:
                bitrate = float(bitrate_dir.name)
            except ValueError:
                continue
            if filter_bitrates and bitrate not in filter_bitrates:
                continue
            codec_bitrate_dirs.append((codec, bitrate, bitrate_dir))

    print(f"Found {len(codec_bitrate_dirs)} codec/bitrate combinations:")
    for codec, bitrate, _ in codec_bitrate_dirs:
        print(f"  {codec} @ {bitrate}")

    original_files = [f.name for f in audio_files]
    total = len(original_files) * len(codec_bitrate_dirs)

    with tqdm(total=total, desc="Reconstructed (low thresh)", unit="file") as pbar:
        for codec, bitrate, bitrate_dir in codec_bitrate_dirs:
            for filename in original_files:
                pbar.update(1)
                key = (filename, codec, bitrate)
                if key in completed:
                    continue
                stem       = Path(filename).stem
                recon_path = bitrate_dir / f"{stem}_reconstructed.wav"
                if not recon_path.exists():
                    continue
                try:
                    import soundfile as sf
                    date        = _date_from_filename(filename)
                    week        = _date_to_birdnet_week(date) if date else None
                    species_set = week_cache.get(week, set()) if week else set()

                    with sf.SoundFile(str(recon_path)) as f:
                        f.read(frames=1000)

                    predictions = acoustic_model.predict(
                        str(recon_path),
                        custom_species_list          = species_set,
                        default_confidence_threshold = MIN_CONF,
                        top_k                        = None,
                    )
                    df = predictions.to_dataframe()
                    if df.empty:
                        df = pd.DataFrame(columns=[
                            'filename', 'codec', 'bitrate',
                            'start_sec', 'end_sec',
                            'scientific_name', 'common_name', 'confidence'
                        ])
                    else:
                        parsed = df['species_name'].apply(_parse_species_name)
                        df['scientific_name'] = [p[0] for p in parsed]
                        df['common_name']     = [p[1] for p in parsed]
                        df['start_sec']       = df['start_time'].astype(float).round(2)
                        df['end_sec']         = df['end_time'].astype(float).round(2)
                        df['confidence']      = df['confidence'].astype(float).round(4)
                        df['filename']        = filename
                        df['codec']           = codec
                        df['bitrate']         = bitrate
                        df = df[['filename', 'codec', 'bitrate',
                                 'start_sec', 'end_sec',
                                 'scientific_name', 'common_name', 'confidence']]

                    write_header = not output_csv.exists()
                    df.to_csv(output_csv, mode='a', header=write_header, index=False)
                    completed.add(key)
                except Exception as e:
                    tqdm.write(f"  Skipping {filename} {codec}@{bitrate}: "
                               f"{type(e).__name__}: {e}")
    print(f"Done. Saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_only',      action='store_true',
                        help='Only run original audio detection')
    parser.add_argument('--reconstructed_only', action='store_true',
                        help='Only run reconstructed audio detection')
    parser.add_argument('--codecs',   type=str, default=None,
                        help='Comma-separated codecs to filter, e.g. mp3')
    parser.add_argument('--bitrates', type=str, default=None,
                        help='Comma-separated bitrates to filter, e.g. 8,16,24')
    args = parser.parse_args()

    filter_codecs   = args.codecs.split(',')                        if args.codecs   else None
    filter_bitrates = [float(b) for b in args.bitrates.split(',')]  if args.bitrates else None

    import birdnet
    print("Loading BirdNET models...")
    acoustic_model = birdnet.load("acoustic", "2.4", "tf")
    geo_model      = birdnet.load("geo",      "2.4", "tf")
    print("Models loaded.")

    audio_files = sorted(DATA_DIR.glob("*.wav"))
    print(f"Found {len(audio_files)} WAV files.")

    n_files = int(os.environ.get('N_FILES', 0))
    if n_files > 0:
        audio_files = audio_files[:n_files]
        print(f"TEST MODE: limiting to first {n_files} files")

    # Build or load week cache
    if WEEK_CACHE_PATH.exists():
        print("Loading cached week/species list...")
        week_cache = load_week_cache(WEEK_CACHE_PATH)
    else:
        print("Building geo model week cache...")
        weeks = []
        for audio_path in audio_files:
            date = _date_from_filename(audio_path.name)
            if date:
                weeks.append(_date_to_birdnet_week(date))
        week_cache = _build_week_species_cache(geo_model, weeks,
                                               lat=NORTHEASTERN_LAT,
                                               lon=NORTHEASTERN_LON)
        save_week_cache(week_cache, WEEK_CACHE_PATH)

    if not args.reconstructed_only:
        print("\n--- Step 1: Original audio (low threshold) ---")
        run_original_lowthresh(
            audio_files    = audio_files,
            acoustic_model = acoustic_model,
            geo_model      = geo_model,
            week_cache     = week_cache,
            output_csv     = OUTPUT_DIR / 'detections_original_lowthresh.csv',
        )

    if not args.original_only:
        print("\n--- Step 2: Reconstructed audio (low threshold) ---")
        run_reconstructed_lowthresh(
            audio_files     = audio_files,
            acoustic_model  = acoustic_model,
            geo_model       = geo_model,
            week_cache      = week_cache,
            output_csv      = OUTPUT_DIR / 'detections_reconstructed_lowthresh.csv',
            filter_codecs   = filter_codecs,
            filter_bitrates = filter_bitrates,
        )

if __name__ == '__main__':
    main()