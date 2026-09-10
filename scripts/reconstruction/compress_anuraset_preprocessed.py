#!/usr/bin/env python3
"""
scripts/reconstruction/compress_anuraset_preprocessed.py

Compresses AnuraSet preprocessed test clips using EnCodec and DAC.

This is a lean compression-only script — it does not compute reconstruction
metrics (SI-SNR, STFT) or call-region metrics, as these were already computed
on the raw AnuraSet recordings. The sole output is reconstructed WAV files
for each codec/bitrate combination, used to extract BirdNET embeddings for
downstream classifier evaluation.

Compression logic delegates entirely to src/compress.run_codec(), the same
function used by the full evaluation pipeline (src/evaluate.py), ensuring
identical codec behaviour.

Output structure mirrors the full evaluation pipeline:
    $RECON_ROOT/anuraset_preprocessed/{codec}/{bitrate}/{stem}_reconstructed.wav

Usage:
    python scripts/reconstruction/compress_anuraset_preprocessed.py --codec encodec --device cuda
    python scripts/reconstruction/compress_anuraset_preprocessed.py --codec dac --device cuda
"""
import argparse
import gc
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import PATHS

from src.compress import run_codec
from src.utils import load_audio

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# EPHEMERAL  = Path(os.environ.get('EPHEMERAL', str(PROJECT_ROOT)))
# AUDIO_DIR  = EPHEMERAL / 'anuraset_classifier' / 'data' / 'anuraset_preprocessed' / 'audio'
# META_PATH  = EPHEMERAL / 'anuraset_classifier' / 'data' / 'anuraset_preprocessed' / 'metadata.csv'
# RECON_ROOT = Path('/rds/general/project/bugg/live/seraphina_compression/reconstructed_audio')
# LOGS_DIR   = PROJECT_ROOT / 'logs'
# LOGS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR  = Path(PATHS['source_dir']) / 'anuraset' / 'preprocessed' / 'audio'
META_PATH  = Path(PATHS['source_dir']) / 'anuraset' / 'preprocessed' / 'metadata.csv'
RECON_ROOT = Path(PATHS['recon_dir'])
LOGS_DIR   = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CODEC CONFIG — matches evaluate.py DATASETS['anuraset']['codec_sr'] and 'bitrates'
# ---------------------------------------------------------------------------
CODEC_CONFIG = {
    'encodec': {'target_sr': 24000, 'bitrates': [1.5, 3.0, 6.0, 12.0, 24.0]},
    'dac':     {'target_sr': 44100, 'bitrates': [2, 3, 6, 9]},
    'mp3':     {'target_sr': 44100, 'bitrates': [8, 16, 24, 32, 64]},
    'opus':    {'target_sr': 44100, 'bitrates': [6, 8, 12, 14, 24, 32]},
}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(codec: str, device: str, bitrates: list = None):
    config    = CODEC_CONFIG[codec]
    target_sr = config['target_sr']
    br_list   = bitrates if bitrates is not None else config['bitrates']

    print(f"{'='*60}")
    print(f"AnuraSet preprocessed {args.split} clip compression")
    print(f"  codec:     {codec}")
    print(f"  target_sr: {target_sr} Hz")
    print(f"  bitrates:  {br_list}")
    print(f"  device:    {device}")
    print(f"  Started:   {datetime.now()}")
    print(f"{'='*60}")

    # Load metadata and filter to test subset
    meta      = pd.read_csv(META_PATH)
    test_meta = meta[meta['subset'] == args.split].reset_index(drop=True)
    print(f"\n  {args.split.capitalize()} clips: {len(test_meta):,}")

    # Build file list: (audio_path, stem) for each test clip
    file_list = []
    for _, row in test_meta.iterrows():
        fname = f"{row['fname']}_{row['min_t']}_{row['max_t']}.wav"
        path  = AUDIO_DIR / row['site'] / fname
        if path.exists():
            file_list.append((path, path.stem))
        else:
            print(f"  WARNING: missing file {path}")

    print(f"  Files found on disk: {len(file_list):,} / {len(test_meta):,}")
    if len(file_list) == 0:
        print("  ERROR: no files found, check AUDIO_DIR path")
        sys.exit(1)

    for bitrate in br_list:
        recon_dir = RECON_ROOT / 'anuraset' / 'preprocessed' / codec / str(bitrate)
        recon_dir.mkdir(parents=True, exist_ok=True)

        # Count already completed files for resumption
        completed = {p.stem.replace('_reconstructed', '')
                     for p in recon_dir.glob('*_reconstructed.wav')}
        todo = [(p, s) for p, s in file_list if s not in completed]

        print(f"\n  {codec} @ {bitrate}: "
              f"{len(completed)} already done, "
              f"{len(todo)} remaining")

        if not todo:
            print(f"  All files already compressed, skipping")
            continue

        n_errors = 0
        for audio_path, stem in tqdm(todo,
                                      desc=f"{codec}@{bitrate}",
                                      unit='file'):
            recon_path = recon_dir / f"{stem}_reconstructed.wav"

            try:
                audio, native_sr = load_audio(str(audio_path))

                recon, _, _, _ = run_codec(
                    audio     = audio,
                    native_sr = native_sr,
                    codec     = codec,
                    bitrate   = bitrate,
                    target_sr = target_sr,
                    device    = device,
                )

                # Save reconstructed audio
                import soundfile as sf
                sf.write(str(recon_path), recon, target_sr)

                del audio, recon
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"\n  CUDA OOM on {stem}, retrying on CPU...")
                torch.cuda.empty_cache()
                try:
                    audio, native_sr = load_audio(str(audio_path))
                    recon, _, _, _ = run_codec(
                        audio     = audio,
                        native_sr = native_sr,
                        codec     = codec,
                        bitrate   = bitrate,
                        target_sr = target_sr,
                        device    = 'cpu',
                    )
                    import soundfile as sf
                    sf.write(str(recon_path), recon, target_sr)
                    del audio, recon
                    gc.collect()
                except Exception as e:
                    print(f"\n  CPU retry failed for {stem}: {e}")
                    n_errors += 1

            except Exception as e:
                print(f"\n  Error on {stem}: {e}")
                n_errors += 1

        completed_now = len(list(recon_dir.glob('*_reconstructed.wav')))
        print(f"\n  {codec} @ {bitrate}: "
              f"{completed_now:,} files compressed, "
              f"{n_errors} errors")

    print(f"\n{'='*60}")
    print(f"Compression complete")
    print(f"Finished: {datetime.now()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compress AnuraSet preprocessed test clips')
    parser.add_argument('--split',   default='test',
                        choices=['train', 'test'],
                        help='Dataset split to compress')
    parser.add_argument('--codec',   required=True,
                        choices=['encodec', 'dac', 'mp3', 'opus'])
    parser.add_argument('--device',  default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--bitrates', default=None,
                        help='Comma-separated bitrates, e.g. "6.0,24.0"')
    args = parser.parse_args()

    bitrates = None
    if args.bitrates:
        bitrates = [float(b) if '.' in b else int(b)
                    for b in args.bitrates.split(',')]

    try:
        main(args.codec, args.device, bitrates)
    except Exception:
        tb = traceback.format_exc()
        print(f"\n{'!'*60}")
        print(f"FAILED: {tb}")
        print(f"{'!'*60}")
        error_log = LOGS_DIR / f"compress_anuraset_preprocessed_{args.codec}_error.log"
        with open(error_log, 'w') as f:
            f.write(tb)
        sys.exit(1)