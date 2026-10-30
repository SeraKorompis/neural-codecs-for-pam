#!/usr/bin/env python3
"""
scripts/reconstruction/compress_anuraset_preprocessed.py

Compresses the AnuraSet pre-processed 3-second clips with a chosen codec,
for the downstream classifier evaluation (BirdNET embeddings are extracted
from the output). The raw one-minute AnuraSet recordings are handled by
run_benchmark.py instead.

This is compression only: no reconstruction metrics are computed. Compression
is delegated to src/compress.run_codec(), the same function used by
src/evaluate.py, so codec behaviour is identical.

Output (reconstructed audio is saved at the codec sample rate):
    {recon_dir}/anuraset/preprocessed/{codec}/{bitrate}/{stem}_reconstructed.wav

Resumable: clips that already exist in the output folder are skipped.

Usage:
    python scripts/reconstruction/compress_anuraset_preprocessed.py \
        --codec encodec --split test --device cuda
    python scripts/reconstruction/compress_anuraset_preprocessed.py \
        --codec dac --split train --bitrates 6,9 --max_files 20   # quick test
"""
import argparse
import gc
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.compress import run_codec
from src.config import PATHS
from src.utils import load_audio

AUDIO_DIR  = Path(PATHS['source_dir']) / 'anuraset' / 'preprocessed' / 'audio'
META_PATH  = Path(PATHS['source_dir']) / 'anuraset' / 'preprocessed' / 'metadata.csv'
RECON_ROOT = Path(PATHS['recon_dir'])
LOGS_DIR   = PROJECT_ROOT / 'logs'

# Codec sample rates and bitrates used to produce the reported downstream
# results. Note: Opus clips here are processed and saved at 44100 Hz, whereas
# run_benchmark.py (raw recordings) uses 48000 Hz for Opus.
CODEC_CONFIG = {
    'encodec': {'target_sr': 24000, 'bitrates': [1.5, 3.0, 6.0, 12.0, 24.0]},
    'dac':     {'target_sr': 44100, 'bitrates': [2, 3, 6, 9]},
    'mp3':     {'target_sr': 44100, 'bitrates': [8, 16, 24]},
    'opus':    {'target_sr': 44100, 'bitrates': [8, 14, 24]},
}


def compress_clip(audio_path, recon_path, codec, bitrate, target_sr, device):
    """Compress one clip and save it. Falls back to CPU on a CUDA OOM."""
    audio, native_sr = load_audio(str(audio_path))
    try:
        recon, _, _, _ = run_codec(audio=audio, native_sr=native_sr, codec=codec,
                                   bitrate=bitrate, target_sr=target_sr,
                                   device=device)
    except torch.cuda.OutOfMemoryError:
        print(f"\n  CUDA OOM on {audio_path.stem}, retrying on CPU...")
        torch.cuda.empty_cache()
        recon, _, _, _ = run_codec(audio=audio, native_sr=native_sr, codec=codec,
                                   bitrate=bitrate, target_sr=target_sr,
                                   device='cpu')
    sf.write(str(recon_path), recon, target_sr)
    del audio, recon
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(codec: str, split: str, device: str,
         bitrates: list = None, max_files: int = None):
    config    = CODEC_CONFIG[codec]
    target_sr = config['target_sr']
    br_list   = bitrates if bitrates is not None else config['bitrates']

    print(f"{'='*60}")
    print(f"AnuraSet pre-processed {split} clip compression")
    print(f"  codec:     {codec}")
    print(f"  target_sr: {target_sr} Hz")
    print(f"  bitrates:  {br_list}")
    print(f"  device:    {device}")
    print(f"  max_files: {max_files}")
    print(f"  Started:   {datetime.now()}")
    print(f"{'='*60}")

    meta       = pd.read_csv(META_PATH)
    split_meta = meta[meta['subset'] == split].reset_index(drop=True)
    print(f"\n  {split.capitalize()} clips: {len(split_meta):,}")

    # Build the file list: (audio_path, stem) for each clip in the split
    file_list = []
    for _, row in split_meta.iterrows():
        fname = f"{row['fname']}_{row['min_t']}_{row['max_t']}.wav"
        path  = AUDIO_DIR / row['site'] / fname
        if path.exists():
            file_list.append((path, path.stem))
        else:
            print(f"  WARNING: missing file {path}")

    print(f"  Files found on disk: {len(file_list):,} / {len(split_meta):,}")
    if not file_list:
        print("  ERROR: no files found, check source_dir in config.yaml")
        sys.exit(1)

    if max_files is not None:
        file_list = file_list[:max_files]
        print(f"  Limited to the first {len(file_list)} clips (--max_files)")

    for bitrate in br_list:
        recon_dir = RECON_ROOT / 'anuraset' / 'preprocessed' / codec / str(bitrate)
        recon_dir.mkdir(parents=True, exist_ok=True)

        # Resume: skip clips that already exist
        completed = {p.stem.replace('_reconstructed', '')
                     for p in recon_dir.glob('*_reconstructed.wav')}
        todo = [(p, s) for p, s in file_list if s not in completed]
        print(f"\n  {codec} @ {bitrate}: "
              f"{len(file_list) - len(todo)} already done, {len(todo)} remaining")

        n_errors = 0
        for audio_path, stem in tqdm(todo, desc=f"{codec}@{bitrate}", unit='file'):
            try:
                compress_clip(audio_path, recon_dir / f"{stem}_reconstructed.wav",
                              codec, bitrate, target_sr, device)
            except Exception as e:
                print(f"\n  Error on {stem}: {e}")
                n_errors += 1

        n_present = sum(1 for _, s in file_list
                        if (recon_dir / f"{s}_reconstructed.wav").exists())
        print(f"\n  {codec} @ {bitrate}: {n_present:,}/{len(file_list):,} clips "
              f"present, {n_errors} errors")
        if n_present == 0:
            raise RuntimeError(f"No clips produced for {codec} @ {bitrate}")

    print(f"\n{'='*60}")
    print(f"Compression complete")
    print(f"Finished: {datetime.now()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compress AnuraSet pre-processed clips')
    parser.add_argument('--split',     default='test', choices=['train', 'test'],
                        help='Dataset split to compress')
    parser.add_argument('--codec',     required=True,
                        choices=['encodec', 'dac', 'mp3', 'opus'])
    parser.add_argument('--device',    default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--bitrates',  default=None,
                        help='Comma-separated bitrates, e.g. "6.0,24.0"')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Only process the first N clips (for testing)')
    args = parser.parse_args()

    bitrates = None
    if args.bitrates:
        bitrates = [float(b) if '.' in b else int(b)
                    for b in args.bitrates.split(',')]

    try:
        main(args.codec, args.split, args.device, bitrates, args.max_files)
    except Exception:
        tb = traceback.format_exc()
        print(f"\n{'!'*60}\nFAILED: {tb}\n{'!'*60}")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        error_log = LOGS_DIR / f"compress_anuraset_preprocessed_{args.codec}_error.log"
        error_log.write_text(tb)
        sys.exit(1)