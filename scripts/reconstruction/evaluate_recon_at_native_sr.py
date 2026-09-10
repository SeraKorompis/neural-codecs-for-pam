#!/usr/bin/env python3
"""
scripts/reconstruction/evaluate_recon_at_native_sr.py

Evaluates existing reconstructed audio files at native dataset SR.
Loads reconstructed WAVs already saved in reconstructed_audio/ (at native SR
for EnCodec/DAC, will be at native SR after resampling for MP3/Opus).
Computes SI-SNR and STFT distance at native SR for fair cross-codec comparison.

Saves to:
    results/reconstruction_quality/raw/{dataset}/{codec}/{bitrate}/metrics_native_sr.csv
    results/reconstruction_quality/raw/{dataset}/{codec}/{bitrate}/call_metrics_native_sr.csv

Usage:
    python scripts/reconstruction/evaluate_recon_at_native_sr.py \
        --dataset anuraset --codec encodec
    python scripts/reconstruction/evaluate_recon_at_native_sr.py \
        --dataset northeastern_birds --codec dac
"""
import argparse
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS
from src.utils import (
    load_audio, resample, si_snr, stft_distance,
    load_anuraset_labels, load_northeastern_labels, load_lemur_labels,
    compute_call_region_metrics, append_result, append_call_metrics
)

RECON_ROOT   = Path(PATHS['recon_dir'])
RESULTS_ROOT = Path(PATHS['results_dir']) / 'reconstruction_quality' / 'raw'

DATASET_CONFIG = {
    'anuraset': {
        'recon_subdir' : 'anuraset/raw',
        'audio_dir'    : Path(PATHS['source_dir']) / 'anuraset' / 'raw' / 'raw_data',
        'label_dir'    : Path(PATHS['source_dir']) / 'anuraset' / 'raw' / 'strong_labels',
        'label_fn'     : load_anuraset_labels,
        'label_ext'    : '.txt',
        'audio_ext'    : '.wav',
        'audio_glob'   : '**/*.wav',
    },
    'northeastern_birds': {
        'recon_subdir' : 'northeastern_birds',
        'audio_dir'    : Path(PATHS['source_dir']) / 'northeastern_us_soundscapes' / 'soundscape_data',
        'label_dir'    : Path(PATHS['source_dir']) / 'northeastern_us_soundscapes',
        'label_fn'     : load_northeastern_labels,
        'label_ext'    : '.csv',
        'audio_ext'    : '.wav',
        'annotation_ext': '.flac',
        'audio_glob'   : '*.wav',
    },
    'lemur_s4a': {
        'recon_subdir' : 'lemur/lemur_s4a',
        'audio_dir'    : Path(PATHS['source_dir']) / 'black_and_white_ruffed_lemur' / 'Audio1',
        'label_dir'    : Path(PATHS['source_dir']) / 'black_and_white_ruffed_lemur' / 'Annotations1',
        'label_fn'     : load_lemur_labels,
        'label_ext'    : '.svl',
        'audio_ext'    : '.wav',
        'audio_glob'   : 'S4A*.wav',
    },
    'lemur_swift1': {
        'recon_subdir' : 'lemur/lemur_swift1',
        'audio_dir'    : Path(PATHS['source_dir']) / 'black_and_white_ruffed_lemur' / 'Audio1',
        'label_dir'    : Path(PATHS['source_dir']) / 'black_and_white_ruffed_lemur' / 'Annotations1',
        'label_fn'     : load_lemur_labels,
        'label_ext'    : '.svl',
        'audio_ext'    : '.wav',
        'audio_glob'   : 'swift1*.wav',
    },
    'lemur_swift2': {
        'recon_subdir' : 'lemur/lemur_swift2',
        'audio_dir'    : Path(PATHS['source_dir']) / 'black_and_white_ruffed_lemur' / 'Audio3',
        'label_dir'    : Path(PATHS['source_dir']) / 'black_and_white_ruffed_lemur' / 'Annotations3',
        'label_fn'     : load_lemur_labels,
        'label_ext'    : '.svl',
        'audio_ext'    : '.wav',
        'audio_glob'   : 'swift2*.wav',
    },
}


def get_bitrates(dataset, codec):
    """Get available bitrate directories for a dataset/codec combination."""
    recon_subdir = DATASET_CONFIG[dataset]['recon_subdir']
    codec_dir    = RECON_ROOT / recon_subdir / codec
    if not codec_dir.exists():
        return []
    return sorted([d.name for d in codec_dir.iterdir() if d.is_dir()])


def evaluate_pair(orig_path, recon_path, label_path, dataset, codec, bitrate, config):
    """Evaluate one original/reconstructed pair at native SR."""
    annotation_ext = config.get('annotation_ext', config['audio_ext'])
    annotation_filename = orig_path.stem + annotation_ext
    label_fn = config['label_fn']

    try:
        annotations = label_fn(str(label_path), filename=annotation_filename)
    except Exception as e:
        print(f"  WARNING: could not load labels for {orig_path.name}: {e}")
        return None, None

    if len(annotations) == 0:
        return None, None

    try:
        audio, native_sr = load_audio(str(orig_path))
    except Exception as e:
        print(f"  WARNING: could not load {orig_path.name}: {e}")
        return None, None

    try:
        recon_info = sf.info(str(recon_path))
        recon_sr   = recon_info.samplerate
        recon, _   = load_audio(str(recon_path))
    except Exception as e:
        print(f"  WARNING: could not load {recon_path.name}: {e}")
        return None, None

    # Resample recon to native SR if needed
    if recon_sr != native_sr:
        recon = resample(recon, recon_sr, native_sr)

    min_len = min(len(audio), len(recon))
    audio   = audio[:min_len]
    recon   = recon[:min_len]

    si_snr_whole = si_snr(audio, recon)
    stft_whole   = stft_distance(audio, recon, native_sr)

    call_metrics, per_call_df = compute_call_region_metrics(
        audio, recon, annotations, native_sr
    )

    if not per_call_df.empty:
        per_call_df['dataset'] = dataset
        per_call_df['filename'] = orig_path.name
        per_call_df['codec']   = codec
        per_call_df['bitrate'] = bitrate

    result = {
        'dataset'     : dataset,
        'filename'    : orig_path.name,
        'codec'       : codec,
        'bitrate'     : bitrate,
        'native_sr'   : native_sr,
        'recon_sr'    : recon_sr,
        'si_snr_whole': round(si_snr_whole, 4),
        'stft_whole'  : round(stft_whole,   4),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in call_metrics.items()},
    }
    return result, per_call_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--codec',   required=True,
                        choices=['encodec', 'dac', 'mp3', 'opus'])
    parser.add_argument('--bitrate', default=None,
                        help='Specific bitrate to evaluate. If omitted, all bitrates.')
    args = parser.parse_args()

    config       = DATASET_CONFIG[args.dataset]
    recon_subdir = config['recon_subdir']
    audio_dir    = config['audio_dir']
    label_dir    = config['label_dir']
    label_ext    = config['label_ext']
    audio_ext    = config['audio_ext']
    audio_glob   = config.get('audio_glob', f'*.{audio_ext}')
    annotation_ext = config.get('annotation_ext', audio_ext)

    bitrates = [args.bitrate] if args.bitrate else get_bitrates(args.dataset, args.codec)
    if not bitrates:
        print(f"ERROR: no reconstructed audio found for {args.dataset}/{args.codec}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Native SR evaluation: {args.dataset} | {args.codec}")
    print(f"Bitrates: {bitrates}")
    print(f"{'='*60}")

    # Get original audio files
    orig_files = sorted(audio_dir.glob(audio_glob))
    print(f"Original files: {len(orig_files)}")

    for bitrate in bitrates:
        recon_dir    = RECON_ROOT / recon_subdir / args.codec / str(bitrate)
        out_dir      = RESULTS_ROOT / args.dataset / args.codec / str(bitrate)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path      = out_dir / 'metrics_native_sr.csv'
        call_metrics_path = out_dir / 'call_metrics_native_sr.csv'

        if not recon_dir.exists():
            print(f"  WARNING: {recon_dir} not found, skipping")
            continue

        print(f"\n  Bitrate: {bitrate}")
        print(f"  Recon dir: {recon_dir}")

        # Load already completed
        completed = set()
        if metrics_path.exists():
            existing = pd.read_csv(metrics_path)
            completed = set(existing['filename'].tolist())
            print(f"  Already completed: {len(completed)}")

        for orig_path in tqdm(orig_files, desc=f"{args.codec} @ {bitrate}", unit='file'):
            if orig_path.name in completed:
                continue

            # Find reconstructed file
            recon_path = recon_dir / (orig_path.stem + '_reconstructed.wav')
            if not recon_path.exists():
                print(f"  WARNING: recon not found for {orig_path.name}")
                continue

            # Find label file
            label_path = label_dir / (orig_path.stem + label_ext)
            if not label_path.exists():
                label_path = label_dir / orig_path.parent.name / (orig_path.stem + label_ext)
            if not label_path.exists():
                label_path = label_dir / f'annotations{label_ext}'
            if not label_path.exists():
                print(f"  WARNING: label not found for {orig_path.name}")
                continue

            result, per_call_df = evaluate_pair(
                orig_path, recon_path, label_path,
                args.dataset, args.codec, bitrate, config
            )

            if result is not None:
                append_result(str(metrics_path), result)
                append_call_metrics(str(call_metrics_path), per_call_df)
                completed.add(orig_path.name)

        print(f"  Saved to {metrics_path}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()