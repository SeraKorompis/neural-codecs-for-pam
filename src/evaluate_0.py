# src/evaluate.py
import os
import gc
import torch
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    load_audio,
    save_audio,
    resample,
    load_anuraset_labels,
    load_northeastern_labels,
    load_lemur_labels,
    si_snr,
    stft_distance,
    compute_call_region_metrics,
    append_result,
    append_call_metrics,
    find_densest_window
)
from src.compress import run_codec, CODEC_CONFIG
from src.config import PATHS

PROJECT_ROOT     = Path(__file__).parent.parent   # code location
DATA_DIR         = Path(PATHS['source_dir'])        # from config
RESULTS_BASE_DIR = Path(PATHS['results_dir'])     # from config

DATASETS = {
    'anuraset': {
        'audio_dir'        : DATA_DIR / 'anuraset' / 'raw' / 'raw_data', # Use raw data for evaluation
        'label_dir'        : DATA_DIR / 'anuraset' / 'raw' /'strong_labels',
        'label_fn'         : load_anuraset_labels,
        'label_ext'        : '.txt',
        'audio_ext'        : '.wav',
        'n_files_test'     : 3,
        'n_files_full'     : None,
        'stratify_key_fn'  : lambda p: p.parent.name,
        'eval_duration_sec': None,
        'codec_sr'         : {
            'encodec' : 24000,
            'dac'     : 44100,
            'mp3'     : 44100,
            'opus'    : 44100,
        },
        'bitrates'         : {
            'encodec' : [1.5, 3.0, 6.0, 12.0, 24.0],
            'dac'     : [2, 3, 6, 9],
            'mp3'     : [8, 16, 24, 32, 64],
            'opus'    : [6, 8, 12, 14, 24, 32],
        },
    },
    'northeastern_birds': {
        'audio_dir'        : DATA_DIR / 'northeastern_us_soundscapes' / 'soundscape_data',
        'label_dir'        : DATA_DIR / 'northeastern_us_soundscapes',
        'label_fn'         : load_northeastern_labels,
        'label_ext'        : '.csv',
        'audio_ext'        : '.wav',
        'annotation_ext'   : '.flac',
        'n_files_test'     : 3,
        'n_files_full'     : None,
        'stratify_key_fn'  : None,
        'eval_duration_sec': None,
        'codec_sr'         : {
            'encodec' : 24000,
            'dac'     : 44100,
            'mp3'     : 44100,
            'opus'    : 44100,
        },
        'bitrates'         : {
            'encodec' : [1.5, 3.0, 6.0, 12.0, 24.0],
            'dac'     : [2, 3, 6, 9],
            'mp3'     : [8, 16, 24, 32, 64],
            'opus'    : [6, 8, 12, 14, 24, 32]
        },
    },
    'lemur_s4a': {
        'audio_dir'        : DATA_DIR / 'black_and_white_ruffed_lemur',
        'label_dir'        : DATA_DIR / 'black_and_white_ruffed_lemur',
        'label_fn'         : load_lemur_labels,
        'label_ext'        : '.svl',
        'audio_ext'        : '.wav',
        'audio_subdirs'    : ['Audio1'],
        'label_subdirs'    : ['Annotations1'],
        'audio_glob'       : 'S4A*.wav',
        'n_files_test'     : 1,
        'n_files_full'     : None,
        'stratify_key_fn'  : None,
        'eval_duration_sec': None,
        'codec_sr'         : {
            'encodec' : 24000,
            'dac'     : 44100,
            'mp3'     : 44100,
            'opus'    : 44100,
        },
        'bitrates'         : {
            'encodec' : [1.5, 3.0, 6.0, 12.0, 24.0],
            'dac'     : [2, 3, 6, 9],
            'mp3'     : [8, 16, 24, 32, 64],
            'opus'    : [6, 8, 12, 14, 24, 32]
        },
    },
    'lemur_swift1': {
        'audio_dir'        : DATA_DIR / 'black_and_white_ruffed_lemur',
        'label_dir'        : DATA_DIR / 'black_and_white_ruffed_lemur',
        'label_fn'         : load_lemur_labels,
        'label_ext'        : '.svl',
        'audio_ext'        : '.wav',
        'audio_subdirs'    : ['Audio1'],
        'label_subdirs'    : ['Annotations1'],
        'audio_glob'       : 'swift1*.wav',
        'n_files_test'     : 1,
        'n_files_full'     : None,
        'stratify_key_fn'  : None,
        'min_duration_min' : 55,
        'eval_duration_sec': None,
        'codec_sr'         : {
            'encodec' : 24000,
            'dac'     : 44100,
            'mp3'     : 44100,
            'opus'    : 44100,
        },
        'bitrates'         : {
            'encodec' : [1.5, 3.0, 6.0, 12.0, 24.0],
            'dac'     : [2, 3, 6, 9],
            'mp3'     : [8, 16, 24, 32, 64],
            'opus'    : [6, 8, 12, 14, 24, 32],
        },
    },
    'lemur_swift2': {
        'audio_dir'        : DATA_DIR / 'black_and_white_ruffed_lemur',
        'label_dir'        : DATA_DIR / 'black_and_white_ruffed_lemur',
        'label_fn'         : load_lemur_labels,
        'label_ext'        : '.svl',
        'audio_ext'        : '.wav',
        'audio_subdirs'    : ['Audio3'],
        'label_subdirs'    : ['Annotations3'],
        'audio_glob'       : 'swift2*.wav',
        'n_files_test'     : 1,
        'n_files_full'     : None,
        'stratify_key_fn'  : None,
        'eval_duration_sec': None,
        'codec_sr'         : {
            'encodec' : 24000,
            'dac'     : 44100,
            'mp3'     : 44100,
            'opus'    : 44100,
        },
        'bitrates'         : {
            'encodec' : [1.5, 3.0, 6.0, 12.0, 24.0],
            'dac'     : [2, 3, 6, 9],
            'mp3'     : [8, 16, 24, 32, 64],
            'opus'    : [6, 8, 12, 14, 24, 32]
        },
    },
}


def select_files_stratified(files, n_files, key_fn=None, min_duration_min=None):
    import soundfile as sf
    from collections import defaultdict
    if min_duration_min is not None:
        min_sec = min_duration_min * 60
        files   = [(a, l) for a, l in files if sf.info(str(a)).duration >= min_sec]
    if not files:
        return files
    if n_files is None or n_files >= len(files):
        return files
    if key_fn is None:
        idx = np.round(np.linspace(0, len(files) - 1, n_files)).astype(int)
        return [files[i] for i in idx]
    strata = defaultdict(list)
    for f in files:
        audio_path, _ = f
        strata[key_fn(audio_path)].append(f)
    n_strata  = len(strata)
    n_per     = n_files // n_strata
    remainder = n_files  % n_strata
    selected  = []
    for i, (key, stratum) in enumerate(sorted(strata.items())):
        n_from = min(n_per + (1 if i < remainder else 0), len(stratum))
        if n_from == 0:
            continue
        idx = np.round(np.linspace(0, len(stratum) - 1, n_from)).astype(int)
        selected.extend([stratum[j] for j in idx])
    return selected


def get_files(dataset_name, mode='test'):
    config    = DATASETS[dataset_name]
    audio_dir = config['audio_dir']
    label_dir = config['label_dir']
    audio_ext = config['audio_ext']
    label_ext = config['label_ext']
    label_fn  = config['label_fn']
    n_files        = (config.get('n_files_full', None)
                      if mode == 'full'
                      else config.get('n_files_test', 15))
    audio_subdirs  = config.get('audio_subdirs', None)
    label_subdirs  = config.get('label_subdirs', None)
    audio_glob     = config.get('audio_glob', f'*{audio_ext}')
    annotation_ext = config.get('annotation_ext', audio_ext)

    if audio_subdirs and label_subdirs:
        matched = []
        for audio_sub, label_sub in zip(audio_subdirs, label_subdirs):
            for audio_path in sorted((audio_dir / audio_sub).glob(audio_glob)):
                label_path = label_dir / label_sub / (audio_path.stem + label_ext)
                if label_path.exists():
                    matched.append((audio_path, label_path))
    else:
        audio_files = sorted(audio_dir.glob(f'**/*{audio_ext}'))
        matched     = []
        for audio_path in audio_files:
            label_path = label_dir / audio_path.parent.name / (audio_path.stem + label_ext)
            if not label_path.exists():
                label_path = label_dir / (audio_path.stem + label_ext)
            if not label_path.exists():
                label_path = label_dir / f'annotations{label_ext}'
            if label_path.exists():
                matched.append((audio_path, label_path))

    matched_annotated = []
    for audio_path, label_path in matched:
        annotation_filename = audio_path.stem + annotation_ext
        try:
            annotations = label_fn(str(label_path), filename=annotation_filename)
            if len(annotations) > 0:
                matched_annotated.append((audio_path, label_path))
        except Exception:
            continue

    selected = matched_annotated[:n_files] if n_files is not None else matched_annotated
    print(f"{dataset_name}: {len(matched)} files found, "
          f"{len(matched_annotated)} with annotations, "
          f"using {len(selected)} (sequential)")
    return selected


def print_results_summary(metrics_path):
    if not metrics_path.exists():
        print(f"No metrics.csv found from {metrics_path}")
        return
    metrics = pd.read_csv(metrics_path)
    print(f"\nmetrics.csv summary from ({metrics_path}) dir")
    print("=" * 45)
    print(f"Total rows: {len(metrics)}")
    print(f"Datasets:   {', '.join(metrics['dataset'].unique())}")
    print(f"Codecs:     {', '.join(metrics['codec'].unique())}\n")
    for dataset in metrics['dataset'].unique():
        print(f"  {dataset}:")
        subset = metrics[metrics['dataset'] == dataset]
        for codec in subset['codec'].unique():
            csub     = subset[subset['codec'] == codec]
            bitrates = sorted(csub['bitrate'].unique().tolist())
            n_files  = csub['filename'].nunique()
            print(f"    {codec}: {n_files} files, bitrates={bitrates}")


def delete_results(results_dir, dataset=None, codec=None, bitrate=None):
    if all(x is None for x in [dataset, codec, bitrate]):
        raise ValueError("Provide at least one filter, refusing to delete everything.")
    metrics_path      = results_dir / 'metrics' / (dataset or '*') / (codec or '*') / 'metrics.csv'
    call_metrics_path = results_dir / 'metrics' / (dataset or '*') / (codec or '*') / 'call_metrics.csv'
    recon_base        = results_dir / 'reconstructed_audio'
    print("delete_results: use metrics_path directly for targeted deletion.")


def evaluate_file(audio_path,
                  label_path,
                  dataset_name,
                  codec,
                  bitrate,
                  recon_dir,
                  device=None,
                  ):
    import time
    import torch
    import soundfile as sf

    config         = DATASETS[dataset_name]
    label_fn       = config['label_fn']
    target_sr      = config['codec_sr'][codec]
    eval_duration  = config.get('eval_duration_sec', None)
    annotation_ext = config.get('annotation_ext', config['audio_ext'])

    annotation_filename = audio_path.stem + annotation_ext
    annotations = label_fn(str(label_path), filename=annotation_filename)
    if len(annotations) == 0:
        print(f"  Skipping {audio_path.name}, no annotations")
        return None, None

    if eval_duration is not None:
        info      = sf.info(str(audio_path))
        total_sec = info.duration
        native_sr = info.samplerate
        window_start = find_densest_window(
            annotations, window_sec=eval_duration, total_sec=total_sec
        )
        window_end = window_start + eval_duration
        try:
            audio, native_sr = load_audio(
                str(audio_path), offset=window_start, duration=eval_duration)
        except Exception as e:
            print(f"  Skipping {audio_path.name}, could not load audio: {e}")
            return None, None
        annotations = annotations[
            (annotations['start_sec'] >= window_start) &
            (annotations['end_sec']   <= window_end)
        ].copy()
        annotations['start_sec'] = (annotations['start_sec'] - window_start).round(4)
        annotations['end_sec']   = (annotations['end_sec']   - window_start).round(4)
        print(f"  Extracted {eval_duration}s window: "
              f"{window_start:.1f}s to {window_end:.1f}s "
              f"({len(annotations)} calls)")
        if len(annotations) == 0:
            print(f"  Skipping {audio_path.name}, no annotations in window")
            return None, None
    else:
        try:
            audio, native_sr = load_audio(str(audio_path))
        except Exception as e:
            print(f"  Skipping {audio_path.name}, could not load audio: {e}")
            return None, None

    codec_save_path = None

    # Embeddings saved relative to recon_dir parent
    embeddings_dir = recon_dir.parent / 'embeddings' / dataset_name / codec / str(bitrate)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_csv_path = embeddings_dir / 'embeddings.csv'

    t_start = time.time()
    try:
        (reconstructed, empirical_kbps,
         encode_time_sec, decode_time_sec) = run_codec(
            audio, native_sr, codec, bitrate,
            target_sr=target_sr, device=device, save_path=codec_save_path,
            embeddings_csv_path=str(embeddings_csv_path),
            embeddings_filename=audio_path.name,
            embeddings_dataset=dataset_name,
        )
    except torch.cuda.OutOfMemoryError:
        if device == 'cpu':
            print(f"  OOM on CPU for {audio_path.name}, skipping")
            return None, None
        print(f"  CUDA OOM on {audio_path.name}, clearing cache and retrying on CPU")
        torch.cuda.empty_cache()
        try:
            (reconstructed, empirical_kbps,
             encode_time_sec, decode_time_sec) = run_codec(
                audio, native_sr, codec, bitrate,
                target_sr=target_sr, device='cpu', save_path=codec_save_path,
                embeddings_csv_path=str(embeddings_csv_path),
                embeddings_filename=audio_path.name,
                embeddings_dataset=dataset_name,
            )
        except Exception as e:
            print(f"  CPU retry also failed for {audio_path.name}: {e}")
            return None, None
    except Exception as e:
        print(f"  Error compressing {audio_path.name} with {codec}: {e}")
        return None, None

    compression_time_sec = round(time.time() - t_start, 2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    audio_ref = resample(audio, native_sr, target_sr)
    min_len       = min(len(audio_ref), len(reconstructed))
    audio_ref     = audio_ref[:min_len]
    reconstructed = reconstructed[:min_len]
    audio_duration_sec = round(min_len / target_sr, 4)

    si_snr_whole = si_snr(audio_ref, reconstructed)
    stft_whole   = stft_distance(audio_ref, reconstructed, target_sr)
    call_metrics, per_call_df = compute_call_region_metrics(
        audio_ref, reconstructed, annotations, target_sr
    )

    # Reconstructed audio: recon_dir/dataset/codec/bitrate/stem_reconstructed.wav
    recon_path = (recon_dir / dataset_name / codec /
                  str(bitrate) / (audio_path.stem + '_reconstructed.wav'))
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_audio(reconstructed, target_sr, str(recon_path))
    except Exception as e:
        print(f"  Error: could not save reconstructed audio for "
              f"{audio_path.name} ({codec} @ {bitrate}): "
              f"{type(e).__name__}: {e}")
        return None, None

    del audio, audio_ref, reconstructed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not per_call_df.empty:
        per_call_df['dataset']  = dataset_name
        per_call_df['filename'] = audio_path.name
        per_call_df['codec']    = codec
        per_call_df['bitrate']  = bitrate

    result = {
        'dataset'             : dataset_name,
        'filename'            : audio_path.name,
        'codec'               : codec,
        'bitrate'             : bitrate,
        'empirical_kbps'      : round(empirical_kbps, 4) if empirical_kbps is not None else None,
        'native_sr'           : native_sr,
        'codec_sr'            : target_sr,
        'audio_duration_sec'  : audio_duration_sec,
        'compression_time_sec': compression_time_sec,
        'encode_time_sec'     : round(encode_time_sec, 4),
        'decode_time_sec'     : round(decode_time_sec, 4),
        'device'              : device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'),
        'si_snr_whole'        : round(si_snr_whole, 4),
        'stft_whole'          : round(stft_whole,   4),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in call_metrics.items()},
    }
    return result, per_call_df


def run_evaluation(datasets    = None,
                   codecs      = None,
                   run_mode    = 'test',
                   results_dir = None,
                   recon_dir   = None,
                   device      = None,
                   bitrates    = None,
                   ):
    """
    Run full evaluation across datasets, codecs and bitrates.

    Metrics saved to:   results_dir/metrics/{dataset}/{codec}/metrics.csv
    Recon audio saved to: recon_dir/{dataset}/{codec}/{bitrate}/ 
                          (falls back to results_dir/reconstructed_audio/ if not set)

    Skips combinations already in metrics.csv, safe to re-run.
    """
    effective_device = device if device is not None else \
                       ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {effective_device}")
    if effective_device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if run_mode not in ['test', 'full']:
        raise ValueError("run_mode must be 'test' or 'full'")

    if datasets is None:
        datasets = list(DATASETS.keys())
    if codecs is None:
        codecs = list(CODEC_CONFIG.keys())

    results_dir = Path(results_dir) if results_dir is not None \
                  else RESULTS_BASE_DIR / run_mode

    # Reconstructed audio directory
    effective_recon_dir = Path(recon_dir) if recon_dir is not None \
                          else results_dir / 'reconstructed_audio'
    effective_recon_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        print(f"\n{'-'*45}")
        print(f"Dataset: {dataset_name} | Mode: {run_mode}")
        print(f"{'-'*45}")
        files = get_files(dataset_name, mode=run_mode)

        for codec in codecs:
            # Metrics saved per dataset/codec/bitrate — no race conditions across jobs
            # even when running one job per bitrate in parallel

            target_sr      = DATASETS[dataset_name]['codec_sr'][codec]
            codec_bitrates = DATASETS[dataset_name]['bitrates'][codec]

            if bitrates is not None:
                codec_bitrates = [b for b in codec_bitrates if b in bitrates]
                if not codec_bitrates:
                    print(f"  Warning: no matching bitrates for {codec} in {bitrates}, skipping")
                    continue

            print(f"\n  Codec: {codec} (SR: {target_sr}Hz)")
            print(f"  Bitrates: {codec_bitrates}")
            print(f"  Recon:    {effective_recon_dir}/{dataset_name}/{codec}/")

            for bitrate in codec_bitrates:
                # Per-bitrate metrics dir — safe for parallel per-bitrate jobs
                metrics_dir       = results_dir / 'metrics' / dataset_name / codec / str(bitrate)
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path      = metrics_dir / 'metrics.csv'
                call_metrics_path = metrics_dir / 'call_metrics.csv'

                # Load completed combinations for this bitrate
                completed = set()
                if metrics_path.exists():
                    existing = pd.read_csv(metrics_path)
                    for _, r in existing.iterrows():
                        completed.add((r['dataset'], r['filename'], r['codec'],
                            float(r['bitrate']) if pd.notna(r['bitrate']) else None))
                    print(f"  Found {len(completed)} already completed, skipping.")

                print(f"\n  Bitrate: {bitrate}")
                print(f"  Metrics: {metrics_path}")
                for audio_path, label_path in tqdm(files,
                                                    desc=f"{codec} @ {bitrate}",
                                                    unit='file'):
                    key = (dataset_name, audio_path.name, codec,
                           float(bitrate) if bitrate is not None else None)
                    if key in completed:
                        tqdm.write(f"  Skipping {audio_path.name} as it is already evaluated")
                        continue

                    result, per_call_df = evaluate_file(
                        audio_path, label_path,
                        dataset_name, codec, bitrate,
                        effective_recon_dir,
                        device=effective_device,
                    )

                    if result is not None:
                        append_result(str(metrics_path), result)
                        append_call_metrics(str(call_metrics_path), per_call_df)
                        completed.add(key)

            # Summary printed per bitrate above

    print(f"\n{'-'*45}")
    print(f"Evaluation complete.")
    print(f"{'-'*45}")


if __name__ == '__main__':
    run_evaluation(
        datasets = ['anuraset'],
        codecs   = ['encodec', 'dac', 'mp3', 'opus'],
        run_mode = 'test',
    )