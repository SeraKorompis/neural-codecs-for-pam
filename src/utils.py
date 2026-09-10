# src/utils.py
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfilt

# AUDIO I/O

def load_audio(path:      str,
               target_sr: int   = None,
               offset:    float = 0.0,
               duration:  float = None) -> tuple[np.ndarray, int]:
    """
    Load audio file at native sample rate as mono float32.
    Uses soundfile directly (faster, no deprecated audioread fallback).
    offset and duration allow loading a specific segment from disk,
    avoids loading the full file into RAM for long recordings.
    Falls back to librosa only for formats soundfile cannot handle.
    """
    try:
        info        = sf.info(path)
        native_sr   = info.samplerate
        start_frame = int(offset * native_sr)
        stop_frame  = int((offset + duration) * native_sr) if duration is not None else None

        # always_2d=True gives consistent (samples, channels) shape for mono and stereo
        audio, sr = sf.read(path, start=start_frame, stop=stop_frame,
                            dtype='float32', always_2d=True)

        # Mix down to mono
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    except Exception:
        # Fallback for formats soundfile cannot handle (e.g. MP3 via audioread)
        audio, sr = librosa.load(path, sr=None, mono=True,
                                  offset=offset, duration=duration)
        audio = audio.astype(np.float32)
        native_sr = sr

    # Resample only if a target SR was explicitly requested
    if target_sr is not None and target_sr != native_sr:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


def peak_normalise(audio: np.ndarray) -> np.ndarray:
    """
    Normalise audio to [-1, 1] using peak normalisation.
    Required before passing audio to any codec as all three codecs
    expect float32 input in the range [-1, 1].
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def save_audio(audio: np.ndarray, sr: int, path: str) -> None:
    """Save audio to WAV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)


# ANNOTATION LOADING

# AnuraSet Label Loader

def load_anuraset_labels(label_path: str,
                          filename:   str = None) -> pd.DataFrame:
    """
    Load AnuraSet tab-separated annotation file.
    AnuraSet has one label file per audio file so filename is not used.
    Frequency bounds are not available in AnuraSet, thus low_freq is set to 0,
    high_freq is set to Nyquist (highest frequency captured at this sample rate).
    Some label files contain blank lines, empty files, or extra columns — handled gracefully.
    Returns DataFrame with standard columns:
        start_sec, end_sec, low_freq, high_freq, label
    """
    try:
        df = pd.read_csv(label_path, sep='\t', header=None,
                         skip_blank_lines=True,
                         on_bad_lines='skip')
        df = df.iloc[:, :3]
        df.columns = ['start_sec', 'end_sec', 'label']
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['start_sec', 'end_sec',
                                     'low_freq', 'high_freq', 'label'])
    df['duration_sec'] = df['end_sec'] - df['start_sec']
    df['low_freq']     = 0.0
    df['high_freq']    = 11025.0
    return df[['start_sec', 'end_sec', 'low_freq', 'high_freq', 'label']]

# Northeastern US Soundscapes Label Loader

_northeastern_annotations = None
_northeastern_ann_path    = None

def load_northeastern_labels(label_path: str,
                              filename:   str = None) -> pd.DataFrame:
    """
    Load Northeastern US Soundscapes annotations from the single
    dataset-level CSV (annotations.csv covers all 285 recordings).
    Caches the CSV in memory: only reads from disk once per session.
    Applies a 30s duration filter to exclude presence-level annotations.

    label_path: path to annotations.csv
    filename:   audio filename to filter for e.g. 'SSW_001_20170225_010000Z.flac'
                if None returns all annotations
    Returns DataFrame with standard columns:
        start_sec, end_sec, low_freq, high_freq, label
    """
    global _northeastern_annotations, _northeastern_ann_path

    # Load and cache — only read from disk once per session
    if _northeastern_annotations is None or _northeastern_ann_path != label_path:
        ann                       = pd.read_csv(label_path)
        ann['duration_sec']       = ann['End Time (s)'] - ann['Start Time (s)']
        ann                       = ann[ann['duration_sec'] <= 30].copy()
        _northeastern_annotations = ann
        _northeastern_ann_path    = label_path

    if filename is None:
        return _northeastern_annotations

    # Filter to this specific file
    file_ann = _northeastern_annotations[
        _northeastern_annotations['Filename'] == filename
    ].copy()

    # Standardise column names
    file_ann = file_ann.rename(columns={
        'Start Time (s)'    : 'start_sec',
        'End Time (s)'      : 'end_sec',
        'Low Freq (Hz)'     : 'low_freq',
        'High Freq (Hz)'    : 'high_freq',
        'Species eBird Code': 'label',
    })

    return file_ann[['start_sec', 'end_sec', 'low_freq', 'high_freq', 'label']].reset_index(drop=True)

# Black and White Ruffed Lemur Label Loader

def load_lemur_labels(label_path: str,
                      filename:   str = None) -> pd.DataFrame:
    """
    Load Black-and-White Ruffed Lemur SVL annotation file.
    SVL is a Sonic Visualiser XML format — times stored in samples,
    frequencies in Hz. One SVL file per audio file.
    filename kwarg accepted for interface compatibility but not used.

    Derived fields:
        start_sec = frame / sampleRate
        end_sec   = (frame + duration) / sampleRate
        low_freq  = value (Hz)
        high_freq = value + extent (Hz)
        label     = roar or no-roar
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(label_path)
    root = tree.getroot()

    # Sample rate stored in model element — used to convert frames to seconds
    model       = root.find('.//model')
    if model is None:
        return pd.DataFrame(columns=['start_sec', 'end_sec',
                                     'low_freq', 'high_freq', 'label'])
    sample_rate = int(model.get('sampleRate', 16000))

    rows = []
    for point in root.findall('.//point'):
        frame    = int(point.get('frame'))
        duration = int(point.get('duration'))
        value    = float(point.get('value'))
        extent   = float(point.get('extent'))
        label    = point.get('label', '')

        rows.append({
            'start_sec' : frame / sample_rate,
            'end_sec'   : (frame + duration) / sample_rate,
            'low_freq'  : value,
            'high_freq' : value + extent,
            'label'     : label,
        })

    if not rows:
        return pd.DataFrame(columns=['start_sec', 'end_sec',
                                     'low_freq', 'high_freq', 'label'])

    return pd.DataFrame(rows)[['start_sec', 'end_sec',
                                'low_freq', 'high_freq', 'label']]

# WINDOWING

def find_densest_window(
    annotations: pd.DataFrame,
    window_sec: float,
    total_sec: float = None,
    step: float = 1.0
) -> float:
    """
    Find the start of the window containing the most annotated calls.

    More robust version:
    - Does NOT trust total_sec if annotations already define the range
    - Infers bounds from annotations when possible
    """

    if len(annotations) == 0:
        return 0.0

    # --- Infer bounds safely ---
    data_max = float(annotations["end_sec"].max())
    data_min = float(annotations["start_sec"].min())

    if total_sec is None or total_sec <= 0:
        total_sec = data_max

    # Ensure we never go outside real annotation range
    total_sec = max(total_sec, data_max)

    max_start = max(0.0, total_sec - window_sec)

    best_start = 0.0
    best_count = -1

    starts = np.arange(0.0, max_start + 1e-6, step)

    for start in starts:
        end = start + window_sec

        # fast overlap check (same logic, just cleaner boundaries)
        mask = (
            (annotations["start_sec"] < end) &
            (annotations["end_sec"] > start)
        )

        count = mask.sum()

        if count > best_count:
            best_count = count
            best_start = float(start)

    return best_start

# METRICS MEASURED

def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.
    Higher is better. Positive values indicate the signal dominates the distortion,
    negative values indicate the distortion dominates the signal.

    reference: original audio
    estimate:  reconstructed audio
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate  = estimate[:min_len]

    # Zero mean
    reference = reference - np.mean(reference)
    estimate  = estimate  - np.mean(estimate)

    # Scale-invariant projection
    dot       = np.dot(estimate, reference)
    ref_power = np.dot(reference, reference) + 1e-8
    s_target  = (dot / ref_power) * reference

    # Noise
    e_noise   = estimate - s_target

    # SI-SNR
    si_snr_val = 10 * np.log10(
        (np.sum(s_target ** 2) + 1e-8) /
        (np.sum(e_noise  ** 2) + 1e-8)
    )
    return float(si_snr_val)


def stft_distance(reference: np.ndarray,
                  estimate:  np.ndarray,
                  sr:        int,
                  fft_sizes: list[int] = None) -> float:
    """
    Multi-scale log STFT distance in dB.
    Lower is better.  Computed as the mean absolute difference between 
    log-magnitude spectrograms at multiple FFT scales.
    FFT sizes default to [2048, 1024, 512] scaled to SR.
    """
    import librosa

    if fft_sizes is None:
        # Scale FFT sizes to sample rate
        # Base sizes assume 24kHz, scale proportionally
        base_sr   = 24000
        base_sizes = [2048, 1024, 512]
        fft_sizes  = [max(256, int(s * sr / base_sr)) for s in base_sizes]
        # Round to nearest power of 2
        fft_sizes  = [2 ** round(np.log2(s)) for s in fft_sizes]

    min_len   = min(len(reference), len(estimate))
    reference = reference[:min_len].astype(np.float32)
    estimate  = estimate[:min_len].astype(np.float32)

    distances = []
    for n_fft in fft_sizes:
        hop = n_fft // 4
        S_ref  = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop))
        S_est  = np.abs(librosa.stft(estimate,  n_fft=n_fft, hop_length=hop))

        # Log magnitude
        S_ref  = librosa.amplitude_to_db(S_ref, ref=1.0)
        S_est  = librosa.amplitude_to_db(S_est, ref=1.0)

        distances.append(np.mean(np.abs(S_ref - S_est)))

    return float(np.mean(distances))

def masked_stft_distance(reference: np.ndarray,
                          estimate:  np.ndarray,
                          sr:        int,
                          low_freq:  float,
                          high_freq: float,
                          fft_sizes: list[int] = None) -> float:
    """
    Multi-scale log STFT distance computed only within the specified
    frequency range. Frequency bins outside [low_freq, high_freq] are
    zeroed out before computing the distance, no waveform filtering,
    no ringing artefacts.
    Falls back to full-spectrum STFT distance if no meaningful
    frequency bounds are provided.
    """
    nyq = sr / 2.0

    # Clamp bounds to Nyquist — annotations may have been made on the original
    # (higher) sample rate and exceed the Nyquist of the resampled codec audio
    low_freq  = min(low_freq,  nyq)
    high_freq = min(high_freq, nyq)

    # If no meaningful bounds after clamping, fall back to full-spectrum STFT
    if low_freq <= 0 and high_freq >= nyq:
        return stft_distance(reference, estimate, sr, fft_sizes)

    if fft_sizes is None:
        base_sr    = 24000
        base_sizes = [2048, 1024, 512]
        fft_sizes  = [2 ** round(np.log2(max(256, int(s * sr / base_sr))))
                      for s in base_sizes]

    min_len   = min(len(reference), len(estimate))
    reference = reference[:min_len].astype(np.float32)
    estimate  = estimate[:min_len].astype(np.float32)

    distances = []
    for n_fft in fft_sizes:
        hop    = n_fft // 4
        n_bins = n_fft // 2 + 1

        # Convert Hz bounds to bin indices
        low_bin  = max(0,      int(low_freq  / nyq * n_bins))
        high_bin = min(n_bins, int(high_freq / nyq * n_bins))

        S_ref = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop))
        S_est = np.abs(librosa.stft(estimate,  n_fft=n_fft, hop_length=hop))

        # Zero out bins outside frequency range
        mask        = np.zeros(n_bins, dtype=bool)
        mask[low_bin:high_bin] = True
        S_ref       = S_ref[mask, :]
        S_est       = S_est[mask, :]

        # Guard: clamping or int() truncation can still yield zero bins
        # (e.g. very narrow call, or low_freq == high_freq after clamping)
        if S_ref.size == 0:
            distances.append(stft_distance(reference, estimate, sr, [n_fft]))
            continue

        # Log magnitude distance on masked spectrogram
        S_ref = librosa.amplitude_to_db(S_ref, ref=1.0)
        S_est = librosa.amplitude_to_db(S_est, ref=1.0)
        distances.append(np.mean(np.abs(S_ref - S_est)))

    return float(np.mean(distances))

def compute_call_region_metrics(reference:   np.ndarray,
                                 estimate:    np.ndarray,
                                 annotations: pd.DataFrame,
                                 sr:          int) -> tuple[dict, pd.DataFrame]:
    """
    Compute SI-SNR and STFT distance within each annotated call region.
    SI-SNR: time-bounded only (waveform metric).
    STFT distance: frequency-masked where bounds available, otherwise
    full-spectrum.
    Calls shorter than 100ms are skipped.

    Returns:
        aggregated: dict of mean/std metrics across all calls
        per_call:   DataFrame with one row per call
    """
    call_si_snrs    = []
    call_stft_dists = []
    per_call_rows   = []

    has_freq_bounds = ('low_freq'  in annotations.columns and
                       'high_freq' in annotations.columns)

    for call_idx, (_, row) in enumerate(annotations.iterrows()):
        start = int(row['start_sec'] * sr)
        end   = int(row['end_sec']   * sr)

        if (end - start) < int(0.1 * sr):
            continue

        ref_slice = reference[start:end]
        est_slice = estimate[start:end]

        if len(ref_slice) < 10 or len(est_slice) < 10:
            continue

        # SI-SNR — time-bounded only
        call_si_snr = si_snr(ref_slice, est_slice)

        # STFT distance — frequency-masked if bounds available
        if has_freq_bounds:
            low_freq  = float(row['low_freq'])
            high_freq = float(row['high_freq'])
            call_stft = masked_stft_distance(
                ref_slice, est_slice, sr, low_freq, high_freq
            )
        else:
            call_stft = stft_distance(ref_slice, est_slice, sr)

        call_si_snrs.append(call_si_snr)
        call_stft_dists.append(call_stft)

        # Per-call row
        per_call_rows.append({
            'call_idx'    : call_idx,
            'start_sec'   : round(float(row['start_sec']), 4),
            'end_sec'     : round(float(row['end_sec']),   4),
            'duration_sec': round(float(row['end_sec']) - float(row['start_sec']), 4),
            'low_freq'    : float(row['low_freq'])  if has_freq_bounds else None,
            'high_freq'   : float(row['high_freq']) if has_freq_bounds else None,
            'label'       : row.get('label', None),
            'si_snr'      : round(call_si_snr, 4),
            'stft_dist'   : round(call_stft,   4),
        })

    # Aggregated metrics
    if not call_si_snrs:
        aggregated = {
            'si_snr_call_mean' : np.nan,
            'si_snr_call_std'  : np.nan,
            'stft_call_mean'   : np.nan,
            'stft_call_std'    : np.nan,
            'n_calls'          : 0,
        }
        per_call = pd.DataFrame(columns=[
            'call_idx', 'start_sec', 'end_sec', 'duration_sec',
            'low_freq', 'high_freq', 'label', 'si_snr', 'stft_dist'
        ])
    else:
        aggregated = {
            'si_snr_call_mean' : float(np.mean(call_si_snrs)),
            'si_snr_call_std'  : float(np.std(call_si_snrs)),
            'stft_call_mean'   : float(np.mean(call_stft_dists)),
            'stft_call_std'    : float(np.std(call_stft_dists)),
            'n_calls'          : len(call_si_snrs),
        }
        per_call = pd.DataFrame(per_call_rows)

    return aggregated, per_call

def compute_aggregated_metrics(metrics: pd.DataFrame,
                                dataset: str) -> pd.DataFrame:
    """
    Aggregate evaluation metrics across files per codec per bitrate.
    Computes mean and std for SI-SNR and STFT distance at whole-recording
    and call-region level. Also computes SI-SNR gap (call minus whole),
    positive values indicate calls are better preserved than background.
    If dataset='lemur', combines lemur_s4a, lemur_swift1, lemur_swift2
    using mean of subset means before aggregating (Option B).
    """
    if dataset == 'lemur':
        metrics = add_lemur_combined(metrics)
    dataset_metrics = metrics[metrics['dataset'] == dataset].copy()
 
    # Base aggregation
    agg = dataset_metrics.groupby(['codec', 'bitrate']).agg(
        si_snr_whole_mean  = ('si_snr_whole',     'mean'),
        si_snr_whole_std   = ('si_snr_whole',     'std'),
        stft_whole_mean    = ('stft_whole',        'mean'),
        stft_whole_std     = ('stft_whole',        'std'),
        si_snr_call_mean   = ('si_snr_call_mean',  'mean'),
        si_snr_call_std    = ('si_snr_call_mean',  'std'),
        stft_call_mean     = ('stft_call_mean',    'mean'),
        stft_call_std      = ('stft_call_mean',    'std'),
        n_files            = ('filename',           'count'),
    ).round(3).reset_index()
 
    agg['si_snr_gap'] = (agg['si_snr_call_mean'] - agg['si_snr_whole_mean']).round(3)
 
    # Empirical bitrate (DAC only): mean ± std across files
    if 'filesize_kbps' in dataset_metrics.columns:
        emp = dataset_metrics.groupby(['codec', 'bitrate'])['filesize_kbps'].agg(
            filesize_kbps_mean = 'mean',
            filesize_kbps_std  = 'std',
        ).round(3).reset_index()
        agg = agg.merge(emp, on=['codec', 'bitrate'], how='left')
 
        # Format as 'mean ± std' string for display; NaN for EnCodec
        def fmt(row):
            if pd.isna(row['filesize_kbps_mean']):
                return float('nan')
            return f"{row['filesize_kbps_mean']:.2f} ± {row['filesize_kbps_std']:.2f}"
        agg['empirical_kbps'] = agg.apply(fmt, axis=1)
        agg = agg.drop(columns=['filesize_kbps_mean', 'filesize_kbps_std'])
 
    # n_q column: explicit for DAC (bitrate = n_q), NaN for EnCodec
    agg['n_q'] = agg.apply(
        lambda r: int(r['bitrate']) if r['codec'] == 'dac' else float('nan'),
        axis=1
    )
 
    return agg

# RESULTS

def append_result(results_path: str, row: dict) -> None:
    """
    Append a single result row to the master CSV.
    Creates the file with headers if it doesn't exist.
    
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df  = pd.DataFrame([row])
    write_header = not os.path.exists(results_path)
    df.to_csv(results_path, mode='a', header=write_header, index=False)

def append_call_metrics(results_path: str, df: pd.DataFrame) -> None:
    """
    Append per-call metrics DataFrame to call_metrics.csv.
    Creates the file with headers if it does not already exist.
    """
    if df.empty:
        return
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    write_header = not os.path.exists(results_path)
    df.to_csv(results_path, mode='a', header=write_header, index=False)
# LEMUR COMBINED

def add_lemur_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends combined 'lemur' rows to a metrics DataFrame using mean of subset means
    (Option B): lemur_s4a, lemur_swift1, and lemur_swift2 each contribute equally
    regardless of file count or recording duration, so no single recorder dominates.

    Step 1: compute per-subset mean of each metric per (codec, bitrate)
    Step 2: average the three subset means -> one combined 'lemur' value per (codec, bitrate)

    Safe to call multiple times -- existing 'lemur' rows are dropped first.
    Call this before compute_aggregated_metrics or plot_metrics_vs_bitrate
    whenever you want 'lemur' to appear as a combined dataset.
    """
    LEMUR_SUBSETS = ['lemur_s4a', 'lemur_swift1', 'lemur_swift2']
    METRIC_COLS   = [
        'si_snr_whole',    'stft_whole',
        'si_snr_call_mean','si_snr_call_std',
        'stft_call_mean',  'stft_call_std',
    ]

    # Drop any previously computed combined rows to avoid duplicates
    df = df[df['dataset'] != 'lemur'].copy()

    lemur_df = df[df['dataset'].isin(LEMUR_SUBSETS)].copy()
    if lemur_df.empty:
        return df

    present_metrics = [c for c in METRIC_COLS if c in lemur_df.columns]

    # Step 1: per-subset mean per (codec, bitrate)
    subset_means = (lemur_df
                    .groupby(['dataset', 'codec', 'bitrate'])[present_metrics]
                    .mean()
                    .reset_index())

    # Step 2: mean of the three subset means
    combined = (subset_means
                .groupby(['codec', 'bitrate'])[present_metrics]
                .mean()
                .reset_index())
    combined['dataset'] = 'lemur'

    # Total calls across all lemur subsets
    if 'n_calls' in lemur_df.columns:
        combined = combined.merge(
            lemur_df.groupby(['codec', 'bitrate'])['n_calls'].sum().reset_index(),
            on=['codec', 'bitrate'], how='left'
        )

    # Carry nominal_kbps through — same value per (codec, bitrate), just take first
    if 'nominal_kbps' in lemur_df.columns:
        combined = combined.merge(
            lemur_df.groupby(['codec', 'bitrate'])['nominal_kbps'].first().reset_index(),
            on=['codec', 'bitrate'], how='left'
        )

    return pd.concat([df, combined], ignore_index=True)