#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_violin_with_spectrograms.py
Combined figure formatted for Methods in Ecology and Evolution (MEE / BES):
    (a) Left:  Per-species confidence change violin plot
               Samples marked with Roman numerals i, ii, iii, iv
    (b) Right top:    Spectrogram pairs for calls with highest confidence gain
    (c) Right bottom: Spectrogram pairs for calls with highest confidence loss
Supports three datasets via --dataset flag:
    birdnet      -- Northeastern US Soundscapes, pre-trained BirdNET
    anuraset     -- AnuraSet, transfer-learned classifier
    lemur        -- Lemur, transfer-learned classifier (binary, 1 class)
    northeastern -- Northeastern US Soundscapes, transfer-learned classifier
Usage:
    python scripts/downstream/figures/plot_violin_with_spectrograms.py --dataset birdnet
    python scripts/downstream/figures/plot_violin_with_spectrograms.py --dataset anuraset
    python scripts/downstream/figures/plot_violin_with_spectrograms.py --dataset lemur
    python scripts/downstream/figures/plot_violin_with_spectrograms.py --dataset northeastern
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from xml.etree import ElementTree as ET
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import PATHS

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(PATHS['results_dir'])
SOURCE_DIR  = Path(PATHS['source_dir'])
RECON_ROOT  = Path(PATHS['recon_dir'])
FIGS_DIR    = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

BIRDNET_RAW_DIR = RESULTS_DIR / 'downstream' / 'pretrained' / 'birdnet' / 'raw'
BIRDNET_AUDIO   = SOURCE_DIR / 'northeastern_us_soundscapes' / 'soundscape_data'
BIRDNET_ANN     = SOURCE_DIR / 'northeastern_us_soundscapes' / 'annotations.csv'
BIRDNET_SPECIES = SOURCE_DIR / 'northeastern_us_soundscapes' / 'species.csv'

ANURASET_SCORES_DIR   = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                        'on_original' / 'anuraset' / 'raw'
ANURASET_META         = SOURCE_DIR / 'anuraset' / 'preprocessed' / 'metadata.csv'
ANURASET_SPECIES_JSON = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                        'embeddings' / 'anuraset' / 'anuraset_species_cols.json'
RAW_ANURASET          = SOURCE_DIR / 'anuraset' / 'raw' / 'raw_data'
STRONG_LABELS_DIR     = SOURCE_DIR / 'anuraset' / 'raw' / 'strong_labels'

LEMUR_SCORES_DIR = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                   'on_original' / 'lemur' / 'raw'
LEMUR_META       = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                   'embeddings' / 'lemur' / 'lemur_all_metadata.csv'
LEMUR_EMB_DIR    = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                   'embeddings' / 'lemur'
LEMUR_AUDIO_DIR  = SOURCE_DIR / 'black_and_white_ruffed_lemur' / 'Audio3'
LEMUR_ANN_DIR    = SOURCE_DIR / 'black_and_white_ruffed_lemur' / 'Annotations3'
LEMUR_SR         = 48000

NORTHEASTERN_SCORES_DIR = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                          'on_original' / 'northeastern' / 'raw'
NORTHEASTERN_EMB_DIR    = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                          'embeddings' / 'northeastern'

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CODEC          = 'encodec'
BITRATE        = 24.0
CODEC_COLOR    = '#08306b'
MIN_ORIG_CONF  = 0.05
N_FFT          = 2048
HOP_LENGTH     = 512
CONTEXT_SEC    = 2.0
LABELS         = ['i', 'ii', 'iii', 'iv']
ROBUST_COLOR   = '#2ca02c'
DEGRADED_COLOR = '#d62728'
FS       = 20     # general font size
FS_SMALL = 16     # sub-labels, tick labels

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
def load_call_df_birdnet():
    print("Loading BirdNET data...")
    emb_dir = NORTHEASTERN_EMB_DIR

    with open(emb_dir / 'northeastern_positive_test_indices.json') as f:
        pos_test_idx = json.load(f)
    with open(emb_dir / 'northeastern_species_cols.json') as f:
        species_cols = json.load(f)

    meta = pd.read_csv(emb_dir / 'northeastern_all_metadata.csv')
    window_lookup = {(row.filename, row.window_start): idx
                     for idx, row in meta.iterrows()}

    det_orig  = pd.read_csv(BIRDNET_RAW_DIR / 'detections_original_lowthresh.csv')
    det_recon = pd.read_csv(BIRDNET_RAW_DIR / 'detections_reconstructed_lowthresh.csv')
    det_enc   = det_recon[
        (det_recon['codec'].astype(str) == CODEC) &
        (det_recon['bitrate'].astype(float) == float(BITRATE))
    ].copy()

    for det in [det_orig, det_enc]:
        det['window_idx'] = det.apply(
            lambda r: window_lookup.get((r['filename'], r['start_sec']), None),
            axis=1)

    det_orig = det_orig.dropna(subset=['window_idx'])
    det_enc  = det_enc.dropna(subset=['window_idx'])
    det_orig['window_idx'] = det_orig['window_idx'].astype(int)
    det_enc['window_idx']  = det_enc['window_idx'].astype(int)

    rows = []
    for sp in species_cols:
        pos_idx = pos_test_idx.get(sp, [])
        if len(pos_idx) == 0:
            continue
        sp_orig = det_orig[
            (det_orig['common_name'] == sp) &
            (det_orig['window_idx'].isin(pos_idx))
        ][['window_idx', 'confidence']].rename(columns={'confidence': 'conf_orig'})
        sp_enc = det_enc[
            (det_enc['common_name'] == sp) &
            (det_enc['window_idx'].isin(pos_idx))
        ][['window_idx', 'confidence']].rename(columns={'confidence': 'conf_enc'})
        pos_df = pd.DataFrame({'window_idx': pos_idx})
        merged = pos_df.merge(sp_orig, on='window_idx', how='left')
        merged = merged.merge(sp_enc,  on='window_idx', how='left')
        merged[['conf_orig', 'conf_enc']] = \
            merged[['conf_orig', 'conf_enc']].fillna(0.0)
        merged = merged.merge(
            meta[['filename', 'window_start']].reset_index().rename(
                columns={'index': 'window_idx'}),
            on='window_idx', how='left')
        merged['species']  = sp
        merged['drop_enc'] = merged['conf_enc'] - merged['conf_orig']
        rows.append(merged)

    call_df = pd.concat(rows, ignore_index=True)
    print(f"  Positive windows: {len(call_df):,} across "
          f"{call_df['species'].nunique()} species")

    ann = pd.read_csv(BIRDNET_ANN)
    ann = ann.rename(columns={
        'Filename'          : 'filename',
        'Start Time (s)'    : 'ann_start',
        'End Time (s)'      : 'ann_end',
        'Low Freq (Hz)'     : 'low_freq',
        'High Freq (Hz)'    : 'high_freq',
        'Species eBird Code': 'ebird_code',
    })
    ann['filename'] = ann['filename'].str.replace('.flac', '.wav', regex=False)
    species_map     = pd.read_csv(BIRDNET_SPECIES)
    ebird_to_common = dict(zip(species_map['Species eBird Code'],
                               species_map['Common Name']))
    ann['common_name'] = ann['ebird_code'].map(ebird_to_common)
    return call_df, ann, 32000


def load_call_df_northeastern():
    print("Loading Northeastern transfer-learned data...")
    meta = pd.read_csv(NORTHEASTERN_EMB_DIR / 'northeastern_all_metadata.csv')

    with open(NORTHEASTERN_EMB_DIR / 'northeastern_positive_test_indices.json') as f:
        pos_test_idx = json.load(f)
    with open(NORTHEASTERN_EMB_DIR / 'northeastern_species_cols.json') as f:
        species_cols = json.load(f)

    scores_orig = pd.read_csv(NORTHEASTERN_SCORES_DIR / 'scores_original_on_original.csv')
    scores_enc  = pd.read_csv(
        NORTHEASTERN_SCORES_DIR / f'scores_original_on_{CODEC}{BITRATE}.csv')

    rows = []
    for sp in species_cols:
        pos_idx = pos_test_idx.get(sp, [])
        if len(pos_idx) == 0:
            continue
        pos_mask = scores_orig['clip_idx'].isin(pos_idx)
        if pos_mask.sum() == 0:
            continue
        pos_orig = scores_orig[pos_mask][['clip_idx', sp]].rename(
            columns={sp: 'conf_orig'})
        pos_enc  = scores_enc[scores_enc['clip_idx'].isin(pos_idx)][
            ['clip_idx', sp]].rename(columns={sp: 'conf_enc'})
        merged = pos_orig.merge(pos_enc, on='clip_idx', how='inner')
        merged = merged.merge(
            meta[['filename', 'window_start']].reset_index().rename(
                columns={'index': 'clip_idx'}),
            on='clip_idx', how='left')
        merged['species']  = sp
        merged['drop_enc'] = merged['conf_enc'] - merged['conf_orig']
        rows.append(merged)

    call_df = pd.concat(rows, ignore_index=True)
    print(f"  Positive windows: {len(call_df):,} across "
          f"{call_df['species'].nunique()} species")

    ann = pd.read_csv(BIRDNET_ANN)
    ann = ann.rename(columns={
        'Filename'          : 'filename',
        'Start Time (s)'    : 'ann_start',
        'End Time (s)'      : 'ann_end',
        'Low Freq (Hz)'     : 'low_freq',
        'High Freq (Hz)'    : 'high_freq',
        'Species eBird Code': 'ebird_code',
    })
    ann['filename'] = ann['filename'].str.replace('.flac', '.wav', regex=False)
    species_map     = pd.read_csv(BIRDNET_SPECIES)
    ebird_to_common = dict(zip(species_map['Species eBird Code'],
                               species_map['Common Name']))
    ann['common_name'] = ann['ebird_code'].map(ebird_to_common)
    return call_df, ann, 32000

def load_call_df_anuraset():
    print("Loading AnuraSet data...")
    with open(ANURASET_SPECIES_JSON) as f:
        species_cols = json.load(f)

    meta      = pd.read_csv(ANURASET_META)
    meta_test = meta[meta['subset'] == 'test'].reset_index(drop=True)

    scores_orig = pd.read_csv(ANURASET_SCORES_DIR / 'scores_original.csv')
    scores_enc  = pd.read_csv(
        ANURASET_SCORES_DIR / f'scores_{CODEC}{BITRATE}.csv')

    # Load n_pos_test from per_species CSV
    per_species = pd.read_csv(
        RESULTS_DIR / 'downstream' / 'transfer_learned' /
        'on_original' / 'anuraset' / 'per_species.csv')
    n_pos_test = (per_species[per_species['condition'] == 'original']
                  [['species', 'n_pos_test']]
                  .drop_duplicates())

    id_cols = ['clip_idx', 'condition', 'codec', 'bitrate', 'kbps']
    orig_long = scores_orig.melt(
        id_vars=[c for c in id_cols if c in scores_orig.columns],
        value_vars=[c for c in species_cols if c in scores_orig.columns],
        var_name='species', value_name='conf_orig')
    enc_long = scores_enc.melt(
        id_vars=[c for c in id_cols if c in scores_enc.columns],
        value_vars=[c for c in species_cols if c in scores_enc.columns],
        var_name='species', value_name='conf_enc')

    call_df = orig_long[['clip_idx', 'species', 'conf_orig']].merge(
        enc_long[['clip_idx', 'species', 'conf_enc']],
        on=['clip_idx', 'species'], how='inner')
    call_df = call_df.merge(
        meta_test[['fname', 'min_t', 'max_t', 'site']].reset_index().rename(
            columns={'index': 'clip_idx'}),
        on='clip_idx', how='left')
    gt_long = meta_test[species_cols].reset_index().rename(
        columns={'index': 'clip_idx'}).melt(
        id_vars='clip_idx', var_name='species', value_name='ground_truth_label')
    call_df = call_df.merge(gt_long, on=['clip_idx', 'species'], how='left')
    call_df = call_df[call_df['ground_truth_label'] == 1].copy()

    # Merge n_pos_test
    call_df = call_df.merge(n_pos_test, on='species', how='left')
    call_df['drop_enc'] = call_df['conf_enc'] - call_df['conf_orig']
    return call_df, meta_test, 22050


# def load_call_df_anuraset():
#     print("Loading AnuraSet data...")
#     with open(ANURASET_SPECIES_JSON) as f:
#         species_cols = json.load(f)

#     meta      = pd.read_csv(ANURASET_META)
#     meta_test = meta[meta['subset'] == 'test'].reset_index(drop=True)

#     scores_orig = pd.read_csv(ANURASET_SCORES_DIR / 'scores_original.csv')
#     scores_enc  = pd.read_csv(
#         ANURASET_SCORES_DIR / f'scores_{CODEC}{BITRATE}.csv')

#     id_cols = ['clip_idx', 'condition', 'codec', 'bitrate', 'kbps']
#     orig_long = scores_orig.melt(
#         id_vars=[c for c in id_cols if c in scores_orig.columns],
#         value_vars=[c for c in species_cols if c in scores_orig.columns],
#         var_name='species', value_name='conf_orig')
#     enc_long = scores_enc.melt(
#         id_vars=[c for c in id_cols if c in scores_enc.columns],
#         value_vars=[c for c in species_cols if c in scores_enc.columns],
#         var_name='species', value_name='conf_enc')

#     call_df = orig_long[['clip_idx', 'species', 'conf_orig']].merge(
#         enc_long[['clip_idx', 'species', 'conf_enc']],
#         on=['clip_idx', 'species'], how='inner')
#     call_df = call_df.merge(
#         meta_test[['fname', 'min_t', 'max_t', 'site']].reset_index().rename(
#             columns={'index': 'clip_idx'}),
#         on='clip_idx', how='left')
#     gt_long = meta_test[species_cols].reset_index().rename(
#         columns={'index': 'clip_idx'}).melt(
#         id_vars='clip_idx', var_name='species', value_name='ground_truth_label')
#     call_df = call_df.merge(gt_long, on=['clip_idx', 'species'], how='left')
#     call_df = call_df[call_df['ground_truth_label'] == 1].copy()
#     call_df['drop_enc'] = call_df['conf_enc'] - call_df['conf_orig']
#     return call_df, meta_test, 22050


def load_call_df_lemur():
    print("Loading Lemur data...")
    meta     = pd.read_csv(LEMUR_META)
    test_idx = np.load(LEMUR_EMB_DIR / 'lemur_temporal_test_indices.npy')
    meta     = meta.iloc[test_idx].reset_index(drop=True)
    meta     = meta.reset_index().rename(columns={'index': 'clip_idx'})

    scores_orig = pd.read_csv(LEMUR_SCORES_DIR / 'scores_original_on_original.csv')
    scores_enc  = pd.read_csv(
        LEMUR_SCORES_DIR / f'scores_original_on_{CODEC}{BITRATE}.csv')

    roar_clips = meta[meta['label'] == 1].copy()
    call_df = roar_clips.merge(
        scores_orig[['clip_idx', 'score']].rename(columns={'score': 'conf_orig'}),
        on='clip_idx', how='left')
    call_df = call_df.merge(
        scores_enc[['clip_idx', 'score']].rename(columns={'score': 'conf_enc'}),
        on='clip_idx', how='left')
    call_df[['conf_orig', 'conf_enc']] = \
        call_df[['conf_orig', 'conf_enc']].fillna(0.0)
    call_df['drop_enc'] = call_df['conf_enc'] - call_df['conf_orig']
    print(f"  Roar calls: {len(call_df)}")
    return call_df, None, LEMUR_SR

# ---------------------------------------------------------------------------
# SVL PARSER
# ---------------------------------------------------------------------------
def parse_svl_annotations(svl_path, native_sr=LEMUR_SR):
    tree = ET.parse(str(svl_path))
    root = tree.getroot()
    rows = []
    for point in root.iter('point'):
        label    = point.get('label', '').strip().lower()
        if label not in ('roar', 'no-roar'):
            continue
        frame    = int(point.get('frame', 0))
        duration = int(point.get('duration', 0))
        value    = float(point.get('value', 0))
        extent   = float(point.get('extent', 0))
        rows.append({
            'start_sec' : frame / native_sr,
            'end_sec'   : (frame + duration) / native_sr,
            'low_freq'  : value,
            'high_freq' : value + extent,
            'label'     : label,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# AUDIO HELPERS
# ---------------------------------------------------------------------------
def load_snippet(path, start, end, target_sr):
    info  = sf.info(str(path))
    sr_in = info.samplerate
    s_fr  = max(0, int(round(start * sr_in)))
    e_fr  = min(int(info.frames), int(round(end * sr_in)))
    audio, _ = sf.read(str(path), start=s_fr, frames=e_fr - s_fr,
                       always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr_in != target_sr:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=target_sr)
    return audio


def load_anuraset_recon_segment(fname, t_start, t_end, target_sr):
    t_start_int     = int(np.floor(t_start))
    t_end_int       = int(np.ceil(t_end))
    stitched_chunks = []
    for t in range(t_start_int, t_end_int):
        if t < 57:
            clip_path   = (RECON_ROOT / 'anuraset' / 'preprocessed' / CODEC /
                           str(BITRATE) / f'{fname}_{t}_{t+3}_reconstructed.wav')
            slice_start, slice_end = 0.0, 1.0
        else:
            clip_path   = (RECON_ROOT / 'anuraset' / 'preprocessed' / CODEC /
                           str(BITRATE) / f'{fname}_57_60_reconstructed.wav')
            offset      = float(t - 57)
            slice_start, slice_end = offset, offset + 1.0
        if clip_path.exists():
            chunk = load_snippet(clip_path, slice_start, slice_end, target_sr)
            if len(chunk) < target_sr:
                chunk = np.pad(chunk, (0, target_sr - len(chunk)))
            elif len(chunk) > target_sr:
                chunk = chunk[:target_sr]
            stitched_chunks.append(chunk)
        else:
            print(f"  [MISSING] {clip_path.name}")
            stitched_chunks.append(np.zeros(target_sr, dtype=np.float32))
    if not stitched_chunks:
        return None
    full_stitched = np.concatenate(stitched_chunks)
    start_sample  = int(round((t_start - t_start_int) * target_sr))
    total_samples = int(round((t_end - t_start) * target_sr))
    return full_stitched[start_sample : start_sample + total_samples]


def compute_spec(audio, sr):
    S     = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db  = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr,
                                   hop_length=HOP_LENGTH)
    return S_db, freqs, times

# ---------------------------------------------------------------------------
# AUDIO LOADERS
# ---------------------------------------------------------------------------
def get_audio_birdnet(row, load_start, load_end, native_sr):
    filename   = row['filename']
    recon_stem = Path(filename).stem
    orig_audio  = load_snippet(BIRDNET_AUDIO / filename,
                               load_start, load_end, native_sr)
    recon_audio = load_snippet(
        RECON_ROOT / 'northeastern_birds' / CODEC / str(BITRATE) /
        f'{recon_stem}_reconstructed.wav',
        load_start, load_end, native_sr)
    return orig_audio, recon_audio


def get_audio_northeastern(row, load_start, load_end, native_sr):
    filename   = row['filename']
    recon_stem = Path(filename).stem
    orig_audio  = load_snippet(BIRDNET_AUDIO / filename,
                               load_start, load_end, native_sr)
    recon_audio = load_snippet(
        RECON_ROOT / 'northeastern_birds' / CODEC / str(BITRATE) /
        f'{recon_stem}_reconstructed.wav',
        load_start, load_end, native_sr)
    return orig_audio, recon_audio


def get_audio_anuraset(row, load_start, load_end, native_sr):
    fname      = row['fname']
    site       = row['site']
    orig_audio  = load_snippet(RAW_ANURASET / site / f'{fname}.wav',
                               load_start, load_end, native_sr)
    recon_audio = load_anuraset_recon_segment(fname, load_start, load_end,
                                              native_sr)
    return orig_audio, recon_audio


def get_audio_lemur(row, load_start, load_end, native_sr):
    filename   = row['filename']
    stem       = Path(filename).stem
    orig_audio  = load_snippet(LEMUR_AUDIO_DIR / filename,
                               load_start, load_end, native_sr)
    recon_audio = load_snippet(
        RECON_ROOT / 'lemur' / 'lemur_swift2' / CODEC / str(BITRATE) /
        f'{stem}_reconstructed.wav',
        load_start, load_end, native_sr)
    return orig_audio, recon_audio

# ---------------------------------------------------------------------------
# ANNOTATION RESOLVERS
# ---------------------------------------------------------------------------
def get_annotation_info_birdnet(row, ann):
    mask = (
        (ann['filename']    == row['filename']) &
        (ann['common_name'] == row['species']) &
        (ann['ann_start']   <  row['window_start'] + 3.0) &
        (ann['ann_end']     >  row['window_start'])
    )
    matches = ann[mask]
    if matches.empty:
        ws = float(row['window_start'])
        print(f"  [FALLBACK] No annotation for {row['species']} at {ws}s")
        return ws, ws + 3.0, 500.0, 8000.0
    r = matches.iloc[0]
    return (float(r['ann_start']), float(r['ann_end']),
            float(r['low_freq']),  float(r['high_freq']))


def get_annotation_info_anuraset(row, meta):
    fname   = row['fname']
    min_t   = float(row['min_t'])
    max_t   = float(row['max_t'])
    species = str(row['species']).strip()
    site    = row['site']
    label_path = STRONG_LABELS_DIR / site / f'{fname}.txt'
    if label_path.exists():
        try:
            labels_df = pd.read_csv(label_path, sep='\t', header=None,
                                    names=['start_sec', 'end_sec', 'label'])
            clean_target = species.replace('_', '').upper()
            clean_labels = labels_df['label'].astype(str).str.replace(
                '_', '').str.upper()
            sp_mask   = clean_labels.str.startswith(clean_target)
            time_mask = ((labels_df['start_sec'] < max_t) &
                         (labels_df['end_sec']   > min_t))
            matches   = labels_df[sp_mask & time_mask].copy()
            if not matches.empty:
                matches['overlap'] = (
                    np.minimum(matches['end_sec'], max_t) -
                    np.maximum(matches['start_sec'], min_t))
                r = matches.sort_values('overlap', ascending=False).iloc[0]
                return float(r['start_sec']), float(r['end_sec']), None, None
            else:
                print(f"  [FALLBACK] No strong label for {species}")
        except Exception as e:
            print(f"  [ERROR] Could not parse {label_path.name}: {e}")
    else:
        print(f"  [FALLBACK] Strong label file not found: {label_path}")
    return min_t, max_t, None, None


def get_annotation_info_lemur(row, _meta):
    filename     = row['filename']
    window_start = float(row['window_start'])
    window_end   = window_start + 3.0
    stem         = Path(filename).stem
    svl_path     = LEMUR_ANN_DIR / f'{stem}.svl'
    if svl_path.exists():
        try:
            anns      = parse_svl_annotations(svl_path)
            roar_anns = anns[
                (anns['label'] == 'roar') &
                (anns['start_sec'] < window_end) &
                (anns['end_sec']   > window_start)
            ]
            if not roar_anns.empty:
                r = roar_anns.iloc[0]
                return (float(r['start_sec']), float(r['end_sec']),
                        float(r['low_freq']),  float(r['high_freq']))
            else:
                print(f"  [FALLBACK] No roar annotation in {stem}.svl")
        except Exception as e:
            print(f"  [ERROR] SVL parse error: {e}")
    else:
        print(f"  [FALLBACK] SVL not found: {svl_path}")
    return window_start, window_end, 300.0, 1400.0

# ---------------------------------------------------------------------------
# DRAW ANNOTATION
# ---------------------------------------------------------------------------
def draw_annotation(ax, ann_start_abs, ann_end_abs, low_freq, high_freq):
    if low_freq is None:
        ax.axvline(ann_start_abs, color='white', linewidth=1.5,
                   linestyle='--', zorder=5)
        ax.axvline(ann_end_abs,   color='white', linewidth=1.5,
                   linestyle='--', zorder=5)
    else:
        ax.add_patch(plt.Rectangle(
            (ann_start_abs, low_freq),
            ann_end_abs - ann_start_abs,
            high_freq - low_freq,
            linewidth=1.5, edgecolor='white',
            facecolor='none', linestyle='--', zorder=5))

# ---------------------------------------------------------------------------
# SELECT EXAMPLES
# ---------------------------------------------------------------------------
# def select_examples(call_df):
#     species_detected = (call_df[call_df['conf_orig'] > 0]
#                         .groupby('species').size())
#     species_detected = species_detected[
#         species_detected >= 10].index.tolist()
#     df = call_df[call_df['species'].isin(species_detected)].copy()
#     species_order = (df.groupby('species')['drop_enc']
#                      .median().sort_values().index.tolist())
#     example_rows = []
#     for sp in [species_order[-1], species_order[-2]]:
#         sp_calls = df[(df['species'] == sp) &
#                       (df['conf_orig'] >= MIN_ORIG_CONF)]
#         if sp_calls.empty:
#             sp_calls = df[df['species'] == sp]
#         example_rows.append(sp_calls.loc[sp_calls['drop_enc'].idxmax()].copy())
#     for sp in [species_order[0], species_order[1]]:
#         sp_calls = df[(df['species'] == sp) &
#                       (df['conf_orig'] >= MIN_ORIG_CONF)]
#         if sp_calls.empty:
#             sp_calls = df[df['species'] == sp]
#         example_rows.append(sp_calls.loc[sp_calls['drop_enc'].idxmin()].copy())
#     examples = pd.DataFrame(example_rows).reset_index(drop=True)
#     examples['label'] = LABELS
#     examples['color'] = [ROBUST_COLOR, ROBUST_COLOR,
#                          DEGRADED_COLOR, DEGRADED_COLOR]
#     return df, species_order, examples

def select_examples(call_df):
    if 'n_pos_test' not in call_df.columns:
        raise ValueError(
            "call_df must contain 'n_pos_test' column. "
            "Ensure the data loader merges n_pos_test before calling select_examples()."
        )
    n_pos = call_df.groupby('species')['n_pos_test'].min()
    species_detected = n_pos[n_pos >= 10].index.tolist()

    df = call_df[call_df['species'].isin(species_detected)].copy()
    species_order = (df.groupby('species')['drop_enc']
                     .median().sort_values().index.tolist())

    example_rows = []
    for sp in [species_order[-1], species_order[-2]]:
        sp_calls = df[(df['species'] == sp) &
                      (df['conf_orig'] >= MIN_ORIG_CONF)]
        if sp_calls.empty:
            sp_calls = df[df['species'] == sp]
        example_rows.append(sp_calls.loc[sp_calls['drop_enc'].idxmax()].copy())
    for sp in [species_order[0], species_order[1]]:
        sp_calls = df[(df['species'] == sp) &
                      (df['conf_orig'] >= MIN_ORIG_CONF)]
        if sp_calls.empty:
            sp_calls = df[df['species'] == sp]
        example_rows.append(sp_calls.loc[sp_calls['drop_enc'].idxmin()].copy())

    examples = pd.DataFrame(example_rows).reset_index(drop=True)
    examples['label'] = LABELS
    examples['color'] = [ROBUST_COLOR, ROBUST_COLOR,
                         DEGRADED_COLOR, DEGRADED_COLOR]
    return df, species_order, examples


def select_examples_lemur(call_df):
    df = call_df[call_df['conf_orig'] >= MIN_ORIG_CONF].copy()
    if len(df) < 4:
        df = call_df.copy()
    example_rows = []
    for _, r in df.nlargest(2, 'drop_enc').iterrows():
        example_rows.append(r.copy())
    for _, r in df.nsmallest(2, 'drop_enc').iterrows():
        example_rows.append(r.copy())
    examples = pd.DataFrame(example_rows).reset_index(drop=True)
    examples['label'] = LABELS
    examples['color'] = [ROBUST_COLOR, ROBUST_COLOR,
                         DEGRADED_COLOR, DEGRADED_COLOR]
    return call_df, None, examples

# ---------------------------------------------------------------------------
# SPECTROGRAM PANEL HELPER
# ---------------------------------------------------------------------------
def render_spectrogram_pair(fig, gs_spec, gs_title, row_in_group,
                            ex_row, get_audio_fn, get_annotation_fn,
                            ann_or_meta, native_sr, is_last_row=False):
    lbl          = ex_row['label']
    drop         = ex_row['drop_enc']
    species_name = ex_row.get('species', '')

    ax_title = fig.add_subplot(gs_title)
    ax_title.axis('off')
    ax_title.text(0.5, 0.85, f'({lbl}) {species_name}',
                  transform=ax_title.transAxes,
                  fontsize=FS, color='black',
                  ha='center', va='center')

    ax_orig  = fig.add_subplot(gs_spec[0])
    ax_recon = fig.add_subplot(gs_spec[1], sharey=ax_orig)

    ann_start_abs, ann_end_abs, low_freq, high_freq = \
        get_annotation_fn(ex_row, ann_or_meta)
    load_start = max(0.0, ann_start_abs - CONTEXT_SEC)
    load_end   = ann_end_abs + CONTEXT_SEC

    if low_freq is not None:
        pad  = (high_freq - low_freq) * 2.0
        ymin = max(0, low_freq - pad)
        ymax = min(native_sr // 2, high_freq + pad)
    else:
        ymin, ymax = 0, native_sr // 2

    try:
        orig_audio, recon_audio = get_audio_fn(
            ex_row, load_start, load_end, native_sr)

        if orig_audio is None or recon_audio is None:
            raise ValueError("Could not load audio segment")

        min_len     = min(len(orig_audio), len(recon_audio))
        orig_audio  = orig_audio[:min_len]
        recon_audio = recon_audio[:min_len]

        if len(recon_audio) == 0 or np.all(recon_audio == 0):
            print(f"  Warning: reconstructed audio for ({lbl}) zeros/empty")
            ax_orig.set_visible(False)
            ax_recon.set_visible(False)
            return None, None

        spec_orig,  freqs, times = compute_spec(orig_audio,  native_sr)
        spec_recon, _,     _     = compute_spec(recon_audio, native_sr)
        times_abs = times + load_start

        vmin = np.percentile(spec_orig, 5)
        vmax = np.percentile(spec_orig, 99)

        for ax, spec in [(ax_orig, spec_orig), (ax_recon, spec_recon)]:
            ax.imshow(spec, aspect='auto', origin='lower',
                      extent=[times_abs[0], times_abs[-1],
                              freqs[0], freqs[-1]],
                      cmap='magma', vmin=vmin, vmax=vmax,
                      interpolation='nearest')
            draw_annotation(ax, ann_start_abs, ann_end_abs,
                            low_freq, high_freq)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(times_abs[0], times_abs[-1])

            # Remove the lowest x tick to avoid overlap with y-axis label
            # ticks = [t for t in ax.get_xticks() if t > times_abs[0]]
            # ax.set_xticks(ticks)
            ax.tick_params(labelsize=FS_SMALL)
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        ax_orig.set_title(
            f'Original\n(conf = {ex_row["conf_orig"]:.2f})',
            fontsize=18, pad=20)
        ax_recon.set_title(
            f'Reconstructed\n'
            f'(conf = {ex_row["conf_enc"]:.2f}, '
            f'$\\Delta$ = {drop:+.2f})',
            fontsize=18, pad=20)

        ax_orig.set_ylabel('Freq (Hz)', fontsize=FS)
        plt.setp(ax_recon.get_yticklabels(), visible=False)

        if is_last_row:
            ax_orig.set_xlabel('Time (s)', fontsize=FS)
            ax_recon.set_xlabel('Time (s)', fontsize=FS)

        return ax_orig, ax_recon

    except Exception as e:
        print(f"  Could not load audio for ({lbl}): {e}")
        ax_orig.set_visible(False)
        ax_recon.set_visible(False)
        return None, None

# ---------------------------------------------------------------------------
# MAIN PLOT — BirdNET / AnuraSet / Northeastern
# ---------------------------------------------------------------------------
def plot_figure(call_df, species_order, examples, ann_or_meta,
                get_audio_fn, get_annotation_fn,
                native_sr, dataset_label, output_path):
    species_n = {sp: len(call_df[call_df['species'] == sp])
                 for sp in species_order}

    fig      = plt.figure(figsize=(18.0, 22.0))
    outer_gs = gridspec.GridSpec(1, 2,
                                 width_ratios=[1.2, 1.0], wspace=0.22)

    # ---- (a) VIOLIN ----
    ax_violin = fig.add_subplot(outer_gs[0, 0])
    data      = [call_df[call_df['species'] == sp]['drop_enc'].values
                 for sp in species_order]
    positions = list(range(len(species_order)))

    parts = ax_violin.violinplot(data, positions=positions,
                                 orientation='horizontal',
                                 showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(CODEC_COLOR)
        pc.set_alpha(0.55)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.4)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.4)
    parts['cmedians'].set_zorder(5)

    ax_violin.axvline(0, color='black', linestyle='-', lw=0.9, alpha=0.4)
    ax_violin.axvspan(-1.15, 0, alpha=0.03, color='red')
    ax_violin.axvspan(0, 1.10, alpha=0.03, color='green')

    for _, ex_row in examples.iterrows():
        lbl     = ex_row['label']
        color   = ex_row['color']
        species = ex_row['species']
        val     = float(ex_row['drop_enc'])
        if species not in species_order:
            continue
        ypos = species_order.index(species)
        ax_violin.scatter(val, ypos, marker='o', s=150,
                          color=color, edgecolors='black',
                          linewidths=1.2, zorder=15)
        if lbl in ['i', 'ii']:
            ax_violin.text(val - 0.055, ypos, lbl,
                           va='center', ha='right',
                           fontsize=FS,
                           color='black', zorder=16)
        else:
            ax_violin.text(val, ypos - 0.60, lbl,
                           va='top', ha='center',
                           fontsize=FS,
                           color='black', zorder=16)

    y_labels = [f"{sp}  (n={species_n[sp]})" for sp in species_order]
    ax_violin.set_yticks(positions)
    ax_violin.set_yticklabels(y_labels, fontsize=FS)
    ax_violin.set_xlabel('Confidence change (compressed \u2212 original)',
                         fontsize=FS, labelpad=9)
    ax_violin.set_xlim(-1.15, 1.10)
    ax_violin.xaxis.set_major_locator(ticker.MultipleLocator(0.50))
    ax_violin.tick_params(axis='x', labelsize=FS)
    ax_violin.grid(True, axis='x', alpha=0.3)
    ax_violin.text(-0.02, 1.015, '(a) Per-species confidence change distribution',
                   transform=ax_violin.transAxes,
                   fontsize=22, color='black',
                   va='bottom', ha='left')

    # ---- (b) & (c) SPECTROGRAMS ----
    right_gs   = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1], hspace=0.15)
    PANEL_TAGS = [
        '(b) Calls with highest confidence gain',
        '(c) Calls with highest confidence loss',
    ]

    for group_idx in range(2):
        group_sub_gs = gridspec.GridSpecFromSubplotSpec(
            5, 2, subplot_spec=right_gs[group_idx, 0],
            height_ratios=[0.12, 0.25, 1.0, 0.25, 1.0],
            hspace=0.55, wspace=0.12)

        # Render panel tag in its own dedicated row
        ax_panel = fig.add_subplot(group_sub_gs[0, :])
        ax_panel.axis('off')
        ax_panel.text(0.0, 0.5, PANEL_TAGS[group_idx],
                    transform=ax_panel.transAxes,
                    fontsize=22, color='black',
                    va='center', ha='left')

        for row_in_group in range(2):
            row_idx        = group_idx * 2 + row_in_group
            ex_row         = examples.iloc[row_idx]
            title_grid_row = 1 if row_in_group == 0 else 3
            spec_grid_row  = 2 if row_in_group == 0 else 4

            render_spectrogram_pair(
                fig=fig,
                gs_spec=[group_sub_gs[spec_grid_row, 0],
                         group_sub_gs[spec_grid_row, 1]],
                gs_title=group_sub_gs[title_grid_row, :],
                row_in_group=row_in_group,
                ex_row=ex_row,
                get_audio_fn=get_audio_fn,
                get_annotation_fn=get_annotation_fn,
                ann_or_meta=ann_or_meta,
                native_sr=native_sr,
                is_last_row=(row_in_group == 1))

    pos_top = right_gs[0, 0].get_position(fig)
    pos_bot = right_gs[1, 0].get_position(fig)
    mid_y   = (pos_top.y0 + pos_bot.y1) / 2.0
    # fig.add_artist(plt.Line2D(
    #     [pos_top.x0 - 0.03, pos_top.x1 + 0.01], [mid_y, mid_y],
    #     transform=fig.transFigure,
    #     color='grey', linewidth=1.1, linestyle='--', alpha=0.5))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

# ---------------------------------------------------------------------------
# LEMUR FIGURE
# ---------------------------------------------------------------------------
def plot_lemur(call_df, examples, native_sr, output_path):
    fig = plt.figure(figsize=(16.0, 12.0))
    gs  = gridspec.GridSpec(2, 1,
                            height_ratios=[1.0, 1.8],
                            hspace=0.35)

    # ---- (a) VIOLIN ----
    ax_violin = fig.add_subplot(gs[0, 0])
    parts = ax_violin.violinplot(
        call_df['drop_enc'].values,
        positions=[0],
        orientation='horizontal',
        showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(CODEC_COLOR)
        pc.set_alpha(0.55)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.4)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.8)
    parts['cmedians'].set_zorder(5)

    ax_violin.axvline(0, color='black', linestyle='-', lw=0.9, alpha=0.4)
    ax_violin.axvspan(-1.05, 0, alpha=0.03, color='red')
    ax_violin.axvspan(0, 1.05, alpha=0.03, color='green')

    for _, ex_row in examples.iterrows():
        lbl   = ex_row['label']
        color = ex_row['color']
        val   = float(ex_row['drop_enc'])
        ax_violin.scatter(val, 0, marker='o', s=150,
                          color=color, edgecolors='black',
                          linewidths=1.2, zorder=15)
        offset = 0.12 if lbl in ['i', 'ii'] else -0.12
        va     = 'bottom' if lbl in ['i', 'ii'] else 'top'
        ax_violin.text(val, offset, lbl,
                       transform=ax_violin.get_xaxis_transform(),
                       fontsize=FS,
                       color='black', ha='center', va=va, zorder=16)

    ax_violin.set_yticks([0])
    ax_violin.set_yticklabels(['Roar calls'], fontsize=FS_SMALL)
    ax_violin.set_xlabel('Confidence change (compressed \u2212 original)',
                         fontsize=FS, labelpad=9)
    ax_violin.set_xlim(-1.05, 1.05)
    ax_violin.tick_params(axis='x', labelsize=FS_SMALL)
    ax_violin.grid(True, axis='x', alpha=0.3)
    ax_violin.text(-0.02, 1.05, '(a) Per-species confidence change distribution',
                   transform=ax_violin.transAxes,
                   fontsize=25, va='bottom', ha='left')

    # ---- (b) & (c) SPECTROGRAMS ----
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 0], wspace=0.15)
    PANEL_TAGS = [
        '(b) Calls with highest confidence gain',
        '(c) Calls with highest confidence loss',
    ]
    GROUP_ROWS = [[0, 1], [2, 3]]

    for group_idx in range(2):
        group_sub_gs = gridspec.GridSpecFromSubplotSpec(
            4, 2, subplot_spec=bottom_gs[0, group_idx],
            height_ratios=[0.09, 1.0, 0.09, 1.0],
            hspace=0.48, wspace=0.12)

        for row_in_group in range(2):
            row_idx        = GROUP_ROWS[group_idx][row_in_group]
            ex_row         = examples.iloc[row_idx]
            title_grid_row = 0 if row_in_group == 0 else 2
            spec_grid_row  = 1 if row_in_group == 0 else 3

            render_spectrogram_pair(
                fig=fig,
                gs_spec=[group_sub_gs[spec_grid_row, 0],
                         group_sub_gs[spec_grid_row, 1]],
                gs_title=group_sub_gs[title_grid_row, :],
                row_in_group=row_in_group,
                ex_row=ex_row,
                get_audio_fn=get_audio_lemur,
                get_annotation_fn=get_annotation_info_lemur,
                ann_or_meta=None,
                native_sr=native_sr,
                panel_tag=PANEL_TAGS[group_idx] if row_in_group == 0 else None,
                is_last_row=(row_in_group == 1))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['birdnet', 'anuraset', 'lemur', 'northeastern'])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else \
             FIGS_DIR / f'violin_with_spectrograms_{args.dataset}.png'

    if args.dataset == 'birdnet':
        call_df, ann_or_meta, native_sr = load_call_df_birdnet()
        call_df, species_order, examples = select_examples(call_df)
        print("\n  Examples selected:")
        for _, r in examples.iterrows():
            print(f"    ({r['label']}) {r['species']:35s}  "
                  f"drop={r['drop_enc']:+.3f}  "
                  f"orig={r['conf_orig']:.2f}  enc={r['conf_enc']:.2f}")
        plot_figure(call_df, species_order, examples, ann_or_meta,
                    get_audio_birdnet, get_annotation_info_birdnet,
                    native_sr, 'Northeastern US Soundscapes (Pre-trained BirdNET)',
                    output)

    elif args.dataset == 'northeastern':
        call_df, ann_or_meta, native_sr = load_call_df_northeastern()
        call_df, species_order, examples = select_examples(call_df)
        print("\n  Examples selected:")
        for _, r in examples.iterrows():
            print(f"    ({r['label']}) {r['species']:35s}  "
                  f"drop={r['drop_enc']:+.3f}  "
                  f"orig={r['conf_orig']:.2f}  enc={r['conf_enc']:.2f}")
        plot_figure(call_df, species_order, examples, ann_or_meta,
                    get_audio_northeastern, get_annotation_info_birdnet,
                    native_sr, 'Northeastern US Soundscapes (Transfer-Learned)',
                    output)

    elif args.dataset == 'anuraset':
        call_df, ann_or_meta, native_sr = load_call_df_anuraset()
        call_df, species_order, examples = select_examples(call_df)
        print("\n  Examples selected:")
        for _, r in examples.iterrows():
            print(f"    ({r['label']}) {r['species']:35s}  "
                  f"drop={r['drop_enc']:+.3f}  "
                  f"orig={r['conf_orig']:.2f}  enc={r['conf_enc']:.2f}")
        plot_figure(call_df, species_order, examples, ann_or_meta,
                    get_audio_anuraset, get_annotation_info_anuraset,
                    native_sr, 'AnuraSet', output)

    else:  # lemur
        call_df, _, native_sr = load_call_df_lemur()
        call_df, _, examples  = select_examples_lemur(call_df)
        print("\n  Examples selected:")
        for _, r in examples.iterrows():
            print(f"    ({r['label']}) {r['filename']}  "
                  f"window={r['window_start']:.1f}s  "
                  f"drop={r['drop_enc']:+.3f}  "
                  f"orig={r['conf_orig']:.2f}  enc={r['conf_enc']:.2f}")
        plot_lemur(call_df, examples, native_sr, output)


if __name__ == '__main__':
    main()