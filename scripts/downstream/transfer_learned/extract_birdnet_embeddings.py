#!/usr/bin/env python3
"""
scripts/downstream/transfer_learned/extract_birdnet_embeddings.py

Extracts BirdNET penultimate layer (1024-dim) embeddings from audio files
for downstream classifier evaluation. BirdNET's weights are fully frozen,
it is used as a fixed feature extractor only. The logistic regression
classifier trained on these embeddings is the only learned component.

Supports AnuraSet (clip-level), Lemur (window-level) and Northeastern
(window-level, all windows kept with multilabel annotations).
Handles both original and compressed audio sources (EnCodec, DAC, MP3, Opus).

Pipeline:
    audio → BirdNET CNN (frozen) → 1024-dim embedding → logistic regression

Usage:
    # AnuraSet original test embeddings
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset anuraset --split test --source original --device CPU

    # AnuraSet compressed test embeddings (EnCodec 6.0 kbps)
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset anuraset --split test --source encodec --bitrate 6.0 --device CPU

    # Lemur original train embeddings
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset lemur --split train --source original --device CPU

    # Northeastern all windows, original
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset northeastern --split all --source original --device CPU

    # Northeastern all windows, compressed
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset northeastern --split all --source encodec --bitrate 6.0 --device CPU

Output:
    AnuraSet: $EPHEMERAL/anuraset_classifier/birdnet_transfer/embeddings/
        anuraset_{split}_{source}{bitrate}_embeddings.npy
        anuraset_{split}_labels.npy
        anuraset_{split}_metadata.csv

    Lemur: $EPHEMERAL/lemur_classifier/embeddings/
        lemur_{split}_{source}{bitrate}_embeddings.npy
        lemur_{split}_labels.npy
        lemur_{split}_metadata.csv

    Northeastern: $EPHEMERAL/northeastern_classifier/embeddings/
        northeastern_all_{source}{bitrate}_embeddings.npy
        northeastern_all_labels.npy          (n_windows, 80) multilabel
        northeastern_all_metadata.csv        (filename, window_start, site, date)
        northeastern_species_cols.json       (80 species in label column order)
"""
import argparse
import os
import sys
import gc
import json
import re
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# EPHEMERAL  = Path(os.environ.get('EPHEMERAL', str(PROJECT_ROOT)))
# RECON_ROOT = Path('/rds/general/project/bugg/live/seraphina_compression/reconstructed_audio')

# LEMUR_DIR  = PROJECT_ROOT / 'data' / 'black_and_white_ruffed_lemur'

# ANURASET_PRE_DIR = EPHEMERAL / 'anuraset_classifier' / 'data' / 'anuraset_preprocessed'
# ANURASET_OUT_DIR = EPHEMERAL / 'anuraset_classifier' / 'birdnet_transfer' / 'embeddings'

# LEMUR_OUT_DIR    = EPHEMERAL / 'lemur_classifier' / 'embeddings'

# NORTHEASTERN_DIR     = PROJECT_ROOT / 'data' / 'northeastern_us_soundscapes'
# NORTHEASTERN_OUT_DIR = EPHEMERAL / 'northeastern_classifier' / 'embeddings'

RECON_ROOT           = Path(PATHS['recon_dir'])
ANURASET_PRE_DIR     = Path(PATHS['source_dir']) / 'anuraset' / 'preprocessed'
ANURASET_OUT_DIR     = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'anuraset'
LEMUR_OUT_DIR        = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'lemur'
NORTHEASTERN_OUT_DIR = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / 'embeddings' / 'northeastern'
LEMUR_DIR            = Path(PATHS['source_dir']) / 'black_and_white_ruffed_lemur'
NORTHEASTERN_DIR     = Path(PATHS['source_dir']) / 'northeastern_us_soundscapes'

BIRDNET_SR    = 48000
BIRDNET_WIN   = 3.0   # seconds
N_WORKERS     = 1
CHUNK_SIZE    = 500   # windows per incremental save

# ---------------------------------------------------------------------------
# BIRDNET MODEL
# ---------------------------------------------------------------------------
def load_birdnet(device: str):
    import birdnet
    print(f"Loading BirdNET v2.4 (pb backend, device={device})...")
    model = birdnet.load('acoustic', '2.4', 'pb')
    print(f"  Embedding dim: {model.get_embeddings_dim()}")
    print(f"  Sample rate:   {model.get_sample_rate()} Hz")
    return model


def extract_embeddings(model, audio_batch: list, device: str) -> np.ndarray:
    result = model.encode_arrays(audio_batch, n_workers=N_WORKERS, device=device)
    return result.embeddings[:, 0, :]


# ---------------------------------------------------------------------------
# AUDIO LOADING
# ---------------------------------------------------------------------------
def load_clip(path: Path, target_sr: int = BIRDNET_SR) -> np.ndarray:
    import librosa
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def load_window(path: Path, start_sec: float, duration: float = BIRDNET_WIN,
                target_sr: int = BIRDNET_SR) -> np.ndarray:
    import librosa
    info  = sf.info(str(path))
    sr    = info.samplerate
    s_fr  = int(start_sec * sr)
    e_fr  = min(s_fr + int(duration * sr), int(info.frames))
    audio, _ = sf.read(str(path), start=s_fr, frames=e_fr - s_fr, always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    target_samples = int(duration * target_sr)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return audio[:target_samples].astype(np.float32)


# ---------------------------------------------------------------------------
# SVL ANNOTATION PARSER (Lemur)
# ---------------------------------------------------------------------------
def parse_svl(svl_path: Path, audio_sr: int) -> pd.DataFrame:
    tree = ET.parse(str(svl_path))
    root = tree.getroot()
    rows = []
    for point in root.iter('point'):
        label = point.get('label', '').strip().lower()
        if label not in ('roar', 'no-roar'):
            continue
        frame    = int(point.get('frame', 0))
        duration = int(point.get('duration', 0))
        rows.append({
            'start_sec': frame / audio_sr,
            'end_sec'  : (frame + duration) / audio_sr,
            'label'    : 1 if label == 'roar' else 0,
        })
    return pd.DataFrame(rows)


def label_window(window_start: float, window_end: float,
                 annotations: pd.DataFrame) -> int:
    overlapping = annotations[
        (annotations['start_sec'] < window_end) &
        (annotations['end_sec']   > window_start)
    ]
    if overlapping.empty:
        return -1
    if (overlapping['label'] == 1).any():
        return 1
    return 0


# ---------------------------------------------------------------------------
# NORTHEASTERN ANNOTATION HELPERS
# ---------------------------------------------------------------------------
def get_northeastern_week(filename: str) -> int:
    """Extract BirdNET week (1-48) from SSW filename."""
    m = re.search(r'_(\d{8})_', filename)
    if not m:
        return None
    date        = datetime.strptime(m.group(1), '%Y%m%d')
    day_of_year = date.timetuple().tm_yday
    return min(48, max(1, int((day_of_year - 1) / 365 * 48) + 1))


def get_northeastern_site(filename: str) -> str:
    """Extract site ID from SSW filename e.g. SSW_001."""
    m = re.match(r'(SSW_\d+)_', filename)
    return m.group(1) if m else 'unknown'


def get_northeastern_date(filename: str) -> str:
    """Extract date string from SSW filename."""
    m = re.search(r'_(\d{8})_', filename)
    return m.group(1) if m else 'unknown'


def label_window_northeastern(window_start: float, window_end: float,
                               filename: str, ann_by_file: dict,
                               species_list: list) -> np.ndarray:
    """
    Label a 3-second window for all species.
    Returns binary array of shape (n_species,).
    Any annotation overlap → label=1 for that species.
    """
    labels = np.zeros(len(species_list), dtype=np.float32)
    if filename not in ann_by_file:
        return labels
    file_ann = ann_by_file[filename]
    overlapping = file_ann[
        (file_ann['ann_start'] < window_end) &
        (file_ann['ann_end']   > window_start)
    ]
    for _, row in overlapping.iterrows():
        sp = row['common_name']
        if sp in species_list:
            idx = species_list.index(sp)
            labels[idx] = 1.0
    return labels


# ---------------------------------------------------------------------------
# PATH RESOLVERS
# ---------------------------------------------------------------------------
def get_audio_path_anuraset(row: pd.Series, source: str, bitrate) -> Path:
    fname = f"{row['fname']}_{row['min_t']}_{row['max_t']}.wav"
    if source == 'original':
        return ANURASET_PRE_DIR / 'audio' / row['site'] / fname
    recon_name = f"{row['fname']}_{row['min_t']}_{row['max_t']}_reconstructed.wav"
    return RECON_ROOT / 'anuraset' / 'preprocessed' / source / str(bitrate) / recon_name


def get_audio_path_lemur(filename: str, recorder: str,
                          source: str, bitrate) -> Path:
    stem = Path(filename).stem
    if source == 'original':
        subdir = 'Audio1' if recorder in ('s4a', 'swift1') else 'Audio3'
        return LEMUR_DIR / subdir / filename
    recon_subdir = f'lemur_{recorder}'
    # return RECON_ROOT / recon_subdir / source / str(bitrate) / f'{stem}_reconstructed.wav'
    return RECON_ROOT / 'lemur' / f'lemur_{recorder}' / source / str(bitrate) / f'{stem}_reconstructed.wav'


def get_audio_path_northeastern(filename: str, source: str, bitrate) -> Path:
    stem = Path(filename).stem
    if source == 'original':
        return NORTHEASTERN_DIR / 'soundscape_data' / filename
    return RECON_ROOT / 'northeastern_birds' / source / str(bitrate) / \
           f'{stem}_reconstructed.wav'


# ---------------------------------------------------------------------------
# ANURASET EXTRACTION
# ---------------------------------------------------------------------------
def extract_anuraset(split: str, source: str, bitrate,
                     device: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    source_tag = 'original' if source == 'original' else f'{source}{bitrate}'
    emb_path   = out_dir / f'anuraset_{split}_{source_tag}_embeddings.npy'
    lbl_path   = out_dir / f'anuraset_{split}_labels.npy'
    meta_path  = out_dir / f'anuraset_{split}_metadata.csv'
    chunk_dir  = out_dir / 'chunks' / f'anuraset_{split}_{source_tag}'
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if emb_path.exists():
        print(f"  Already exists: {emb_path.name} — skipping")
        return

    meta       = pd.read_csv(ANURASET_PRE_DIR / 'metadata.csv')
    split_meta = meta[meta['subset'] == split].reset_index(drop=True)
    species_cols = [c for c in meta.columns
                    if c not in ('sample_name','fname','min_t','max_t',
                                 'site','date','species_number','subset')]
    labels = split_meta[species_cols].values.astype(np.float32)
    print(f"  {split} clips: {len(split_meta):,}")

    done_chunks = {int(p.stem.replace('chunk_', ''))
                   for p in chunk_dir.glob('chunk_*.npy')}
    n_chunks = (len(split_meta) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Chunks: {n_chunks} total, {len(done_chunks)} done")

    model = load_birdnet(device)
    for chunk_idx in range(n_chunks):
        if chunk_idx in done_chunks:
            continue
        start      = chunk_idx * CHUNK_SIZE
        end        = min(start + CHUNK_SIZE, len(split_meta))
        chunk_meta = split_meta.iloc[start:end]
        audio_batch = []
        valid_idx   = []
        for local_i, (_, row) in enumerate(chunk_meta.iterrows()):
            path = get_audio_path_anuraset(row, source, bitrate)
            if not path.exists():
                print(f"    WARNING: missing {path.name}")
                continue
            try:
                audio = load_clip(path)
                audio_batch.append((audio, BIRDNET_SR))
                valid_idx.append(local_i)
            except Exception as e:
                print(f"    WARNING: could not load {path.name}: {e}")
        if not audio_batch:
            print(f"    Chunk {chunk_idx}: no valid files, skipping")
            continue
        embs = extract_embeddings(model, audio_batch, device)
        np.save(chunk_dir / f'chunk_{chunk_idx:04d}.npy', embs)
        print(f"    Chunk {chunk_idx+1}/{n_chunks}: {len(audio_batch)} clips embedded")
        gc.collect()

    print("  Merging chunks...")
    all_embs = []
    for chunk_idx in range(n_chunks):
        p = chunk_dir / f'chunk_{chunk_idx:04d}.npy'
        if p.exists():
            all_embs.append(np.load(p))
    embeddings = np.concatenate(all_embs, axis=0)
    np.save(emb_path, embeddings)
    print(f"  Saved: {emb_path.name} {embeddings.shape}")

    if not lbl_path.exists():
        np.save(lbl_path, labels)
        split_meta[['sample_name','site','fname','min_t','max_t']].to_csv(
            meta_path, index=False)
        print(f"  Saved: {lbl_path.name}, {meta_path.name}")


# ---------------------------------------------------------------------------
# LEMUR EXTRACTION
# ---------------------------------------------------------------------------
def extract_lemur(split: str, source: str, bitrate,
                  device: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    source_tag = 'original' if source == 'original' else f'{source}{bitrate}'
    emb_path   = out_dir / f'lemur_{split}_{source_tag}_embeddings.npy'
    lbl_path   = out_dir / f'lemur_{split}_labels.npy'
    meta_path  = out_dir / f'lemur_{split}_metadata.csv'
    chunk_dir  = out_dir / 'chunks' / f'lemur_{split}_{source_tag}'
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if emb_path.exists():
        print(f"  Already exists: {emb_path.name} — skipping")
        return

    if split == 'train':
        recorders = [
            ('s4a',    sorted(LEMUR_DIR.glob('Audio1/S4A*.wav')),
                       LEMUR_DIR / 'Annotations1'),
            ('swift1', sorted(LEMUR_DIR.glob('Audio1/swift1*.wav')),
                       LEMUR_DIR / 'Annotations1'),
        ]
    else:
        recorders = [
            ('swift2', sorted(LEMUR_DIR.glob('Audio3/swift2*.wav')),
                       LEMUR_DIR / 'Annotations3'),
        ]

    model      = load_birdnet(device)
    all_embs   = []
    all_labels = []
    all_meta   = []

    for recorder, audio_files, ann_dir in recorders:
        print(f"\n  Recorder: {recorder} ({len(audio_files)} files)")
        for audio_path in audio_files:
            svl_path = ann_dir / (audio_path.stem + '.svl')
            if not svl_path.exists():
                print(f"    WARNING: no annotation for {audio_path.name}")
                continue
            orig_path = get_audio_path_lemur(
                audio_path.name, recorder, 'original', bitrate)
            info      = sf.info(str(orig_path if orig_path.exists() else audio_path))
            native_sr = info.samplerate
            duration  = info.duration
            annotations = parse_svl(svl_path, native_sr)
            if annotations.empty:
                print(f"    WARNING: no annotations in {svl_path.name}")
                continue
            src_path = get_audio_path_lemur(
                audio_path.name, recorder, source, bitrate)
            if not src_path.exists():
                print(f"    WARNING: missing {src_path.name}")
                continue

            chunk_path      = chunk_dir / f'{audio_path.stem}.npy'
            chunk_meta_path = chunk_dir / f'{audio_path.stem}_meta.csv'
            if chunk_path.exists():
                print(f"    {audio_path.name}: already done, loading")
                all_embs.append(np.load(chunk_path))
                all_labels.extend(pd.read_csv(chunk_meta_path)['label'].tolist())
                all_meta.append(pd.read_csv(chunk_meta_path))
                continue

            file_embs   = []
            file_labels = []
            file_meta   = []
            window_start = 0.0
            audio_batch  = []
            window_info  = []

            while window_start + BIRDNET_WIN <= duration:
                window_end = window_start + BIRDNET_WIN
                label = label_window(window_start, window_end, annotations)
                if label != -1:
                    try:
                        audio = load_window(src_path, window_start)
                        audio_batch.append((audio, BIRDNET_SR))
                        window_info.append({
                            'filename'    : audio_path.name,
                            'recorder'    : recorder,
                            'window_start': window_start,
                            'label'       : label,
                        })
                    except Exception as e:
                        print(f"      WARNING: window {window_start:.1f}s: {e}")
                window_start += BIRDNET_WIN
                if len(audio_batch) >= CHUNK_SIZE:
                    embs = extract_embeddings(model, audio_batch, device)
                    file_embs.append(embs)
                    file_labels.extend([w['label'] for w in window_info])
                    file_meta.extend(window_info)
                    audio_batch = []
                    window_info = []
                    gc.collect()

            if audio_batch:
                embs = extract_embeddings(model, audio_batch, device)
                file_embs.append(embs)
                file_labels.extend([w['label'] for w in window_info])
                file_meta.extend(window_info)

            if file_embs:
                file_embs_arr = np.concatenate(file_embs, axis=0)
                file_meta_df  = pd.DataFrame(file_meta)
                np.save(chunk_path, file_embs_arr)
                file_meta_df.to_csv(chunk_meta_path, index=False)
                all_embs.append(file_embs_arr)
                all_labels.extend(file_labels)
                all_meta.append(file_meta_df)
                print(f"    {audio_path.name}: {len(file_labels)} windows "
                      f"({sum(1 for l in file_labels if l==1)} roar, "
                      f"{sum(1 for l in file_labels if l==0)} no-roar)")
            gc.collect()

    if all_embs:
        embeddings = np.concatenate(all_embs, axis=0)
        labels     = np.array(all_labels, dtype=np.float32)
        meta_df    = pd.concat(all_meta, ignore_index=True)
        np.save(emb_path, embeddings)
        print(f"\n  Saved: {emb_path.name} {embeddings.shape}")
        if not lbl_path.exists():
            np.save(lbl_path, labels)
            meta_df.to_csv(meta_path, index=False)
            print(f"  Saved: {lbl_path.name}, {meta_path.name}")
    else:
        print("  WARNING: no embeddings extracted")

def extract_lemur_all(source: str, bitrate, device: str, out_dir: Path):
    """
    Extract BirdNET embeddings for all Lemur recordings from all recorders.
    All annotated windows are extracted (no recorder-based split).
    Split into train/test can be done later by indexing into metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    source_tag = 'original' if source == 'original' else f'{source}{bitrate}'
    emb_path   = out_dir / f'lemur_all_{source_tag}_embeddings.npy'
    lbl_path   = out_dir / 'lemur_all_labels.npy'
    meta_path  = out_dir / 'lemur_all_metadata.csv'
    chunk_dir  = out_dir / 'chunks' / f'lemur_all_{source_tag}'
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if emb_path.exists():
        print(f"  Already exists: {emb_path.name} — skipping")
        return

    # All three recorders
    recorders = [
        ('s4a',    sorted(LEMUR_DIR.glob('Audio1/S4A*.wav')),
                   LEMUR_DIR / 'Annotations1'),
        ('swift1', sorted(LEMUR_DIR.glob('Audio1/swift1*.wav')),
                   LEMUR_DIR / 'Annotations1'),
        ('swift2', sorted(LEMUR_DIR.glob('Audio3/swift2*.wav')),
                   LEMUR_DIR / 'Annotations3'),
    ]

    model      = load_birdnet(device)
    all_embs   = []
    all_labels = []
    all_meta   = []

    for recorder, audio_files, ann_dir in recorders:
        print(f"\n  Recorder: {recorder} ({len(audio_files)} files)")
        for audio_path in audio_files:
            svl_path = ann_dir / (audio_path.stem + '.svl')
            if not svl_path.exists():
                print(f"    WARNING: no annotation for {audio_path.name}")
                continue
            orig_path = get_audio_path_lemur(
                audio_path.name, recorder, 'original', bitrate)
            info      = sf.info(str(orig_path if orig_path.exists() else audio_path))
            native_sr = info.samplerate
            duration  = info.duration
            annotations = parse_svl(svl_path, native_sr)
            if annotations.empty:
                print(f"    WARNING: no annotations in {svl_path.name}")
                continue
            src_path = get_audio_path_lemur(
                audio_path.name, recorder, source, bitrate)
            if not src_path.exists():
                print(f"    WARNING: missing {src_path.name}")
                continue

            chunk_path      = chunk_dir / f'{audio_path.stem}.npy'
            chunk_meta_path = chunk_dir / f'{audio_path.stem}_meta.csv'
            if chunk_path.exists():
                print(f"    {audio_path.name}: already done, loading")
                all_embs.append(np.load(chunk_path))
                chunk_meta_df = pd.read_csv(chunk_meta_path)
                all_labels.extend(chunk_meta_df['label'].tolist())
                all_meta.append(chunk_meta_df)
                continue

            file_embs   = []
            file_labels = []
            file_meta   = []
            window_start = 0.0
            audio_batch  = []
            window_info  = []

            while window_start + BIRDNET_WIN <= duration:
                window_end = window_start + BIRDNET_WIN
                label = label_window(window_start, window_end, annotations)
                if label != -1:
                    try:
                        audio = load_window(src_path, window_start)
                        audio_batch.append((audio, BIRDNET_SR))
                        window_info.append({
                            'filename'    : audio_path.name,
                            'recorder'    : recorder,
                            'window_start': window_start,
                            'label'       : label,
                        })
                    except Exception as e:
                        print(f"      WARNING: window {window_start:.1f}s: {e}")
                window_start += BIRDNET_WIN
                if len(audio_batch) >= CHUNK_SIZE:
                    embs = extract_embeddings(model, audio_batch, device)
                    file_embs.append(embs)
                    file_labels.extend([w['label'] for w in window_info])
                    file_meta.extend(window_info)
                    audio_batch = []
                    window_info = []
                    gc.collect()

            if audio_batch:
                embs = extract_embeddings(model, audio_batch, device)
                file_embs.append(embs)
                file_labels.extend([w['label'] for w in window_info])
                file_meta.extend(window_info)

            if file_embs:
                file_embs_arr = np.concatenate(file_embs, axis=0)
                file_meta_df  = pd.DataFrame(file_meta)
                np.save(chunk_path, file_embs_arr)
                file_meta_df.to_csv(chunk_meta_path, index=False)
                all_embs.append(file_embs_arr)
                all_labels.extend(file_labels)
                all_meta.append(file_meta_df)
                print(f"    {audio_path.name}: {len(file_labels)} windows "
                      f"({sum(1 for l in file_labels if l==1)} roar, "
                      f"{sum(1 for l in file_labels if l==0)} no-roar)")
            gc.collect()

    if all_embs:
        embeddings = np.concatenate(all_embs, axis=0)
        labels     = np.array(all_labels, dtype=np.float32)
        meta_df    = pd.concat(all_meta, ignore_index=True)
        np.save(emb_path, embeddings)
        print(f"\n  Saved: {emb_path.name} {embeddings.shape}")
        if not lbl_path.exists():
            np.save(lbl_path, labels)
            meta_df.to_csv(meta_path, index=False)
            print(f"  Saved: {lbl_path.name}, {meta_path.name}")
    else:
        print("  WARNING: no embeddings extracted")


# ---------------------------------------------------------------------------
# NORTHEASTERN EXTRACTION
# ---------------------------------------------------------------------------
def extract_northeastern(source: str, bitrate, device: str, out_dir: Path):
    """
    Extract BirdNET embeddings for all Northeastern US Soundscape recordings.
    All 3-second windows are extracted (no annotation filtering).
    Labels are multilabel binary arrays (n_windows, n_species).
    Split into train/test can be done later by indexing into metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    source_tag = 'original' if source == 'original' else f'{source}{bitrate}'
    emb_path   = out_dir / f'northeastern_all_{source_tag}_embeddings.npy'
    lbl_path   = out_dir / 'northeastern_all_labels.npy'
    meta_path  = out_dir / 'northeastern_all_metadata.csv'
    sp_path    = out_dir / 'northeastern_species_cols.json'
    chunk_dir  = out_dir / 'chunks' / f'northeastern_all_{source_tag}'
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if emb_path.exists():
        print(f"  Already exists: {emb_path.name} — skipping")
        return

    # Load annotations and species map
    ann = pd.read_csv(NORTHEASTERN_DIR / 'annotations.csv')
    ann = ann.rename(columns={
        'Filename'          : 'filename',
        'Start Time (s)'    : 'ann_start',
        'End Time (s)'      : 'ann_end',
        'Low Freq (Hz)'     : 'low_freq',
        'High Freq (Hz)'    : 'high_freq',
        'Species eBird Code': 'ebird_code',
    })
    ann['filename'] = ann['filename'].str.replace('.flac', '.wav', regex=False)

    species_map     = pd.read_csv(NORTHEASTERN_DIR / 'species.csv')
    ebird_to_common = dict(zip(species_map['Species eBird Code'],
                               species_map['Common Name']))
    ann['common_name'] = ann['ebird_code'].map(ebird_to_common)
    ann = ann.dropna(subset=['common_name'])
    ann = ann[ann['common_name'] != 'unknown']

    # Species list — 80 intersection species, sorted
    species_list = sorted(ann['common_name'].unique().tolist())
    print(f"  Species: {len(species_list)}")

    # Save species list once
    if not sp_path.exists():
        with open(sp_path, 'w') as f:
            json.dump(species_list, f, indent=2)
        print(f"  Saved: {sp_path.name}")

    # Build annotation lookup per file
    ann_by_file = {fn: grp for fn, grp in ann.groupby('filename')}

    # Get all audio files
    audio_dir  = NORTHEASTERN_DIR / 'soundscape_data'
    wav_files  = sorted(audio_dir.glob('*.wav'))
    print(f"  Audio files: {len(wav_files)}")

    model      = load_birdnet(device)
    all_embs   = []
    all_labels = []
    all_meta   = []

    for audio_path in wav_files:
        filename = audio_path.name

        # Resolve source path
        if source == 'original':
            src_path = audio_path
        else:
            src_path = get_audio_path_northeastern(filename, source, bitrate)

        if not src_path.exists():
            print(f"  WARNING: missing {src_path.name}")
            continue

        chunk_path      = chunk_dir / f'{audio_path.stem}.npy'
        chunk_meta_path = chunk_dir / f'{audio_path.stem}_meta.csv'

        if chunk_path.exists():
            print(f"  {filename}: already done, loading")
            all_embs.append(np.load(chunk_path))
            chunk_meta_df = pd.read_csv(chunk_meta_path)
            all_labels.append(chunk_meta_df[[f'sp_{i}' for i in range(len(species_list))]].values)
            all_meta.append(chunk_meta_df[['filename', 'window_start', 'site', 'date']])
            continue

        # Get file duration from original
        orig_path = audio_path if source == 'original' else \
                    NORTHEASTERN_DIR / 'soundscape_data' / filename
        try:
            info     = sf.info(str(orig_path))
            duration = info.duration
        except Exception as e:
            print(f"  WARNING: could not read {filename}: {e}")
            continue

        site = get_northeastern_site(filename)
        date = get_northeastern_date(filename)

        file_embs   = []
        file_labels = []
        file_meta   = []
        audio_batch = []
        window_info = []

        window_start = 0.0
        while window_start + BIRDNET_WIN <= duration:
            window_end = window_start + BIRDNET_WIN
            # Multilabel: all species
            label = label_window_northeastern(
                window_start, window_end, filename,
                ann_by_file, species_list)
            try:
                audio = load_window(src_path, window_start)
                audio_batch.append((audio, BIRDNET_SR))
                window_info.append({
                    'filename'    : filename,
                    'window_start': window_start,
                    'site'        : site,
                    'date'        : date,
                    'label'       : label,  # numpy array
                })
            except Exception as e:
                print(f"    WARNING: window {window_start:.1f}s: {e}")

            window_start += BIRDNET_WIN

            if len(audio_batch) >= CHUNK_SIZE:
                embs = extract_embeddings(model, audio_batch, device)
                file_embs.append(embs)
                file_labels.extend([w['label'] for w in window_info])
                file_meta.extend([{k: v for k, v in w.items()
                                   if k != 'label'} for w in window_info])
                audio_batch = []
                window_info = []
                gc.collect()

        if audio_batch:
            embs = extract_embeddings(model, audio_batch, device)
            file_embs.append(embs)
            file_labels.extend([w['label'] for w in window_info])
            file_meta.extend([{k: v for k, v in w.items()
                               if k != 'label'} for w in window_info])

        if file_embs:
            file_embs_arr    = np.concatenate(file_embs, axis=0)
            file_labels_arr  = np.stack(file_labels, axis=0)
            file_meta_df     = pd.DataFrame(file_meta)
            # Store labels as columns in meta for easy resumption
            for i, sp in enumerate(species_list):
                file_meta_df[f'sp_{i}'] = file_labels_arr[:, i]

            np.save(chunk_path, file_embs_arr)
            file_meta_df.to_csv(chunk_meta_path, index=False)

            all_embs.append(file_embs_arr)
            all_labels.append(file_labels_arr)
            all_meta.append(file_meta_df[['filename', 'window_start', 'site', 'date']])

            n_pos = int(file_labels_arr.sum(axis=1).astype(bool).sum())
            print(f"  {filename}: {len(file_embs_arr)} windows "
                  f"({n_pos} with at least 1 positive)")
        gc.collect()

    if all_embs:
        embeddings = np.concatenate(all_embs, axis=0)
        labels     = np.concatenate(all_labels, axis=0)
        meta_df    = pd.concat(all_meta, ignore_index=True)

        np.save(emb_path, embeddings)
        print(f"\n  Saved: {emb_path.name} {embeddings.shape}")

        # Save labels and metadata once (same for all sources)
        if not lbl_path.exists():
            np.save(lbl_path, labels)
            meta_df.to_csv(meta_path, index=False)
            print(f"  Saved: {lbl_path.name} {labels.shape}")
            print(f"  Saved: {meta_path.name}")
    else:
        print("  WARNING: no embeddings extracted")


# # ---------------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(
#         description='Extract BirdNET embeddings for downstream evaluation')
#     parser.add_argument('--dataset', required=True,
#                         choices=['anuraset', 'lemur', 'northeastern'])
#     parser.add_argument('--split',   required=True,
#                         choices=['train', 'test', 'all'],
#                         help='all = extract all files (Northeastern only)')
#     parser.add_argument('--source',  required=True,
#                         choices=['original', 'encodec', 'dac', 'mp3', 'opus'])
#     parser.add_argument('--bitrate', default=None,
#                         help='Codec bitrate (e.g. 6.0 for EnCodec, 9 for DAC)')
#     parser.add_argument('--device',  default='CPU',
#                         choices=['CPU', 'GPU'])
#     args = parser.parse_args()

#     if args.source != 'original' and args.bitrate is None:
#         parser.error('--bitrate required when --source is not original')
#     if args.dataset != 'northeastern' and args.split == 'all':
#         parser.error('--split all is only valid for northeastern')
    

#     bitrate = None
#     if args.bitrate is not None:
#         bitrate = float(args.bitrate) if '.' in args.bitrate else int(args.bitrate)

#     print(f"{'='*60}")
#     print(f"BirdNET embedding extraction")
#     print(f"  dataset: {args.dataset}")
#     print(f"  split:   {args.split}")
#     print(f"  source:  {args.source}"
#           + (f" @ {bitrate}" if bitrate is not None else ""))
#     print(f"  device:  {args.device}")
#     print(f"{'='*60}")

#     if args.dataset == 'anuraset':
#         extract_anuraset(args.split, args.source, bitrate,
#                          args.device, ANURASET_OUT_DIR)
#     elif args.dataset == 'lemur':
#         extract_lemur(args.split, args.source, bitrate,
#                       args.device, LEMUR_OUT_DIR)
#     else:  # northeastern
#         extract_northeastern(args.source, bitrate,
#                              args.device, NORTHEASTERN_OUT_DIR)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Extract BirdNET embeddings for downstream evaluation')
    parser.add_argument('--dataset', required=True,
                        choices=['anuraset', 'lemur', 'northeastern'])
    parser.add_argument('--split',   required=True,
                        choices=['train', 'test', 'all'],
                        help='all = extract all files (Northeastern and Lemur only)')
    parser.add_argument('--source',  required=True,
                        choices=['original', 'encodec', 'dac', 'mp3', 'opus'])
    parser.add_argument('--bitrate', default=None,
                        help='Codec bitrate (e.g. 6.0 for EnCodec, 9 for DAC)')
    parser.add_argument('--device',  default='CPU',
                        choices=['CPU', 'GPU'])
    args = parser.parse_args()

    if args.source != 'original' and args.bitrate is None:
        parser.error('--bitrate required when --source is not original')
    if args.dataset == 'anuraset' and args.split == 'all':
        parser.error('--split all is not valid for anuraset')

    bitrate = None
    if args.bitrate is not None:
        bitrate = float(args.bitrate) if '.' in args.bitrate else int(args.bitrate)

    print(f"{'='*60}")
    print(f"BirdNET embedding extraction")
    print(f"  dataset: {args.dataset}")
    print(f"  split:   {args.split}")
    print(f"  source:  {args.source}"
          + (f" @ {bitrate}" if bitrate is not None else ""))
    print(f"  device:  {args.device}")
    print(f"{'='*60}")

    if args.dataset == 'anuraset':
        extract_anuraset(args.split, args.source, bitrate,
                         args.device, ANURASET_OUT_DIR)
    elif args.dataset == 'lemur':
        if args.split == 'all':
            extract_lemur_all(args.source, bitrate,
                              args.device, LEMUR_OUT_DIR)
        else:
            extract_lemur(args.split, args.source, bitrate,
                          args.device, LEMUR_OUT_DIR)
    else:  # northeastern
        extract_northeastern(args.source, bitrate,
                             args.device, NORTHEASTERN_OUT_DIR)

if __name__ == '__main__':
    main()