"""
src/birdnet_embeddings.py

Shared BirdNET embedding extraction utilities.
Used by extract_anuraset_embeddings.py and extract_lemur_embeddings.py.

encode_arrays API (birdnet 0.2.16):
  - Takes: Iterable of (audio_ndarray, sample_rate) tuples
  - Returns: AcousticEncodingResultBase with .embeddings property
  - .embeddings shape: (n_inputs, n_segments, emb_dim)
  - For a 3s clip with no overlap: n_inputs=1, n_segments=1
  - Extract single embedding: result.embeddings[0][0]
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

BIRDNET_SR      = 48000
BIRDNET_VERSION = '2.4'
BIRDNET_BACKEND = 'pb'


def load_birdnet_model():
    """Load BirdNET v2.4 acoustic model."""
    import birdnet
    print(f"Loading BirdNET {BIRDNET_VERSION} ({BIRDNET_BACKEND} backend)...")
    model = birdnet.load('acoustic', BIRDNET_VERSION, BIRDNET_BACKEND)
    print(f"  Embedding dim: {model.get_embeddings_dim()}")
    print(f"  Sample rate:   {model.get_sample_rate()} Hz")
    print(f"  Segment size:  {model.get_segment_size_s()}s")
    return model


def load_and_resample(path: Path, target_sr: int = BIRDNET_SR) -> np.ndarray:
    """Load audio file and resample to target_sr. Returns mono float32 array."""
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def extract_embedding(model, audio: np.ndarray, sr: int = BIRDNET_SR) -> np.ndarray:
    """
    Extract BirdNET embedding from a single audio array.
    Returns 1D array of shape (embedding_dim,).
    Audio should already be at BIRDNET_SR.
    """
    result = model.encode_arrays([(audio, sr)])
    # result.embeddings shape: (n_inputs, n_segments, emb_dim)
    # For a 3s clip: n_inputs=1, n_segments=1
    return result.embeddings[0][0]


def extract_embeddings_batch(model,
                              audio_list: list,
                              sr: int = BIRDNET_SR,
                              desc: str = 'Extracting') -> np.ndarray:
    """
    Extract BirdNET embeddings from a list of audio arrays.
    Returns array of shape (n_clips, embedding_dim).
    Audio arrays should already be at BIRDNET_SR.
    """
    from tqdm import tqdm
    embeddings  = []
    failed      = 0
    first_error = None
    for audio in tqdm(audio_list, desc=desc):
        try:
            emb = extract_embedding(model, audio, sr)
            embeddings.append(emb)
        except Exception as e:
            if first_error is None:
                first_error = str(e)
                print(f"  First failure: {e}")
            embeddings.append(np.zeros(model.get_embeddings_dim()))
            failed += 1
    if failed > 0:
        print(f"  Warning: {failed} clips failed — zero embeddings used")
        print(f"  First error: {first_error}")
    return np.array(embeddings)


def extract_embeddings_from_files(model,
                                   file_list: list,
                                   desc: str = 'Extracting') -> np.ndarray:
    """
    Extract BirdNET embeddings from a list of audio file paths.
    Handles loading, resampling, and encoding.
    Returns array of shape (n_files, embedding_dim).
    """
    from tqdm import tqdm
    embeddings  = []
    failed      = 0
    first_error = None
    for path in tqdm(file_list, desc=desc):
        path = Path(path)
        if not path.exists():
            embeddings.append(np.zeros(model.get_embeddings_dim()))
            failed += 1
            continue
        try:
            audio  = load_and_resample(path, target_sr=BIRDNET_SR)
            result = model.encode_arrays([(audio, BIRDNET_SR)])
            emb    = result.embeddings[0][0]
            embeddings.append(emb)
        except Exception as e:
            if first_error is None:
                first_error = str(e)
            print(f"  Warning: {path.name}: {e}")
            embeddings.append(np.zeros(model.get_embeddings_dim()))
            failed += 1
    if failed > 0:
        print(f"  Warning: {failed} files failed — zero embeddings used")
        print(f"  First error: {first_error}")
    return np.array(embeddings)