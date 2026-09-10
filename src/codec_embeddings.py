"""
src/embeddings.py

Saves post-quantization discrete codes (tokens) per 30-second chunk,
for both DAC and EnCodec. Codes are taken directly from what the codec
methods already return — no extra forward passes.

Extraction functions (extract_dac_embeddings, extract_encodec_embeddings)
are kept for future use. Current implementation does a separate forward pass.

CSV schema (one row per quantizer level):
    filename, dataset, codec, bitrate, chunk_id, embedding_type, level, t0, t1, ..., tT

embedding_type = 'post_quant' for now. Column kept for future types
(e.g. 'post_encoding', 'latents') without schema changes.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# SAVING

def save_codes_to_csv(codes: torch.Tensor,
                      filename: str,
                      dataset: str,
                      codec: str,
                      bitrate,
                      chunk_id: int,
                      output_path: str,
                      embedding_type: str = 'post_quant') -> None:
    """
    Save a codes tensor [1, N_q, T] as rows in a CSV.
    One row per quantizer level (N_q rows total per chunk).
    T frames become columns t0, t1, ..., tT stretching to the right.
    N_q levels become rows stretching downwards.

    Columns: filename, dataset, codec, bitrate, chunk_id,
             embedding_type, level, t0, t1, ..., tT

    embedding_type defaults to 'post_quant'. Pass a different value
    if saving other embedding types in the future (e.g. 'latents').
    """
    arr    = codes[0].numpy()  # [N_q, T]
    n_q, T = arr.shape

    rows = []
    for q in range(n_q):
        row = {
            'filename'      : filename,
            'dataset'       : dataset,
            'codec'         : codec,
            'bitrate'       : bitrate,
            'chunk_id'      : chunk_id,
            'embedding_type': embedding_type,
            'level'         : q,
        }
        for t, v in enumerate(arr[q]):
            row[f't{t}'] = int(v)
        rows.append(row)

    df          = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)


def save_embedding_to_csv(tensor: torch.Tensor,
                          filename: str,
                          dataset: str,
                          codec: str,
                          bitrate,
                          chunk_id: int,
                          embedding_type: str,
                          output_path: str) -> None:
    """
    Save any [1, C, T] tensor as rows in a CSV, one row per channel/level.
    Works for both continuous (e.g. post_encoding, latents) and discrete
    (post_quant) tensors. Values stored as float for continuous, int for
    post_quant.

    Columns: filename, dataset, codec, bitrate, chunk_id, embedding_type,
             level, t0, t1, ..., tT
    """
    arr   = tensor[0].numpy()  # [C, T]
    C, T  = arr.shape
    is_int = embedding_type == 'post_quant'

    rows = []
    for c in range(C):
        row = {
            'filename'      : filename,
            'dataset'       : dataset,
            'codec'         : codec,
            'bitrate'       : bitrate,
            'chunk_id'      : chunk_id,
            'embedding_type': embedding_type,
            'level'         : c,
        }
        values = arr[c]
        if is_int:
            for t, v in enumerate(values):
                row[f't{t}'] = int(v)
        else:
            for t, v in enumerate(values):
                row[f't{t}'] = round(float(v), 6)
        rows.append(row)

    df          = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)


def save_chunk_embeddings(embeddings: dict,
                          filename: str,
                          dataset: str,
                          codec: str,
                          bitrate,
                          chunk_id: int,
                          output_path: str) -> None:
    """
    Convenience wrapper — saves all embedding types from an
    extract_*_embeddings() result dict to the same CSV.
    Loops over dict keys so it works for any combination of types.
    """
    for embedding_type, tensor in embeddings.items():
        save_embedding_to_csv(tensor, filename, dataset, codec,
                              bitrate, chunk_id, embedding_type, output_path)


# EXTRACTION

def extract_dac_embeddings(chunk: np.ndarray,
                           sr: int,
                           model,
                           n_quantizers: int,
                           device: str) -> dict:
    """
    Extract embeddings from DAC via a separate model.encode() forward pass.
    NOTE: preprocessing differs from model.compress() (no loudness normalisation,
    no delay zero-padding): embeddings do not exactly correspond to the
    internal representations during reconstruction. Use with awareness of
    this limitation. Intended for exploratory analysis in 06_chunk_exploration.

    Returns:
        latents   : projected codebook inputs [1, N*D_cb, T]
        post_quant: discrete codes            [1, N, T]
    """
    from audiotools import AudioSignal

    signal       = AudioSignal(chunk, sample_rate=sr).to(device)
    audio_padded = model.preprocess(signal.audio_data, signal.sample_rate)

    with torch.inference_mode():
        _, codes, latents, _, _ = model.encode(audio_padded, n_quantizers=n_quantizers)

    return {
        'latents'   : latents.cpu(),  # [1, N*D_cb, T]
        'post_quant': codes.cpu(),    # [1, N, T]
    }


def extract_encodec_embeddings(chunk: np.ndarray,
                               sr: int,
                               model,
                               device: str) -> dict:
    """
    Extract embeddings from EnCodec by calling encoder and quantizer submodules
    directly. Note: assumes model.segment is None (24kHz default). Preprocessing
    differs from model.encode() if normalization is applied. Intended for
    exploratory analysis in 06_chunk_exploration.

    Returns:
        post_encoding: raw encoder output [1, D, T]
        post_quant:    discrete codes     [1, N_q, T]
    """
    assert model.segment is None, \
        "extract_encodec_embeddings assumes segment=None (24kHz model default)"

    with torch.inference_mode():
        x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Replicate _encode_frame normalization
        if model.normalize:
            mono   = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            x      = x / (1e-8 + volume)

        emb   = model.encoder(x)
        codes = model.quantizer.encode(emb, model.frame_rate, model.bandwidth)
        codes = codes.transpose(0, 1)  # [1, N_q, T]

    return {
        'post_encoding': emb.cpu(),
        'post_quant'   : codes.cpu(),
    }