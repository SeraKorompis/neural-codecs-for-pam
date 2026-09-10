import gc
import time
from src.utils import resample
from src.codec_embeddings import save_embedding_to_csv
import numpy as np
import torch
from typing import Optional

# Global model cache to prevent reloading models on every single iteration
_MODEL_CACHE = {}

def get_cached_model(codec_name: str, sr: int, device: str, bandwidth: Optional[float] = None):
    """
    Helper to load and cache models so they are only initialized once.
    Loading a model is expensive and can cause Out of Memory errors if done repeatedly in a loop.
    This function returns a cached model if it has already been loaded for the given (codec, sr, device) combination.
    """
    cache_key = f"{codec_name}_{sr}_{device}"
    if codec_name == 'encodec':
        cache_key += f"_{bandwidth}"
        
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key] # Return cached model if already loaded

    # Load EnCodec model
    if codec_name == 'encodec': 
        from encodec import EncodecModel
        if sr == 24000:
            model = EncodecModel.encodec_model_24khz()
        else:
            model = EncodecModel.encodec_model_48khz()
        model.set_target_bandwidth(bandwidth)
        model.eval()
        model = model.to(device)
        _MODEL_CACHE[cache_key] = model  
        print(f"  [Encodec] Loaded model to {device} for SR {sr}")
        return model

    # Load DAC model
    elif codec_name == 'dac':
        import dac
        model_type_map = {16000: '16khz', 24000: '24khz', 44100: '44khz'}
        model_path = dac.utils.download(model_type=model_type_map[sr])
        model = dac.DAC.load(model_path)
        model.eval()
        model = model.to(device)
        _MODEL_CACHE[cache_key] = model
        print(f"  [DAC] Loaded model to {device} for SR {sr}")
        return model

    raise ValueError(f"Unknown codec: {codec_name}")


# CHUNKED CODEC FUNCTIONS
#
# Root cause of OOM on long files:
#   EncodecModel.encode() slices from `x` AFTER it is on GPU:
#       frame = x[:, :, offset: offset + segment_length]
#   So setting model.segment does NOT save VRAM, the full tensor must
#   already be on the GPU before any segmentation happens.
#
#   DAC's model.compress(win_duration=1.0) chunks internally, but
#   AudioSignal(...).to(device) still moves the full signal to GPU first.
#
#   Both codecs therefore require external Python-level chunking:
#   only one chunk is converted to a GPU tensor at a time, keeping peak
#   VRAM = model weights + one chunk (3-5 MB at 30 s) regardless of
#   file length. This also eliminates CUDA allocator fragmentation that
#   accumulates across hundreds of sequential files.
#
# Chunk boundary note:
#   Each 30 s chunk is encoded/decoded independently (no cross-chunk
#   context). This introduces hard boundaries every 30 s. For a 60-min
#   file these affect <0.01% of samples and have negligible impact on
#   SI-SNR or STFT distance averaged over the recording or call regions.
#   Disclose in methods as: "audio processed in 30-second chunks".

_CHUNK_SEC = 30.0  # seconds per GPU chunk


def _save_ecdc(model, full_codes: torch.Tensor,
               audio_length: int, save_path: str) -> None:
    """Write a .ecdc file from a pre-computed codes tensor.
    Uses EnCodec binary module directly (internal API).
    Format: ECDC header (JSON metadata) + bit-packed codes.
    """
    from encodec import binary
    import io, os
    from pathlib import Path as _Path

    fo = io.BytesIO()
    metadata = {
        'm' : model.name,
        'al': audio_length,
        'nc': full_codes.shape[1],   # n_q
        'lm': False,
    }
    binary.write_ecdc_header(fo, metadata)

    packer   = binary.BitPacker(model.bits_per_codebook, fo)
    codes_np = full_codes[0].cpu().numpy()   # [n_q, T]
    n_q, T   = codes_np.shape

    for t in range(T):
        for k in range(n_q):
            packer.push(int(codes_np[k, t]))
        packer.flush()

    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
    _Path(save_path).write_bytes(fo.getvalue())


@torch.inference_mode()
def compress_encodec(audio: np.ndarray,
                     sr: int,
                     bandwidth: float = 6.0,
                     device: Optional[str] = None,
                     save_path: Optional[str] = None,
                     embeddings_csv_path: Optional[str] = None,
                     embeddings_filename: Optional[str] = None,
                     embeddings_dataset: Optional[str] = None,
                     ) -> tuple[np.ndarray, float, float, float]:
    """
    Returns (recon, nominal_kbps, encode_time_sec, decode_time_sec).
    encode_time_sec / decode_time_sec are summed across all chunks.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert sr in [24000, 48000], f"Encodec requires sr in [24000, 48000], got {sr}"

    model = get_cached_model('encodec', sr, device, bandwidth)

    chunk_samples  = int(_CHUNK_SEC * sr)
    bits_per_cb    = model.bits_per_codebook
    recon_chunks   = []
    all_codes_list = [] if save_path is not None else None

    encode_time_sec = 0.0
    decode_time_sec = 0.0
    chunk_id        = 0

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        wav   = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Time encode (compress) for this chunk
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoded_frames = model.encode(wav)
        if device == 'cuda':
            torch.cuda.synchronize()
        encode_time_sec += time.perf_counter() - t0

        # Time decode (decompress) for this chunk
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        recon_chunk = model.decode(encoded_frames).squeeze().cpu().numpy()
        if device == 'cuda':
            torch.cuda.synchronize()
        decode_time_sec += time.perf_counter() - t0

        recon_chunks.append(recon_chunk[:len(chunk)])

        if all_codes_list is not None:
            chunk_codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
            all_codes_list.append(chunk_codes.cpu())

        # Save post-quantization codes directly from encoded_frames —
        # no extra forward pass needed, codes already computed above
        if embeddings_csv_path is not None:
            try:
                chunk_codes_save = torch.cat([f[0] for f in encoded_frames], dim=-1).cpu()
                save_embedding_to_csv(chunk_codes_save, embeddings_filename, embeddings_dataset,
                                      'encodec', bandwidth, chunk_id,
                                      'post_quant', embeddings_csv_path)
            except Exception as e:
                print(f"  [EnCodec] Could not save codes for chunk {chunk_id}: {e}")
            chunk_id += 1

        del wav, encoded_frames, recon_chunk

    # Nominal kbps: for EnCodec the bandwidth parameter IS the nominal bitrate.
    # Equivalent to: n_q_active * frame_rate * bits_per_codebook / 1000
    # where n_q_active = int(bandwidth * 1000 / (frame_rate * bits_per_codebook))
    nominal_kbps = float(bandwidth)

    # Save .ecdc file if requested
    if save_path is not None and all_codes_list:
        full_codes = torch.cat(all_codes_list, dim=-1)   # [1, n_q, T_total]
        try:
            _save_ecdc(model, full_codes, len(audio), save_path)
        except Exception as e:
            print(f"  [EnCodec] Could not save .ecdc file: {e}")
        finally:
            del full_codes, all_codes_list

    recon = np.concatenate(recon_chunks)
    del recon_chunks
    return recon, nominal_kbps, encode_time_sec, decode_time_sec


@torch.inference_mode()
def compress_dac(audio: np.ndarray,
                 sr: int,
                 n_quantizers: Optional[int] = None,
                 device: Optional[str] = None,
                 save_path: Optional[str] = None,
                 embeddings_csv_path: Optional[str] = None,
                 embeddings_filename: Optional[str] = None,
                 embeddings_dataset: Optional[str] = None,
                 use_encode_decode: bool = False,
                 ) -> tuple[np.ndarray, float, float, float]:
    """
    Returns (recon, nominal_kbps, encode_time_sec, decode_time_sec).
    encode_time_sec / decode_time_sec are summed across all chunks.

    use_encode_decode=False (default): model.compress(win_duration=1.0) +
        model.decompress(). Internal 1s windowing with delay zero-padding.
        Saves codes only (post_quant).
    use_encode_decode=True: model.encode() + model.decode() directly.
        No internal windowing — each 30s chunk processed as one single pass.
        Saves z (post_quant_z), codes (post_quant), latents in one pass.
    """
    from audiotools import AudioSignal

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert sr in [16000, 24000, 44100], f"DAC requires sr in [16000, 24000, 44100], got {sr}"

    model = get_cached_model('dac', sr, device)

    chunk_samples     = int(_CHUNK_SEC * sr)
    bits_per_cb       = int(np.log2(model.codebook_size))
    recon_chunks      = []
    n_quantizers_used = None

    # Compute global loudness of the entire recording before chunking.
    # Used as a single consistent input_db for all chunks and the saved
    # DACFile, ensuring the .dac file decodes to the same audio as the
    # per-chunk .wav reconstruction.
    global_input_db = AudioSignal(audio, sample_rate=sr).loudness()

    # Only collect codes tensors when saving a .dac file
    all_codes_list = [] if save_path is not None else None
    first_meta     = None

    encode_time_sec = 0.0
    decode_time_sec = 0.0
    chunk_id        = 0

    for start in range(0, len(audio), chunk_samples):
        chunk  = audio[start : start + chunk_samples]
        signal = AudioSignal(chunk, sample_rate=sr).to(device)

        if not use_encode_decode:
            # ── compress/decompress mode (default) ───────────────────────────
            # model.compress() handles internal 1s windowing + delay zero-padding
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dac_file = model.compress(signal, win_duration=1.0,
                                      n_quantizers=n_quantizers, normalize_db=None)
            if device == 'cuda':
                torch.cuda.synchronize()
            encode_time_sec += time.perf_counter() - t0

            dac_file.input_db = global_input_db

            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            recon_chunk = model.decompress(dac_file).audio_data.squeeze().cpu().numpy()
            if device == 'cuda':
                torch.cuda.synchronize()
            decode_time_sec += time.perf_counter() - t0

            recon_chunks.append(recon_chunk[:len(chunk)])

            codes = dac_file.codes
            if n_quantizers_used is None:
                n_quantizers_used = codes.shape[1]

            if all_codes_list is not None:
                all_codes_list.append(codes.cpu())

            if first_meta is None:
                first_meta = {
                    'channels'    : dac_file.channels,
                    'sample_rate' : dac_file.sample_rate,
                    'chunk_length': dac_file.chunk_length,
                }

            # Save codes (post_quant), directly from dac_file, no extra pass
            if embeddings_csv_path is not None:
                try:
                    save_embedding_to_csv(dac_file.codes.cpu(), embeddings_filename,
                                          embeddings_dataset, 'dac', n_quantizers_used,
                                          chunk_id, 'post_quant_codes', embeddings_csv_path)
                except Exception as e:
                    print(f"  [DAC] Could not save codes for chunk {chunk_id}: {e}")
                chunk_id += 1

            del signal, dac_file, recon_chunk, codes

        else:
            # ── encode/decode mode ───────────────────────────────────────────
            # Calls model.encoder() + model.quantizer() + model.decode() directly.
            # No internal 1s windowing — each 30s chunk processed as one pass.
            # Saves z (post_quant_z), codes (post_quant), latents in one pass.
            audio_padded = model.preprocess(signal.audio_data, signal.sample_rate)

            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            z, codes, latents, _, _ = model.encode(audio_padded, n_quantizers=n_quantizers)
            if device == 'cuda':
                torch.cuda.synchronize()
            encode_time_sec += time.perf_counter() - t0

            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            recon_chunk = model.decode(z).squeeze().cpu().numpy()
            if device == 'cuda':
                torch.cuda.synchronize()
            decode_time_sec += time.perf_counter() - t0

            recon_chunks.append(recon_chunk[:len(chunk)])

            if n_quantizers_used is None:
                n_quantizers_used = codes.shape[1]

            if all_codes_list is not None:
                all_codes_list.append(codes.cpu())

            if first_meta is None:
                first_meta = {
                    'channels'    : 1,
                    'sample_rate' : sr,
                    'chunk_length': codes.shape[-1],  # full chunk, not 72-frame windows
                }

            # Save all three embedding types in one pass — no extra forward pass
            if embeddings_csv_path is not None:
                try:
                    save_embedding_to_csv(codes.cpu(), embeddings_filename, embeddings_dataset,
                                          'dac', n_quantizers_used, chunk_id,
                                          'post_quant_codes', embeddings_csv_path)
                    save_embedding_to_csv(z.cpu(), embeddings_filename, embeddings_dataset,
                                          'dac', n_quantizers_used, chunk_id,
                                          'post_quant_z', embeddings_csv_path)
                    save_embedding_to_csv(latents.cpu(), embeddings_filename, embeddings_dataset,
                                          'dac', n_quantizers_used, chunk_id,
                                          'latents', embeddings_csv_path)
                except Exception as e:
                    print(f"  [DAC encode/decode] Could not save embeddings for chunk {chunk_id}: {e}")
                chunk_id += 1

            del signal, audio_padded, z, codes, latents, recon_chunk

    # Nominal kbps: theoretical bitrate from architecture constants
    # Matches the DAC paper (Kumar et al., 2023) Table 1:
    # bitrate = (fs / M) * Nq * log2(C) / 1000
    nominal_kbps = n_quantizers_used * (model.sample_rate / model.hop_length) * bits_per_cb / 1000

    # Save .dac file if requested
    if save_path is not None and all_codes_list:
        import dac as dac_lib
        full_codes = torch.cat(all_codes_list, dim=-1)   # (1, n_q, total_frames)
        try:
            full_dac = dac_lib.DACFile(
                codes           = full_codes,
                chunk_length    = first_meta['chunk_length'],
                original_length = len(audio),
                input_db        = global_input_db,
                channels        = first_meta['channels'],
                sample_rate     = first_meta['sample_rate'],
                padding         = False,
                dac_version     = getattr(model, 'metadata', {}).get('version', '1.0.0'),
            )
            import os
            os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
            full_dac.save(str(save_path))
        except Exception as e:
            print(f"  [DAC] Could not save .dac file: {e}")
        finally:
            del full_codes, all_codes_list

    recon = np.concatenate(recon_chunks).astype(np.float32)
    del recon_chunks
    return recon, nominal_kbps, encode_time_sec, decode_time_sec

def compress_mp3(audio: np.ndarray, sr: int, bitrate_kbps: int = 32) -> tuple[np.ndarray, float, float, float]:
    """
    Compresses audio using MP3 (via pydub/ffmpeg).
    Returns (recon, nominal_kbps, encode_time_sec, decode_time_sec).
    """
    import io
    from pydub import AudioSegment
    
    # Convert numpy float32 [-1.0, 1.0] to int16 PCM bytes for pydub
    audio_int16 = (audio * 32767).astype(np.int16) # According to pydub docs, AudioSegment expects PCM16 data
    audio_segment = AudioSegment(
        audio_int16.tobytes(),  
        frame_rate=sr, 
        sample_width=2, 
        channels=1
    )
    
    bitrate_str = f"{bitrate_kbps}k"
    
    # Time Encoding
    t0 = time.perf_counter()
    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3", bitrate=bitrate_str)
    encode_time = time.perf_counter() - t0
    
    mp3_buffer.seek(0)
    
    # Time Decoding
    t0 = time.perf_counter()
    decoded_segment = AudioSegment.from_file(mp3_buffer, format="mp3")
    decode_time = time.perf_counter() - t0
    
    # Convert back to float32 numpy array, matching original length
    recon = np.frombuffer(decoded_segment.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
    if len(recon) > len(audio):
        recon = recon[:len(audio)]
    elif len(recon) < len(audio):
        recon = np.pad(recon, (0, len(audio) - len(recon)))
        
    return recon, float(bitrate_kbps), encode_time, decode_time


# def compress_opus(audio: np.ndarray, sr: int, bitrate_kbps: int = 6) -> tuple[np.ndarray, float, float, float]:
#     import io
#     from pydub import AudioSegment
    
#     audio_int16 = (audio * 32767).astype(np.int16)
#     audio_segment = AudioSegment(
#         audio_int16.tobytes(), 
#         frame_rate=sr,
#         sample_width=2, 
#         channels=1
#     )
    
#     bitrate_str = f"{bitrate_kbps}k"
    
#     t0 = time.perf_counter()
#     opus_buffer = io.BytesIO()
#     # Opus must be wrapped in Ogg container — 'opus' is not a valid container format
#     audio_segment.export(
#         opus_buffer,
#         format="ogg",
#         parameters=["-c:a", "libopus", "-b:a", bitrate_str]
#     )
#     encode_time = time.perf_counter() - t0
    
#     opus_buffer.seek(0)
    
#     t0 = time.perf_counter()
#     decoded_segment = AudioSegment.from_file(opus_buffer, format="ogg")
#     decode_time = time.perf_counter() - t0
    
#     recon = np.frombuffer(decoded_segment.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
#     if len(recon) > len(audio):
#         recon = recon[:len(audio)]
#     elif len(recon) < len(audio):
#         recon = np.pad(recon, (0, len(audio) - len(recon)))
        
#     return recon, float(bitrate_kbps), encode_time, decode_time

def compress_opus(audio: np.ndarray, sr: int, bitrate_kbps: int = 6) -> tuple[np.ndarray, float, float, float]:
    import subprocess
    import tempfile
    import os
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, 'input.wav')
        opus_path   = os.path.join(tmpdir, 'compressed.opus')
        output_path = os.path.join(tmpdir, 'reconstructed.wav')

        # Save input
        sf.write(input_path, audio, sr)

        # Encode to opus
        t0 = time.perf_counter()
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-c:a', 'libopus',
            '-b:a', f'{bitrate_kbps}k',
            opus_path
        ], check=True, capture_output=True)
        encode_time = time.perf_counter() - t0

        # Decode back to wav
        t0 = time.perf_counter()
        subprocess.run([
            'ffmpeg', '-y', '-i', opus_path,
            '-ar', str(sr),  # resample back to original sr
            output_path
        ], check=True, capture_output=True)
        decode_time = time.perf_counter() - t0

        recon, _ = sf.read(output_path)
        recon = recon.astype(np.float32)

    # Match length
    if len(recon) > len(audio):
        recon = recon[:len(audio)]
    elif len(recon) < len(audio):
        recon = np.pad(recon, (0, len(audio) - len(recon)))

    return recon, float(bitrate_kbps), encode_time, decode_time


# --- CODEC REGISTRY ---

CODEC_CONFIG = {
    'encodec': {
        'available_sr'  : [24000, 48000],
        'bitrates_by_sr': {
            24000: [1.5, 3.0, 6.0, 12.0, 24.0],
            48000: [3.0, 6.0, 12.0, 24.0],
        },
        'bitrates'      : [1.5, 3.0, 6.0, 12.0, 24.0],
        'fn'            : compress_encodec,
        'bitrate_kwarg' : 'bandwidth',
    },
    'dac': {
        'available_sr'   : [16000, 24000, 44100],
        'max_codebooks_by_sr': {
            16000: 12,
            24000: 32,
            44100: 9,
        },
        'bitrates_by_sr': {
            16000: [2, 4, 8, 12],
            24000: [2, 4, 8, 16, 32],
            44100: [2, 3, 6, 9],
        },
        'bitrates'      : [2, 4, 8, 16, 32],
        'fn'            : compress_dac,
        'bitrate_kwarg' : 'n_quantizers',
    },
    'mp3': {
    'available_sr' : [44100],
    'bitrates'     : [8, 16, 24, 32, 64],
    'fn'           : compress_mp3,
    'bitrate_kwarg': 'bitrate_kbps',
    },
    'opus': {
        'available_sr' : [48000],
        'bitrates'     : [6, 8, 12, 14, 24, 32],
        'fn'           : compress_opus,
        'bitrate_kwarg': 'bitrate_kbps',
    },
}


@torch.inference_mode()
def run_codec(audio: np.ndarray,
              native_sr: int,
              codec: str,
              bitrate,
              target_sr: int,
              device: Optional[str] = None,
              save_path: Optional[str] = None,
              embeddings_csv_path: Optional[str] = None,
              embeddings_filename: Optional[str] = None,
              embeddings_dataset: Optional[str] = None,
              use_encode_decode: bool = False,
              ) -> tuple[np.ndarray, float, float, float]:
    """
    Resample audio to target_sr and compress with the specified codec.
    save_path: if provided, saves the compressed representation to disk
               (.dac for DAC, .ecdc for EnCodec) for future analysis.
    embeddings_csv_path: if provided, saves per-chunk pre/post-quantization
               embeddings to this CSV (see src/embeddings.py).

    Returns (recon, nominal_kbps, encode_time_sec, decode_time_sec).
    """
    audio_resampled = resample(audio, native_sr, target_sr)
    config          = CODEC_CONFIG[codec]
    fn              = config['fn']
    bitrate_kwarg   = config['bitrate_kwarg']

    kwargs = {}
    if bitrate_kwarg is not None:
        kwargs[bitrate_kwarg] = bitrate
    # Only pass device, save_path, embeddings to neural codecs as MP3 and Opus are CPU-only
    if codec in ['encodec', 'dac']:
        if device is not None:
            kwargs['device'] = device
        if save_path is not None:
            kwargs['save_path'] = save_path
        if embeddings_csv_path is not None:
            kwargs['embeddings_csv_path'] = embeddings_csv_path
            kwargs['embeddings_filename'] = embeddings_filename
            kwargs['embeddings_dataset']  = embeddings_dataset

    # use_encode_decode only applies to DAC — EnCodec already uses encode/decode directly
    if use_encode_decode and codec == 'dac':
        kwargs['use_encode_decode'] = True

    result = fn(audio_resampled, sr=target_sr, **kwargs)

    # Post-execution cleanup: empty CUDA cache before gc.collect so the
    # allocator releases pages back to the OS immediately, not on next alloc.
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return result