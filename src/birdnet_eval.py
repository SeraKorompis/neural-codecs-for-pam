"""
src/birdnet_eval.py

BirdNET evaluation pipeline for bioacoustic PAM benchmark.
Runs BirdNET acoustic model on audio files with location/date-based
species filtering via the geo model.

Workflow:
    1. Load geo model once, compute expected species per unique week (cached)
    2. Load acoustic model once
    3. For each audio file:
        a. Extract date from filename
        b. Get cached location/week-filtered species list from geo model
        c. Run acoustic model with species filter
        d. Save detections to CSV (append, with skip logic)

Output CSV schema:
    filename, start_sec, end_sec, scientific_name, common_name, confidence

Usage:
    from src.birdnet_eval import run_birdnet_northeastern

    run_birdnet_northeastern(
        data_dir   = Path('data/northeastern_us_soundscapes/soundscape_data'),
        output_csv = Path('results/downstream/birdnet/northeastern_birds/detections.csv'),
        min_conf   = 0.25,
    )
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from typing import Optional


# SAPSUCKER WOODS, ITHACA, NY
NORTHEASTERN_LAT = 42.4768
NORTHEASTERN_LON = -76.4527


def _date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract recording date from Northeastern Birds filename.
    e.g. SSW_001_20170225_010000Z.wav -> datetime(2017, 2, 25)
    Returns None if date cannot be parsed.
    """
    m = re.search(r'_(\d{8})_', filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), '%Y%m%d')
    except ValueError:
        return None


def _date_to_birdnet_week(date: datetime) -> int:
    """
    Convert a datetime to BirdNET week number (1-48).
    BirdNET divides the year into 48 weeks of ~7.6 days each.
    """
    day_of_year = date.timetuple().tm_yday   # 1-365
    week        = min(48, max(1, int((day_of_year - 1) / 365 * 48) + 1))
    return week


def _parse_species_name(species_name: str) -> tuple[str, str]:
    """
    Parse BirdNET species name string into (scientific_name, common_name).
    Format: 'Poecile atricapillus_Black-capped Chickadee'
    """
    parts = species_name.split('_', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return species_name.strip(), species_name.strip()


def _build_week_species_cache(
    geo_model,
    weeks:          list[int],
    lat:            float = NORTHEASTERN_LAT,
    lon:            float = NORTHEASTERN_LON,
    min_confidence: float = 0.03,
) -> dict[int, set]:
    """
    Pre-compute geo model species sets for each unique week.
    Returns a dict mapping week number to set of species name strings.
    Avoids calling geo_model.predict() redundantly for files sharing the same week.

    Parameters
    ----------
    geo_model      : loaded birdnet geo model
    weeks          : list of unique week numbers to cache
    lat            : recording latitude
    lon            : recording longitude
    min_confidence : minimum geo model confidence for species inclusion
    """
    cache = {}
    for week in sorted(set(weeks)):
        geo_result  = geo_model.predict(lat, lon, week=week,
                                        min_confidence=min_confidence)
        cache[week] = geo_result.to_set()
        print(f"  Week {week:2d}: {len(cache[week])} expected species")
    return cache


def run_birdnet_on_file(
    audio_path:    Path,
    acoustic_model,
    geo_model,
    lat:           float = NORTHEASTERN_LAT,
    lon:           float = NORTHEASTERN_LON,
    min_conf:      float = 0.1,
    top_k:         Optional[int] = None,
    week_cache:    Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run BirdNET on a single audio file with location/date-based species filtering.

    Parameters
    ----------
    audio_path     : path to audio file
    acoustic_model : loaded birdnet acoustic model
    geo_model      : loaded birdnet geo model
    lat            : recording latitude
    lon            : recording longitude
    min_conf       : minimum confidence threshold
    top_k          : max detections per 3s window (None = all above threshold)
    week_cache     : optional dict mapping week number to species set,
                     pre-computed via _build_week_species_cache().
                     If provided, avoids calling geo_model.predict() per file.

    Returns
    -------
    DataFrame with columns: filename, start_sec, end_sec,
                             scientific_name, common_name, confidence
    """
    filename = Path(audio_path).name
    date     = _date_from_filename(filename)

    if date is None:
        print(f"  Warning: could not parse date from {filename}, using week=None")
        week = None
    else:
        week = _date_to_birdnet_week(date)

    # Get location/week-filtered species list, use cache if available
    if week_cache is not None and week in week_cache:
        species_set = week_cache[week]
    else:
        geo_result  = geo_model.predict(lat, lon, week=week, min_confidence=0.03)
        species_set = geo_result.to_set()

    # Pre-flight check: try reading a small chunk of the file before passing
    # it to BirdNET. Catches decoding issues that cause BirdNET to hang
    # indefinitely rather than raising an exception.
    # Reading 1000 frames (~30ms) is fast and reliably detects corrupt files.
    import soundfile as sf
    try:
        with sf.SoundFile(str(audio_path)) as f:
            f.read(frames=1000)
    except Exception as e:
        raise RuntimeError(f"Audio pre-flight check failed, skipping: {e}")

    # Run acoustic model
    predictions = acoustic_model.predict(
        str(audio_path),
        custom_species_list          = species_set,
        default_confidence_threshold = min_conf,
        top_k                        = top_k,
    )

    # Convert to DataFrame
    df = predictions.to_dataframe()

    if df.empty:
        return pd.DataFrame(columns=[
            'filename', 'start_sec', 'end_sec',
            'scientific_name', 'common_name', 'confidence'
        ])

    # Parse species name into scientific + common
    parsed = df['species_name'].apply(_parse_species_name)
    df['scientific_name'] = [p[0] for p in parsed]
    df['common_name']     = [p[1] for p in parsed]

    df['start_sec']  = df['start_time'].astype(float).round(2)
    df['end_sec']    = df['end_time'].astype(float).round(2)
    df['filename']   = filename
    df['confidence'] = df['confidence'].astype(float).round(4)

    return df[['filename', 'start_sec', 'end_sec',
               'scientific_name', 'common_name', 'confidence']]


def run_birdnet_northeastern(
    data_dir:   Path,
    output_csv: Path,
    min_conf:   float = 0.25,
    top_k:      Optional[int] = None,
    lat:        float = NORTHEASTERN_LAT,
    lon:        float = NORTHEASTERN_LON,
    n_files:    Optional[int] = None,
) -> None:
    """
    Run BirdNET on all Northeastern Birds WAV files in data_dir.
    Audio was converted from FLAC to WAV -- FLAC decoding was unreliable
    for some files (libsndfile sync errors causing hangs). The pre-flight
    check in run_birdnet_on_file() catches any remaining problematic files
    gracefully without hardcoding filenames.
    Saves detections to output_csv incrementally with skip logic.
    Geo model species list is cached per unique week to avoid redundant calls.

    Parameters
    ----------
    data_dir   : path to soundscape_data/ folder containing .wav files
    output_csv : path to save detections CSV
    min_conf   : minimum confidence threshold (default 0.25)
    top_k      : max detections per 3s window (None = all above threshold)
    lat        : recording latitude (default Sapsucker Woods, Ithaca NY)
    lon        : recording longitude (default Sapsucker Woods, Ithaca NY)
    n_files    : if set, process only the first n_files files
    """
    import birdnet

    data_dir   = Path(data_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Get all WAV files sorted
    audio_files = sorted(data_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} WAV files in {data_dir}")

    # Load models once
    print("Loading BirdNET models...")
    acoustic_model = birdnet.load("acoustic", "2.4", "tf")
    geo_model      = birdnet.load("geo",      "2.4", "tf")
    print("Models loaded.")

    # Load already-completed files for skip logic
    completed = set()
    if output_csv.exists():
        existing  = pd.read_csv(output_csv)
        completed = set(existing['filename'].unique())
        print(f"Found {len(completed)} already-processed files, skipping.")

    # Pre-compute geo model species list per unique week
    print("\nBuilding geo model week cache...")
    weeks = []
    for audio_path in audio_files:
        date = _date_from_filename(audio_path.name)
        if date is not None:
            weeks.append(_date_to_birdnet_week(date))
    week_cache = _build_week_species_cache(geo_model, weeks, lat=lat, lon=lon)
    print(f"Cached {len(week_cache)} unique weeks.\n")

    # Limit to n_files if specified
    if n_files is not None:
        audio_files = audio_files[:n_files]
        print(f"Running on first {n_files} files only.")

    # Process each file
    for audio_path in tqdm(audio_files, desc="BirdNET", unit="file"):
        filename = audio_path.name

        if filename in completed:
            tqdm.write(f"  Skipping {filename} (already processed)")
            continue

        try:
            df = run_birdnet_on_file(
                audio_path     = audio_path,
                acoustic_model = acoustic_model,
                geo_model      = geo_model,
                lat            = lat,
                lon            = lon,
                min_conf       = min_conf,
                top_k          = top_k,
                week_cache     = week_cache,
            )
            write_header = not output_csv.exists()
            df.to_csv(output_csv, mode='a', header=write_header, index=False)
            completed.add(filename)
            tqdm.write(f"  {filename}: {len(df)} detections")

        except Exception as e:
            tqdm.write(f"  Skipping {filename}, {type(e).__name__}: {e}")
            continue

    print(f"\nDone. Detections saved to {output_csv}")
    print(f"Total files processed: {len(completed)}")


def run_birdnet_reconstructed(
    detections_csv: Path,
    recon_dir:      Path,
    output_csv:     Path,
    dataset:        str   = 'northeastern_birds',
    min_conf:       float = 0.25,
    top_k:          Optional[int] = None,
    lat:            float = NORTHEASTERN_LAT,
    lon:            float = NORTHEASTERN_LON,
) -> None:
    """
    Run BirdNET on reconstructed audio files for files already in detections_csv.
    Saves detections to output_csv with codec and bitrate columns added.
    Skip logic: skips (filename, codec, bitrate) combinations already in output_csv.

    Parameters
    ----------
    detections_csv : path to original audio detections CSV (used to get file list)
    recon_dir      : path to reconstructed_audio/{dataset}/ folder
    output_csv     : path to save reconstructed detections CSV
    dataset        : dataset name (default 'northeastern_birds')
    min_conf       : minimum confidence threshold (default 0.25)
    top_k          : max detections per 3s window (None = all above threshold)
    lat            : recording latitude
    lon            : recording longitude
    """
    import birdnet

    detections_csv = Path(detections_csv)
    recon_dir      = Path(recon_dir)
    output_csv     = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Get list of original files from detections.csv
    detections     = pd.read_csv(detections_csv)
    original_files = detections['filename'].unique().tolist()
    print(f"Original files to process: {len(original_files)}")

    # Load models once
    print("Loading BirdNET models...")
    acoustic_model = birdnet.load("acoustic", "2.4", "tf")
    geo_model      = birdnet.load("geo",      "2.4", "tf")
    print("Models loaded.")

    # Load already-completed (filename, codec, bitrate) combinations
    completed = set()
    if output_csv.exists():
        existing  = pd.read_csv(output_csv)
        for _, row in existing.iterrows():
            completed.add((row['filename'], row['codec'], float(row['bitrate'])))
        print(f"Found {len(completed)} already-processed combinations, skipping.")

    # Pre-compute geo model week cache for all unique weeks
    print("\nBuilding geo model week cache...")
    weeks = []
    for filename in original_files:
        date = _date_from_filename(filename)
        if date is not None:
            weeks.append(_date_to_birdnet_week(date))
    week_cache = _build_week_species_cache(geo_model, weeks, lat=lat, lon=lon)
    print(f"Cached {len(week_cache)} unique weeks.\n")

    # Get all codec/bitrate subdirectories under recon_dir/dataset/
    dataset_recon_dir = recon_dir / dataset
    if not dataset_recon_dir.exists():
        print(f"No reconstructed audio found at {dataset_recon_dir}")
        return

    codec_bitrate_dirs = []
    for codec_dir in sorted(dataset_recon_dir.iterdir()):
        if not codec_dir.is_dir():
            continue
        codec = codec_dir.name
        for bitrate_dir in sorted(codec_dir.iterdir()):
            if not bitrate_dir.is_dir():
                continue
            try:
                bitrate = float(bitrate_dir.name)
            except ValueError:
                continue
            codec_bitrate_dirs.append((codec, bitrate, bitrate_dir))

    print(f"Found {len(codec_bitrate_dirs)} codec/bitrate combinations.")

    total = len(original_files) * len(codec_bitrate_dirs)
    print(f"Total files to process: {total}\n")

    with tqdm(total=total, desc="BirdNET reconstructed", unit="file") as pbar:
        for codec, bitrate, bitrate_dir in codec_bitrate_dirs:
            for filename in original_files:
                pbar.update(1)
                key = (filename, codec, bitrate)

                if key in completed:
                    tqdm.write(f"  Skipping {filename} {codec}@{bitrate} (already processed)")
                    continue

                # Construct reconstructed audio path
                stem       = Path(filename).stem
                recon_path = bitrate_dir / f"{stem}_reconstructed.wav"

                if not recon_path.exists():
                    tqdm.write(f"  Missing: {recon_path.name}")
                    continue

                try:
                    # Pre-flight check
                    import soundfile as sf
                    with sf.SoundFile(str(recon_path)) as f:
                        f.read(frames=1000)

                    # Get species list from cache
                    date = _date_from_filename(filename)
                    week = _date_to_birdnet_week(date) if date else None
                    if week_cache is not None and week in week_cache:
                        species_set = week_cache[week]
                    else:
                        geo_result  = geo_model.predict(lat, lon, week=week,
                                                        min_confidence=0.03)
                        species_set = geo_result.to_set()

                    # Run BirdNET
                    predictions = acoustic_model.predict(
                        str(recon_path),
                        custom_species_list          = species_set,
                        default_confidence_threshold = min_conf,
                        top_k                        = top_k,
                    )

                    df = predictions.to_dataframe()

                    if df.empty:
                        df = pd.DataFrame(columns=[
                            'filename', 'codec', 'bitrate',
                            'start_sec', 'end_sec',
                            'scientific_name', 'common_name', 'confidence'
                        ])
                    else:
                        parsed = df['species_name'].apply(_parse_species_name)
                        df['scientific_name'] = [p[0] for p in parsed]
                        df['common_name']     = [p[1] for p in parsed]
                        df['start_sec']       = df['start_time'].astype(float).round(2)
                        df['end_sec']         = df['end_time'].astype(float).round(2)
                        df['confidence']      = df['confidence'].astype(float).round(4)
                        df['filename']        = filename  # use original filename
                        df['codec']           = codec
                        df['bitrate']         = bitrate
                        df = df[['filename', 'codec', 'bitrate',
                                 'start_sec', 'end_sec',
                                 'scientific_name', 'common_name', 'confidence']]

                    write_header = not output_csv.exists()
                    df.to_csv(output_csv, mode='a', header=write_header, index=False)
                    completed.add(key)
                    tqdm.write(f"  {filename} {codec}@{bitrate}: {len(df)} detections")

                except Exception as e:
                    tqdm.write(f"  Skipping {filename} {codec}@{bitrate}, {type(e).__name__}: {e}")
                    continue

    print(f"\nDone. Detections saved to {output_csv}")
    print(f"Total combinations processed: {len(completed)}")