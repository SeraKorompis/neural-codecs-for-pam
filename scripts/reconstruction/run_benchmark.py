#!/usr/bin/env python3
"""
scripts/reconstruction/run_benchmark.py
Full bioacoustic codec benchmark for HPC.
Each (codec, dataset, device) combination is submitted as an independent
PBS job via submit_jobs.sh — no job array.
Usage (via submit_jobs.sh):
    bash submit_jobs.sh
Or manually:
    qsub -v CODEC=encodec,DATASET=anuraset,DEVICE=cuda run_benchmark.sh
Results are saved to:
    metrics:              {results_dir}/metrics/{dataset}/{codec}/metrics.csv
    call metrics:         {results_dir}/metrics/{dataset}/{codec}/call_metrics.csv
    reconstructed audio:  {recon_dir}/{dataset}/{codec}/{bitrate}/{stem}_reconstructed.wav

If --recon_dir is not provided, reconstructed audio is saved to:
    {results_dir}/reconstructed_audio/{dataset}/{codec}/{bitrate}/

If a specific file OOMs on GPU and falls back to CPU, that row is still
written to the SAME results dir — results_dir is fixed for the whole job
based on the REQUESTED device, with the actual device used recorded
per-row in metrics.csv.
If a job fails, the full traceback is written to
logs/task_{codec}_{dataset}_{device}_error.log.
"""
import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.evaluate import run_evaluation
from src.config import PATHS

LOGS_DIR = PROJECT_ROOT / 'logs'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--codec',   type=str, required=True,
                        choices=['encodec', 'dac', 'mp3', 'opus'],
                        help="Codec to run")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['anuraset', 'northeastern_birds',
                                 'lemur_s4a', 'lemur_swift1', 'lemur_swift2'],
                        help="Dataset to run")
    parser.add_argument('--device',  type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help="Device to run on")
    parser.add_argument('--mode',    type=str, default='full',
                        choices=['test', 'full'],
                        help="Run mode")
    parser.add_argument('--bitrates', type=str, default=None,
                        help="Comma-separated bitrates to run, e.g. '2,3'. "
                             "If omitted, all bitrates for the codec/dataset are used.")
    parser.add_argument('--results_dir', type=str, default=None,
                        help="Base directory for results (metrics CSVs). "
                             "Metrics saved to {results_dir}/metrics/{dataset}/{codec}/. "
                             "Defaults to PROJECT_ROOT/results/{mode}/{device}/.")
    parser.add_argument('--recon_dir', type=str, default=None,
                        help="Directory for reconstructed audio. "
                             "Saved to {recon_dir}/{dataset}/{codec}/{bitrate}/. "
                             "If not set, falls back to {results_dir}/reconstructed_audio/.")
    args = parser.parse_args()

    # Parse bitrates
    bitrates = None
    if args.bitrates is not None:
        bitrates = [float(b) if '.' in b else int(b) for b in args.bitrates.split(',')]

    # Resolve results dir
    if args.results_dir is not None:
        RESULTS_DIR = Path(args.results_dir)
    else:
        RESULTS_DIR = Path(PATHS['results_dir']) / args.mode / args.device

    # Resolve recon dir
    if args.recon_dir is not None:
        RECON_DIR = Path(args.recon_dir)
    else:
        RECON_DIR = Path(PATHS['recon_dir'])

    print(f"{'='*60}")
    print(f"codec={args.codec}, dataset={args.dataset}, device={args.device}, mode={args.mode}")
    print(f"Results dir: {RESULTS_DIR}/metrics/{args.dataset}/{args.codec}/")
    print(f"Recon dir:   {RECON_DIR or RESULTS_DIR / 'reconstructed_audio'}/{args.dataset}/{args.codec}/")
    print(f"Started: {datetime.now()}")
    print(f"{'='*60}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        run_evaluation(
            datasets    = [args.dataset],
            codecs      = [args.codec],
            run_mode    = args.mode,
            results_dir = RESULTS_DIR,
            recon_dir   = RECON_DIR,
            device      = args.device,
            bitrates    = bitrates,
        )
        print(f"\n{'='*60}")
        print(f"COMPLETE: codec={args.codec}, dataset={args.dataset}, device={args.device}")
        print(f"Finished: {datetime.now()}")
        print(f"{'='*60}")
    except Exception:
        tb_text = traceback.format_exc()
        print(f"\n{'!'*60}")
        print(f"FAILED: codec={args.codec}, dataset={args.dataset}, device={args.device}")
        print(f"Failed at: {datetime.now()}")
        print(tb_text)
        print(f"{'!'*60}\n")
        error_log = LOGS_DIR / f"task_{args.codec}_{args.dataset}_{args.device}_error.log"
        with open(error_log, 'w') as f:
            f.write(f"codec={args.codec}, dataset={args.dataset}, device={args.device}\n")
            f.write(f"Failed at: {datetime.now()}\n\n")
            f.write(tb_text)
        print(f"Error details written to {error_log}")
        sys.exit(1)