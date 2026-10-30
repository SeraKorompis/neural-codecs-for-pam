#!/bin/bash
#PBS -N birdnet_detect
#PBS -l walltime=16:00:00
#PBS -l select=1:ncpus=4:mem=32gb
# Run BirdNET detection on Northeastern Birds audio. Variables, set with `qsub -v`:
#   STEP      original | reconstructed | all (default: all)
#   CODECS    comma-separated, e.g. mp3 or mp3,opus (default: all codecs found)
#   BITRATES  comma-separated, e.g. 8,16,24 (default: all bitrates found)
#   N_FILES   limit to the first N files, for quick tests only (never use with the real config)
#   CONDA_ENV conda environment name (default: conda-birdnet-env)
# Run from the repository root after `mkdir -p logs`. Do not run two detection jobs at once,
# because they append to the same CSV file.

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
echo "step=${STEP:-all} codecs=${CODECS:-all} bitrates=${BITRATES:-all} n_files=${N_FILES:-all}"

# Defaults to the directory you submitted from (run qsub from the repository root).
PROJECT_ROOT="${PROJECT_ROOT:-$PBS_O_WORKDIR}"
cd "$PROJECT_ROOT" || exit 1

eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-conda-birdnet-env}"
export PYTHONUNBUFFERED=1

# An empty N_FILES would crash the script, so remove it unless it has a value.
[[ -z "$N_FILES" ]] && unset N_FILES

ARGS=()
[[ "$STEP" == "original" ]]      && ARGS+=(--original_only)
[[ "$STEP" == "reconstructed" ]] && ARGS+=(--reconstructed_only)
[[ -n "$CODECS" ]]               && ARGS+=(--codecs "$CODECS")
[[ -n "$BITRATES" ]]             && ARGS+=(--bitrates "$BITRATES")

python scripts/downstream/pretrained/birdnet/run_birdnet_detection.py "${ARGS[@]}"

echo "Finished: $(date)"
