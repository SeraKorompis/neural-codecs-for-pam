#!/bin/bash
#PBS -N bn_embed
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=32gb
# Extract BirdNET embeddings for one (dataset, split, source, bitrate).
# Submitted by submit_extract_embeddings.sh, which sets via `qsub -v`:
#   DATASET anuraset|lemur|northeastern   SPLIT train|test|all
#   SOURCE original|encodec|dac|mp3|opus  BITRATE (not for original)
#   LIMIT (first N clips/files, quick tests only)   CONDA_ENV   DEVICE (CPU|GPU, default CPU)

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
echo "dataset=$DATASET split=$SPLIT source=$SOURCE bitrate=${BITRATE:-none} limit=${LIMIT:-all}"

# Defaults to the directory you submitted from (run qsub from the repository root).
PROJECT_ROOT="${PROJECT_ROOT:-$PBS_O_WORKDIR}"
cd "$PROJECT_ROOT" || exit 1

eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-conda-birdnet-env}"
export PYTHONUNBUFFERED=1

ARGS=(--dataset "$DATASET" --split "$SPLIT" --source "$SOURCE" --device "${DEVICE:-CPU}")
[[ -n "$BITRATE" ]] && ARGS+=(--bitrate "$BITRATE")
[[ -n "$LIMIT" ]]   && ARGS+=(--limit "$LIMIT")

python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py "${ARGS[@]}"

echo "Finished: $(date)"
