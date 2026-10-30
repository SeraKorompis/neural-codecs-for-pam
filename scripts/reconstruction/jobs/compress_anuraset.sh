#!/bin/bash
#PBS -N compress_anuraset
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
# Submitted by submit_compress_anuraset.sh, which sets CODEC, SPLIT, BITRATE,
# DEVICE and optionally MAX_FILES via `qsub -v` and overrides the resources
# and log paths on the command line.

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
echo "codec=$CODEC split=$SPLIT bitrate=$BITRATE device=$DEVICE max_files=${MAX_FILES:-all}"

# Defaults to the directory you submitted from (run qsub from the repository root).
# To use a different location: qsub -v PROJECT_ROOT=/path/to/repo ...
PROJECT_ROOT="${PROJECT_ROOT:-$PBS_O_WORKDIR}"
cd "$PROJECT_ROOT" || exit 1

eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1

python scripts/reconstruction/compress_anuraset_preprocessed.py \
    --split "$SPLIT" --codec "$CODEC" \
    --bitrates "$BITRATE" --device "$DEVICE" \
    ${MAX_FILES:+--max_files "$MAX_FILES"}

echo "Finished: $(date)"
