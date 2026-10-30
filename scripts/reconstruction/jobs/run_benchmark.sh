#!/bin/bash
#PBS -N benchmark
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
# Submitted by submit_benchmarks.sh, which sets CODEC, DATASET, BITRATE and DEVICE
# via `qsub -v` and overrides the resources and log paths on the command line.

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
echo "codec=$CODEC dataset=$DATASET bitrate=$BITRATE device=$DEVICE"

# Defaults to the directory you submitted from (run qsub from the repository root).
# To use a different location: qsub -v PROJECT_ROOT=/path/to/repo ...
PROJECT_ROOT="${PROJECT_ROOT:-$PBS_O_WORKDIR}"
cd "$PROJECT_ROOT"

eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1

python scripts/reconstruction/run_benchmark.py \
    --codec "$CODEC" --dataset "$DATASET" \
    --bitrates "$BITRATE" --device "$DEVICE" --mode "${MODE:-full}"

echo "Finished: $(date)"
