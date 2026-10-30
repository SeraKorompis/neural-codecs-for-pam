#!/bin/bash
#PBS -N plot_recon_quality
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/plot_metrics_vs_bitrate.out
#PBS -e logs/plot_metrics_vs_bitrate.err
# Run from the repository root after `mkdir -p logs`:
#   qsub scripts/reconstruction/jobs/plot_metrics_vs_bitrate.sh

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"

# Defaults to the directory you submitted from (run qsub from the repository root).
# To use a different location: qsub -v PROJECT_ROOT=/path/to/repo ...
PROJECT_ROOT="${PROJECT_ROOT:-$PBS_O_WORKDIR}"
cd "$PROJECT_ROOT" || exit 1

eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1

python scripts/reconstruction/figures/plot_metrics_vs_bitrate.py

echo "Finished: $(date)"