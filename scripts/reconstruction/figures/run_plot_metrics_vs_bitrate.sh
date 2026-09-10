#!/bin/bash
#PBS -N plot_recon_quality
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/plot_metrics_vs_bitrate.out
#PBS -e logs/plot_metrics_vs_bitrate.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/reconstruction/figures/plot_metrics_vs_bitrate.py

echo "Finished: $(date)"
