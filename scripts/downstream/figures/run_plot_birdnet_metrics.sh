#!/bin/bash
#PBS -N plot_birdnet_metrics
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -o logs/plot_birdnet_metrics.out
#PBS -e logs/plot_birdnet_metrics.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/figures/plot_birdnet_macro_metrics.py

echo "Finished: $(date)"
