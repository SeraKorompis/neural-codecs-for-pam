#!/bin/bash
#PBS -N plot_aggregated
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -o logs/plot_aggregated_figures.out
#PBS -e logs/plot_aggregated_figures.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/figures/plot_transfer_learned_og_agg.py
python scripts/downstream/figures/plot_transfer_learned_comp_agg.py

echo "Finished: $(date)"
