#!/bin/bash
#PBS -N plot_auc_scatter
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/plot_auc_scatter.out
#PBS -e logs/plot_auc_scatter.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "--- Running BirdNET ---"
python scripts/downstream/figures/plot_auc_scatter_all.py \
    --dataset birdnet

echo "--- Running Northeastern ---"
python scripts/downstream/figures/plot_auc_scatter_all.py \
    --dataset northeastern

echo "--- Running AnuraSet ---"
python scripts/downstream/figures/plot_auc_scatter_all.py \
    --dataset anuraset

echo "Finished: $(date)"