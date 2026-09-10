#!/bin/bash
#PBS -N plot_violin_ne
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/plot_violin_northeastern.out
#PBS -e logs/plot_violin_northeastern.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/figures/plot_violin_with_spectrograms.py \
    --dataset northeastern

echo "Finished: $(date)"
