#!/bin/bash
#PBS -N plot_violin
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/plot_violin_with_spectrograms.out
#PBS -e logs/plot_violin_with_spectrograms.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# python scripts/downstream/figures/plot_violin_with_spectrograms.py \
#     --dataset birdnet

# python scripts/downstream/figures/plot_violin_with_spectrograms.py \
#     --dataset northeastern

python scripts/downstream/figures/plot_violin_with_spectrograms.py \
    --dataset anuraset

echo "Finished: $(date)"
