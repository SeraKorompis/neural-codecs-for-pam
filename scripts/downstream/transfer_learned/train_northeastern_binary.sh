#!/bin/bash
#PBS -N train_ne_binary
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_northeastern_binary.out
#PBS -e logs/train_northeastern_binary.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/train_binary_classifiers.py \
    --dataset northeastern --bg_ratio 5

echo "Finished: $(date)"