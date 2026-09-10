#!/bin/bash
#PBS -N train_ne_shared
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_northeastern_shared.out
#PBS -e logs/train_northeastern_shared.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset northeastern_shared

echo "Finished: $(date)"
