#!/bin/bash
#PBS -N train_lemur_opus24
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -o logs/train_lemur_opus24.out
#PBS -e logs/train_lemur_opus24.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "--- Opus 24 kbps lemur ---"
python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset lemur --train_source opus --train_bitrate 24

echo "Finished: $(date)"
