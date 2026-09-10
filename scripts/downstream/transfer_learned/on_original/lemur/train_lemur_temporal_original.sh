#!/bin/bash
#PBS -N train_lemur_temporal
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/train_lemur_temporal_original.out
#PBS -e logs/train_lemur_temporal_original.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset lemur_temporal

echo "Finished: $(date)"
