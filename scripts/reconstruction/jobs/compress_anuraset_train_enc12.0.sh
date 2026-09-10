#!/bin/bash
#PBS -N compress_train_enc12.0
#PBS -l walltime=13:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -o logs/compress_anuraset_train_enc12.0.out
#PBS -e logs/compress_anuraset_train_enc12.0.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "--- Starting compression ---"
python scripts/reconstruction/compress_anuraset_preprocessed.py \
    --split train \
    --codec encodec \
    --bitrates 12.0 \
    --device cuda

echo "Finished: $(date)"
