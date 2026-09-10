#!/bin/bash
#PBS -N create_lemur_split
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -o logs/create_lemur_temporal_split.out
#PBS -e logs/create_lemur_temporal_split.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/create_temporal_split.py \
    --dataset lemur \
    --test_ratio 0.2

echo "Finished: $(date)"
