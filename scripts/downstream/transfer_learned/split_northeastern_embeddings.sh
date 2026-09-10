#!/bin/bash
#PBS -N split_ne_embeddings
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/split_northeastern_embeddings.out
#PBS -e logs/split_northeastern_embeddings.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/split_northeastern_embeddings.py

echo "Finished: $(date)"
