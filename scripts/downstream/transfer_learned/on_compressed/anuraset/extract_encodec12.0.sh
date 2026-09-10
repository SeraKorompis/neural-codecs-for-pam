#!/bin/bash
#PBS -N extract_anuraset_enc12.0
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/extract_anuraset_enc12.0.out
#PBS -e logs/extract_anuraset_enc12.0.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py     --dataset anuraset --split train --source encodec --bitrate 12.0 --device CPU

echo "Finished: $(date)"
