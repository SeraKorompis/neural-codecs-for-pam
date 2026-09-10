#!/bin/bash
#PBS -N ext_orig_ne
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/ext_orig_ne.out
#PBS -e logs/ext_orig_ne.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
    --dataset northeastern \
    --split all \
    --source original \
    --device CPU

echo "Finished: $(date)"
