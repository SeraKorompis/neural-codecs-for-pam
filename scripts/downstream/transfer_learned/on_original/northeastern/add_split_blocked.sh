#!/bin/bash
#PBS -N add_ne_split_blocked
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -o logs/add_northeastern_split_blocked.out
#PBS -e logs/add_northeastern_split_blocked.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/add_northeastern_split.py \
    --strategy blocked --test_every 2

echo "Finished: $(date)"
