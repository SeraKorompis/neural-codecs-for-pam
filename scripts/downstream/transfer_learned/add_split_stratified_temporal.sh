#!/bin/bash
#PBS -N add_ne_split_strat
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/add_northeastern_split_stratified.out
#PBS -e logs/add_northeastern_split_stratified.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/add_northeastern_split.py \
    --strategy stratified_temporal \
    --test_ratio 0.2 \
    --min_pos_train 10

echo "Finished: $(date)"