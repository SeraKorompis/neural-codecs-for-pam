#!/bin/bash
#PBS -N train_ne_bin_dac6
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_ne_binary_dac6.out
#PBS -e logs/train_ne_binary_dac6.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/train_binary_classifiers.py     --dataset northeastern     --train_source dac     --train_bitrate 6     --bg_ratio 5

echo "Finished: $(date)"
