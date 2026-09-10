#!/bin/bash
#PBS -N train_ne_ovr_orig
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_ne_ovr_original.out
#PBS -e logs/train_ne_ovr_original.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/train_northeastern_ovr_classifiers.py \
    --train_source original

echo "Finished: $(date)"
