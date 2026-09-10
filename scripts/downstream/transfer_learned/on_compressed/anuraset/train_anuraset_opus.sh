#!/bin/bash
#PBS -N train_anu_opus
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -o logs/train_anuraset_opus.out
#PBS -e logs/train_anuraset_opus.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 8 14; do
    echo "--- Opus $br kbps ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset anuraset --train_source opus --train_bitrate $br
done

echo "Finished: $(date)"
