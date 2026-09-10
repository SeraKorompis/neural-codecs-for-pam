#!/bin/bash
#PBS -N ext_mp3_lemur_test_24
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/ext_mp3_lemur_test_b24.out
#PBS -e logs/ext_mp3_lemur_test_b24.err
echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs
python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py     --dataset lemur --split test --source mp3 --bitrate 24 --device CPU
echo "Finished: $(date)"
