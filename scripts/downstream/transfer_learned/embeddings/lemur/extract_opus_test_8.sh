#!/bin/bash
#PBS -N ext_opus_lemur_test_8
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/ext_opus_lemur_test_b8.out
#PBS -e logs/ext_opus_lemur_test_b8.err
echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs
python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py     --dataset lemur --split test --source opus --bitrate 8 --device CPU
echo "Finished: $(date)"
