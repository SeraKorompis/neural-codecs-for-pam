#!/bin/bash
#PBS -N ext_enc_anu_test_24.0
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/ext_enc_anu_test_b24.0.out
#PBS -e logs/ext_enc_anu_test_b24.0.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py     --dataset anuraset --split test --source encodec --bitrate 24.0 --device CPU

echo "Finished: $(date)"
