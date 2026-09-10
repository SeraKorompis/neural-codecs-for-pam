#!/bin/bash
#PBS -N compress_anu_mp3_32_test
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/compress_anuraset_preprocessed_mp3_32_test.out
#PBS -e logs/compress_anuraset_preprocessed_mp3_32_test.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/reconstruction/compress_anuraset_preprocessed.py     --split test     --codec mp3     --bitrates 32     --device cpu

echo "Finished: $(date)"
