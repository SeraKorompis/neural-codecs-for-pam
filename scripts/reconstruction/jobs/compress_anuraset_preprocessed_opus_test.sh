#!/bin/bash
#PBS -N comp_anu_opus_test
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/compress_anuraset_preprocessed_opus_test.out
#PBS -e logs/compress_anuraset_preprocessed_opus_test.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/reconstruction/compress_anuraset_preprocessed.py     --split test     --codec opus     --device cpu     --bitrates 8,14

echo "Finished: $(date)"
