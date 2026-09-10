#!/bin/bash
#PBS -N birdnet_opus_1
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/birdnet_detection_opus_1.out
#PBS -e logs/birdnet_detection_opus_1.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/pretrained/birdnet/run_birdnet_detection.py \
    --reconstructed_only \
    --codecs opus \
    --bitrates 8,14,24

echo "Finished: $(date)"
