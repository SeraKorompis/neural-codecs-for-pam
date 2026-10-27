#!/bin/bash
#PBS -N birdnet_detect_mp3
#PBS -l walltime=16:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/birdnet_detection_mp3.out
#PBS -e logs/birdnet_detection_mp3.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/pretrained/birdnet/run_birdnet_detection.py \
    --reconstructed_only \
    --codecs mp3 \
    --bitrates 8,16,24

echo "Finished: $(date)"
