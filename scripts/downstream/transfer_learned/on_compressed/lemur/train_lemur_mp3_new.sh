#!/bin/bash
#PBS -N train_lemur_mp3_new
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -o logs/train_lemur_mp3_new.out
#PBS -e logs/train_lemur_mp3_new.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 8 16 24; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur --train_source mp3 --train_bitrate $br
done

echo "Finished: $(date)"
