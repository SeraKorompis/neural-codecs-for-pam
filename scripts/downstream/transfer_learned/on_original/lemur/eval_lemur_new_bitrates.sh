#!/bin/bash
#PBS -N eval_lemur_new_br
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_lemur_new_bitrates.out
#PBS -e logs/eval_lemur_new_bitrates.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 8 16 24; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur --source mp3 --bitrate $br
done
for br in 8 14; do
    echo "--- Opus $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur --source opus --bitrate $br
done

echo "Finished: $(date)"
