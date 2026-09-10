#!/bin/bash
#PBS -N eval_ne_mp3_opus
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/eval_ne_mp3_opus.out
#PBS -e logs/eval_ne_mp3_opus.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 32 64; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern --source mp3 --bitrate $br \
        --train_source mp3 --train_bitrate $br --min_pos_test 10 --bg_ratio 5
done

for br in 6 12 24; do
    echo "--- Opus $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern --source opus --bitrate $br \
        --train_source opus --train_bitrate $br --min_pos_test 10 --bg_ratio 5
done

echo "Finished: $(date)"
