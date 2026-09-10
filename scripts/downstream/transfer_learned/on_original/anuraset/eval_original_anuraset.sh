#!/bin/bash
#PBS -N eval_og_anuraset
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/eval_original_anuraset.out
#PBS -e logs/eval_original_anuraset.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# Original
echo "--- Original ---"
python scripts/downstream/transfer_learned/evaluate_classifier.py \
    --dataset anuraset --source original

# EnCodec
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source encodec --bitrate $br
done

# DAC
for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source dac --bitrate $nq
done

# MP3
for br in 8 16 24 32 64; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source mp3 --bitrate $br
done

# Opus
for br in 8 14 24; do
    echo "--- Opus $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source opus --bitrate $br
done

echo "Finished: $(date)"
