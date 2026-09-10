#!/bin/bash
#PBS -N eval_lemur_all_comp
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_lemur_all_compressed.out
#PBS -e logs/eval_lemur_all_compressed.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# EnCodec domain-matched
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps domain-matched ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur --source encodec --bitrate $br \
        --train_source encodec --train_bitrate $br
done

# DAC domain-matched
for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq domain-matched ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur --source dac --bitrate $nq \
        --train_source dac --train_bitrate $nq
done

# MP3 domain-matched
for br in 8 16 24 32 64; do
    echo "--- MP3 $br kbps domain-matched ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur --source mp3 --bitrate $br \
        --train_source mp3 --train_bitrate $br
done

# Opus domain-matched
for br in 8 14 24; do
    echo "--- Opus $br kbps domain-matched ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur --source opus --bitrate $br \
        --train_source opus --train_bitrate $br
done

echo "Finished: $(date)"
