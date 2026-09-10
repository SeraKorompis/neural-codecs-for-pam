#!/bin/bash
#PBS -N eval_ne_binary_compressed
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/eval_northeastern_all_compressed.out
#PBS -e logs/eval_northeastern_all_compressed.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern --train_source encodec --train_bitrate $br --source encodec --bitrate $br
done

for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern --train_source dac --train_bitrate $nq --source dac --bitrate $nq
done

for br in 8 16 24; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern --train_source mp3 --train_bitrate $br --source mp3 --bitrate $br
done

for br in 8 14 24; do
    echo "--- Opus $br kbps ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern --train_source opus --train_bitrate $br --source opus --bitrate $br
done

echo "Finished: $(date)"
