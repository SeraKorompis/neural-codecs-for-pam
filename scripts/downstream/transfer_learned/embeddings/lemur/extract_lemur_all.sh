#!/bin/bash
#PBS -N ext_lemur_all
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/ext_lemur_all.out
#PBS -e logs/ext_lemur_all.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# Original
echo "--- Original ---"
python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
    --dataset lemur --split all --source original --device CPU

# EnCodec
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps ---"
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset lemur --split all --source encodec --bitrate $br --device CPU
done

# DAC
for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq ---"
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset lemur --split all --source dac --bitrate $nq --device CPU
done

# MP3
for br in 8 16 24; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset lemur --split all --source mp3 --bitrate $br --device CPU
done

# Opus
for br in 8 14 24; do
    echo "--- Opus $br kbps ---"
    python scripts/downstream/transfer_learned/extract_birdnet_embeddings.py \
        --dataset lemur --split all --source opus --bitrate $br --device CPU
done

echo "Finished: $(date)"
