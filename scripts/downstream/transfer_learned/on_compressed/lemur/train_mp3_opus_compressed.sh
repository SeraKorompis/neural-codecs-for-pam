#!/bin/bash
#PBS -N train_mp3_opus_compressed
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -o logs/train_mp3_opus_compressed.out
#PBS -e logs/train_mp3_opus_compressed.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 32 64; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur --train_source mp3 --train_bitrate $br
done

for nq in 6 12 24; do
    echo "--- Opus n_q=$nq ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur --train_source opus --train_bitrate $nq
done

echo "Finished: $(date)"
