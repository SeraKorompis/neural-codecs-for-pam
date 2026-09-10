#!/bin/bash
#PBS -N train_ne_ovr_all
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_northeastern_ovr_all.out
#PBS -e logs/train_northeastern_ovr_all.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# --- northeastern_ovr on original ---
echo "--- northeastern_ovr original ---"
python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset northeastern_ovr

# --- northeastern_ovr on compressed ---
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- northeastern_ovr EnCodec $br ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset northeastern_ovr --train_source encodec --train_bitrate $br
done

for nq in 2 3 6 9; do
    echo "--- northeastern_ovr DAC $nq ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset northeastern_ovr --train_source dac --train_bitrate $nq
done

for br in 8 16 24; do
    echo "--- northeastern_ovr MP3 $br ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset northeastern_ovr --train_source mp3 --train_bitrate $br
done

for br in 8 14 24; do
    echo "--- northeastern_ovr Opus $br ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset northeastern_ovr --train_source opus --train_bitrate $br
done

echo "Finished: $(date)"
