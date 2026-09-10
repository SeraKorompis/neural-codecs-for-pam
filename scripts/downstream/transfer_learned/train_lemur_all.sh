#!/bin/bash
#PBS -N train_lemur_all
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_lemur_all.out
#PBS -e logs/train_lemur_all.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# --- lemur_temporal on original ---
echo "--- lemur_temporal original ---"
python scripts/downstream/transfer_learned/train_classifier.py \
    --dataset lemur_temporal

# --- lemur_temporal on compressed ---
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- lemur_temporal EnCodec $br ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur_temporal --train_source encodec --train_bitrate $br
done

for nq in 2 3 6 9; do
    echo "--- lemur_temporal DAC $nq ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur_temporal --train_source dac --train_bitrate $nq
done

for br in 8 16 24; do
    echo "--- lemur_temporal MP3 $br ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur_temporal --train_source mp3 --train_bitrate $br
done

for br in 8 14 24; do
    echo "--- lemur_temporal Opus $br ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset lemur_temporal --train_source opus --train_bitrate $br
done

echo "Finished: $(date)"
