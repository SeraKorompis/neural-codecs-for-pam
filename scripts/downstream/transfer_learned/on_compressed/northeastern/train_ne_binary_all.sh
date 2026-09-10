#!/bin/bash
#PBS -N train_ne_binary_all
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_ne_binary_all.out
#PBS -e logs/train_ne_binary_all.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# Original
echo "--- Original ---"
python scripts/downstream/transfer_learned/train_binary_classifiers.py \
    --dataset northeastern

# EnCodec
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps ---"
    python scripts/downstream/transfer_learned/train_binary_classifiers.py \
        --dataset northeastern --train_source encodec --train_bitrate $br
done

# DAC
for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq ---"
    python scripts/downstream/transfer_learned/train_binary_classifiers.py \
        --dataset northeastern --train_source dac --train_bitrate $nq
done

# MP3
for br in 8 16 24; do
    echo "--- MP3 $br kbps ---"
    python scripts/downstream/transfer_learned/train_binary_classifiers.py \
        --dataset northeastern --train_source mp3 --train_bitrate $br
done

echo "Finished: $(date)"
