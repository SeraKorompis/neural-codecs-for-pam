#!/bin/bash
#PBS -N train_eval_anu_mp3
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_eval_anuraset_mp3_new.out
#PBS -e logs/train_eval_anuraset_mp3_new.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 8 16 24; do
    echo "--- MP3 $br kbps train ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset anuraset --train_source mp3 --train_bitrate $br

    echo "--- MP3 $br kbps eval on original ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source mp3 --bitrate $br

    echo "--- MP3 $br kbps eval domain-matched ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source mp3 --bitrate $br \
        --train_source mp3 --train_bitrate $br
done

echo "Finished: $(date)"
