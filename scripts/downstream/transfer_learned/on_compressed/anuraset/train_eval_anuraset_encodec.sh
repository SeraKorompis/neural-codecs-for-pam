#!/bin/bash
#PBS -N train_eval_anu_enc
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o logs/train_eval_anuraset_encodec.out
#PBS -e logs/train_eval_anuraset_encodec.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps train ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset anuraset --train_source encodec --train_bitrate $br

    echo "--- EnCodec $br kbps eval on original ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source encodec --bitrate $br

    echo "--- EnCodec $br kbps eval domain-matched ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset --source encodec --bitrate $br \
        --train_source encodec --train_bitrate $br
done

echo "Finished: $(date)"
