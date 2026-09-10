#!/bin/bash
#PBS -N train_anuraset_dac_compressed
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -o logs/train_anuraset_dac_compressed.out
#PBS -e logs/train_anuraset_dac_compressed.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq ---"
    python scripts/downstream/transfer_learned/train_classifier.py \
        --dataset anuraset --train_source dac --train_bitrate $nq
done

echo "Finished: $(date)"
