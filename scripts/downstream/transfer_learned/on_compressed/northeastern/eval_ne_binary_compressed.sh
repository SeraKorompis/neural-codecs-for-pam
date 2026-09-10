#!/bin/bash
#PBS -N eval_ne_binary_comp
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o logs/eval_ne_binary_compressed.out
#PBS -e logs/eval_ne_binary_compressed.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# EnCodec domain-matched
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps (domain-matched) ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern \
        --source encodec --bitrate $br \
        --train_source encodec --train_bitrate $br \
        --min_pos_test 10
done

# DAC domain-matched
for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq (domain-matched) ---"
    python scripts/downstream/transfer_learned/evaluate_binary_classifiers.py \
        --dataset northeastern \
        --source dac --bitrate $nq \
        --train_source dac --train_bitrate $nq \
        --min_pos_test 10
done

echo "Finished: $(date)"
