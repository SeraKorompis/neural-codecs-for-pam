#!/bin/bash
#PBS -N eval_anuraset_all
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_anuraset_all_compressed.out
#PBS -e logs/eval_anuraset_all_compressed.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# Original baseline
echo "--- Original ---"
python scripts/downstream/transfer_learned/on_original/evaluate_classifier.py \
    --dataset anuraset \
    --source  original

# DAC
for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq ---"
    python scripts/downstream/transfer_learned/on_original/evaluate_classifier.py \
        --dataset anuraset \
        --source  dac \
        --bitrate $nq
done

# EnCodec
for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps ---"
    python scripts/downstream/transfer_learned/on_original/evaluate_classifier.py \
        --dataset anuraset \
        --source  encodec \
        --bitrate $br
done

echo "Finished: $(date)"
