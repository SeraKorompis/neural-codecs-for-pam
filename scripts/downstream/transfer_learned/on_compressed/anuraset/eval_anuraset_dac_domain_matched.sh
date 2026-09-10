#!/bin/bash
#PBS -N eval_anuraset_dac_matched
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_anuraset_dac_domain_matched.out
#PBS -e logs/eval_anuraset_dac_domain_matched.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq (domain-matched) ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset anuraset \
        --source dac --bitrate $nq \
        --train_source dac --train_bitrate $nq
done

echo "Finished: $(date)"
