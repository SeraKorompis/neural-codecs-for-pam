#!/bin/bash
#PBS -N eval_ne_ovr_orig
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_ne_ovr_original.out
#PBS -e logs/eval_ne_ovr_original.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# Original only first to test
python scripts/downstream/transfer_learned/evaluate_classifier.py \
    --dataset northeastern_ovr \
    --source original

echo "Finished: $(date)"
