#!/bin/bash
#PBS -N filter_ne_species
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -o logs/filter_ne_shared_species.out
#PBS -e logs/filter_ne_shared_species.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

python scripts/downstream/transfer_learned/filter_northeastern_shared_species.py

echo "Finished: $(date)"
