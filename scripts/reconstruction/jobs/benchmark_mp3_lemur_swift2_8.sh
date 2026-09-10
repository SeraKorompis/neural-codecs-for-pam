#!/bin/bash
#PBS -N benchmark_mp3_lemur_swift2_8
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/benchmark_mp3_lemur_swift2_8.out
#PBS -e logs/benchmark_mp3_lemur_swift2_8.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

RESULTS_BASE="${EPHEMERAL}/ai_audio_compression/results/full/cpu"
mkdir -p "${RESULTS_BASE}"

python scripts/run_benchmark.py     --codec mp3     --dataset lemur_swift2     --device cpu     --mode full     --results_dir "${RESULTS_BASE}"     --bitrates "8"

echo "Finished: $(date)"
