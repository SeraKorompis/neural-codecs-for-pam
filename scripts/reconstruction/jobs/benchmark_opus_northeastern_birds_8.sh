#!/bin/bash
#PBS -N opus_northeastern_birds_8
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/benchmark_opus_northeastern_birds_8.out
#PBS -e logs/benchmark_opus_northeastern_birds_8.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# Safety check — verify old opus deleted from both locations
if [ -d "/rds/general/project/bugg/live/seraphina_compression/reconstructed_audio/northeastern_birds/opus" ]; then
    echo "ERROR: old opus still exists in reconstructed_audio. Aborting."
    exit 1
fi
if [ -d "/northeastern_birds/opus" ]; then
    echo "ERROR: old opus still exists in supplementary. Aborting."
    exit 1
fi

python scripts/reconstruction/run_benchmark.py     --codec opus     --dataset northeastern_birds     --device cpu     --mode full     --bitrates 8     --results_dir /rds/general/project/bugg/live/seraphina_compression/results/reconstruction_quality     --recon_dir /rds/general/project/bugg/live/seraphina_compression/reconstructed_audio

echo "Finished: $(date)"
