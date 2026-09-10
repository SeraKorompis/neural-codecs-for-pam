#!/bin/bash
#PBS -N eval_native_sr
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_native_sr.out
#PBS -e logs/eval_native_sr.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for dataset in anuraset northeastern_birds lemur_s4a lemur_swift1 lemur_swift2; do
    for codec in encodec dac mp3; do
        echo "--- $dataset / $codec ---"
        python scripts/reconstruction/evaluate_recon_at_native_sr.py \
            --dataset $dataset \
            --codec $codec
        echo "  Done: $(date)"
    done
done

echo "Finished: $(date)"
