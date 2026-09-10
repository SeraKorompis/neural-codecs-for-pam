#!/bin/bash
#PBS -N eval_lemur_domain_matched
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_lemur_domain_matched.out
#PBS -e logs/eval_lemur_domain_matched.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

for br in 1.5 3.0 6.0 12.0 24.0; do
    echo "--- EnCodec $br kbps (domain-matched) ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur \
        --source encodec --bitrate $br \
        --train_source encodec --train_bitrate $br
done

for nq in 2 3 6 9; do
    echo "--- DAC n_q=$nq (domain-matched) ---"
    python scripts/downstream/transfer_learned/evaluate_classifier.py \
        --dataset lemur \
        --source dac --bitrate $nq \
        --train_source dac --train_bitrate $nq
done

echo "Finished: $(date)"
