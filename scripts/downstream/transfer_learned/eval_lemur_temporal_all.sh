#!/bin/bash
#PBS -N eval_lemur_temporal_all
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_lemur_temporal_all.out
#PBS -e logs/eval_lemur_temporal_all.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

EVAL="python scripts/downstream/transfer_learned/evaluate_classifier.py --dataset lemur_temporal"

# ---- Classifier trained on original, evaluated on all conditions ----
echo "=== Trained on original ==="
$EVAL --source original

for br in 1.5 3.0 6.0 12.0 24.0; do
    $EVAL --source encodec --bitrate $br
done
for nq in 2 3 6 9; do
    $EVAL --source dac --bitrate $nq
done
for br in 8 16 24; do
    $EVAL --source mp3 --bitrate $br
done
for br in 8 14 24; do
    $EVAL --source opus --bitrate $br
done

# ---- Domain-matched: trained and evaluated on same compressed condition ----
echo "=== Domain-matched ==="
for br in 1.5 3.0 6.0 12.0 24.0; do
    $EVAL --source encodec --bitrate $br \
          --train_source encodec --train_bitrate $br
done
for nq in 2 3 6 9; do
    $EVAL --source dac --bitrate $nq \
          --train_source dac --train_bitrate $nq
done
for br in 8 16 24; do
    $EVAL --source mp3 --bitrate $br \
          --train_source mp3 --train_bitrate $br
done
for br in 8 14 24; do
    $EVAL --source opus --bitrate $br \
          --train_source opus --train_bitrate $br
done

echo "Finished: $(date)"
