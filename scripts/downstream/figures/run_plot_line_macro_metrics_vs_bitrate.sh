#!/bin/bash
#PBS -N plot_macro_metrics
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -o logs/plot_line_macro_metrics_vs_bitrate.out
#PBS -e logs/plot_line_macro_metrics_vs_bitrate.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

FIGS=$EPHEMERAL/ai_audio_compression/figures/paper
RESULTS=$EPHEMERAL/ai_audio_compression/results/downstream

# BirdNET -- Balanced Evaluation
echo "--- BirdNET (Balanced) ---"
python scripts/downstream/figures/plot_line_macro_metrics_vs_bitrate.py \
    --inputs balanced:$RESULTS/pretrained/birdnet/macro_metrics_balanced.csv \
    --output $FIGS/birdnet_macro_metrics_vs_bitrate_balanced.png \
    --title  "BirdNET Pre-trained Species Detection (Balanced Evaluation)"

# BirdNET -- Full Evaluation
echo "--- BirdNET (Full) ---"
python scripts/downstream/figures/plot_line_macro_metrics_vs_bitrate.py \
    --inputs full:$RESULTS/pretrained/birdnet/macro_metrics_full.csv \
    --output $FIGS/birdnet_macro_metrics_vs_bitrate_full.png \
    --title  "BirdNET Pre-trained Species Detection (Full Evaluation)"

# AnuraSet
echo "--- AnuraSet ---"
python scripts/downstream/figures/plot_line_macro_metrics_vs_bitrate.py \
    --inputs anuraset:$RESULTS/transfer_learned/on_original/anuraset/macro_metrics.csv \
    --output $FIGS/anuraset_macro_metrics_vs_bitrate.png \
    --title  "AnuraSet Transfer-learned Classifier"

# Lemur
echo "--- Lemur ---"
python scripts/downstream/figures/plot_line_macro_metrics_vs_bitrate.py \
    --inputs lemur:$RESULTS/transfer_learned/on_original/lemur/macro_metrics.csv \
    --output $FIGS/lemur_macro_metrics_vs_bitrate.png \
    --title  "Lemur Call Detection"

echo "Finished: $(date)"