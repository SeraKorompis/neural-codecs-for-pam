#!/bin/bash
# Reproduces the figures and numbers in the paper.
# Run from anywhere with the conda environment active:  bash make_paper_figures.sh
# Outputs go to the figures directory set in src/config.py.
set -e
cd "$(dirname "$0")"

# Pre-trained BirdNET: macro metrics against bitrate
python scripts/downstream/figures/plot_pretrained_figure.py

# Transfer-learned classifier trained on raw audio: macro ROC-AUC against bitrate
python scripts/downstream/figures/plot_fine_tuned_on_og.py

# Change in AUC from training on compressed audio (ΔAUC)
python scripts/downstream/figures/plot_fine_tuning_delta.py

# Per-species confidence change (needs about 32 GB of memory and 30 mins)
python scripts/downstream/figures/plot_confidence_violins.py --codec encodec --kbps 6.0

# Numbers quoted in the text (ΔAUC ranges, counts, per-condition values)
python scripts/downstream/figures/delta_auc_numbers.py
