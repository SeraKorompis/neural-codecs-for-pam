#!/bin/bash
#PBS -N plot_train_strategy
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -o logs/plot_roc_auc_train_strategy_comparison.out
#PBS -e logs/plot_roc_auc_train_strategy_comparison.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

# Figures
python scripts/downstream/figures/plot_roc_auc_train_strategy_comparison.py --dataset lemur
python scripts/downstream/figures/plot_roc_auc_train_strategy_comparison.py --dataset anuraset

# Tables
python scripts/downstream/transfer_learned/make_training_strategy_comparison_table.py --dataset lemur
python scripts/downstream/transfer_learned/make_training_strategy_comparison_table.py --dataset anuraset

echo "Finished: $(date)"
