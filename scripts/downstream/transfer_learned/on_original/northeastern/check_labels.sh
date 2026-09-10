#!/bin/bash
#PBS -N check_ne_labels
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -o logs/check_ne_labels.out
#PBS -e logs/check_ne_labels.err

echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env

python3 -c "
import numpy as np, os
from pathlib import Path
EPHEMERAL = Path(os.environ['EPHEMERAL'])
y = np.load(EPHEMERAL / 'northeastern_classifier/embeddings/northeastern_train_labels.npy')
print('Shape:', y.shape)
print('Total positives:', int(y.sum()))
print('Positives per species (mean):', y.sum(axis=0).mean().round(1))
print('Min positives per species:', int(y.sum(axis=0).min()))
print('Max positives per species:', int(y.sum(axis=0).max()))
print('Windows with >= 1 positive:', int((y.sum(axis=1) > 0).sum()))
print('Windows with all zeros:', int((y.sum(axis=1) == 0).sum()))
print('Fraction positive windows:', round((y.sum(axis=1) > 0).mean(), 4))
"
echo "Finished: $(date)"
