#!/bin/bash
#PBS -N check_ne_species
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -o logs/check_ne_species_overlap.out
#PBS -e logs/check_ne_species_overlap.err

echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env

python3 -c "
import numpy as np, os, json
from pathlib import Path
EPHEMERAL = Path(os.environ['EPHEMERAL'])
EMB_DIR   = EPHEMERAL / 'northeastern_classifier/embeddings'

y_train = np.load(EMB_DIR / 'northeastern_train_labels.npy')
y_test  = np.load(EMB_DIR / 'northeastern_test_labels.npy')

with open(EMB_DIR / 'northeastern_species_cols.json') as f:
    species = json.load(f)

train_pos = y_train.sum(axis=0)
test_pos  = y_test.sum(axis=0)

print('Species with 0 positives in train but >0 in test:')
for i, sp in enumerate(species):
    if train_pos[i] == 0 and test_pos[i] > 0:
        print(f'  {sp}: train=0, test={int(test_pos[i])}')

print()
print('Species with 0 positives in test but >0 in train:')
for i, sp in enumerate(species):
    if test_pos[i] == 0 and train_pos[i] > 0:
        print(f'  {sp}: train={int(train_pos[i])}, test=0')

print()
print(f'Species present in both: {int(((train_pos > 0) & (test_pos > 0)).sum())}')
print(f'Species train only:      {int(((train_pos > 0) & (test_pos == 0)).sum())}')
print(f'Species test only:       {int(((train_pos == 0) & (test_pos > 0)).sum())}')
print(f'Species in neither:      {int(((train_pos == 0) & (test_pos == 0)).sum())}')
"
echo "Finished: $(date)"
