#!/bin/bash
#PBS -N figure
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:15:00
#PBS -j oe
cd "$PBS_O_WORKDIR"
eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"
conda activate conda-codec-env-test
python -u "$SCRIPT" $ARGS
