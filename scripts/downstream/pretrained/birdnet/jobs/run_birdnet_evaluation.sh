#!/bin/bash
#PBS -N eval_birdnet
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_birdnet.out
#PBS -e logs/eval_birdnet.err
# Evaluate BirdNET detections (ROC-AUC, F1, precision, recall) for each condition.
# Run from the repository root after `mkdir -p logs`: qsub <this script>
# Optional variables, set with `qsub -v`:
#   CODECS     conditions to evaluate (default: all of them). One value with -v, e.g.
#              -v CODECS=original. For several, use: CODECS="mp3 opus" qsub -V <script>
#   CONDA_ENV  conda environment name (default: conda-birdnet-env)

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"

# Defaults to the directory you submitted from (run qsub from the repository root).
PROJECT_ROOT="${PROJECT_ROOT:-$PBS_O_WORKDIR}"
cd "$PROJECT_ROOT" || exit 1

eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-conda-birdnet-env}"
export PYTHONUNBUFFERED=1

CODECS="${CODECS//:/ }"
CODECS="${CODECS:-original encodec dac mp3 opus}"

# Read paths from config.yaml via src.config
RESULTS_DIR=$(python3 -c "import sys; sys.path.insert(0, '.'); from src.config import PATHS; print(PATHS['results_dir'])")
EVAL="python scripts/downstream/pretrained/birdnet/evaluate_birdnet.py"
RECON="${RESULTS_DIR}/downstream/pretrained/birdnet/raw/detections_reconstructed_lowthresh.csv"

# run_eval CODEC BITRATE KBPS: evaluate one condition if its codec is selected
run_eval() {
    [[ " $CODECS " == *" $1 "* ]] || return 0
    $EVAL --detections_csv "$RECON" \
          --condition "$1$2" --codec "$1" --bitrate "$2" --kbps "$3"
}

# Original audio (uses the original detections file)
if [[ " $CODECS " == *" original "* ]]; then
    $EVAL --condition original --codec original --kbps 0
fi

for br in 1.5 3.0 6.0 12.0 24.0; do run_eval encodec $br $br; done

run_eval dac 2 1.78
run_eval dac 3 2.67
run_eval dac 6 5.33
run_eval dac 9 8.0

for br in 8 16 24; do run_eval mp3 $br $br; done
for br in 8 14 24; do run_eval opus $br $br; done

echo "Finished: $(date)"
