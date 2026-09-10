#!/bin/bash
#PBS -N eval_birdnet_ovr
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o logs/eval_birdnet_ovr.out
#PBS -e logs/eval_birdnet_ovr.err

echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
cd /rds/general/user/stk25/home/ai_audio_compression/project

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate conda-birdnet-env
export PYTHONUNBUFFERED=1
mkdir -p logs

EVAL="python scripts/downstream/pretrained/birdnet/evaluate_birdnet_ovr.py"
RECON="/rds/general/project/bugg/live/seraphina_compression/results/downstream/pretrained/birdnet/raw/detections_reconstructed_lowthresh.csv"

# Original
$EVAL --condition original --codec original --kbps 0

# EnCodec
for br in 1.5 3.0 6.0 12.0 24.0; do
    $EVAL --detections_csv $RECON \
          --condition encodec${br} --codec encodec --bitrate $br --kbps $br
done

# DAC
for nq in 2 3 6 9; do
    case $nq in
        2) kbps=1.78 ;; 3) kbps=2.67 ;; 6) kbps=5.33 ;; 9) kbps=8.0 ;;
    esac
    $EVAL --detections_csv $RECON \
          --condition dac${nq} --codec dac --bitrate $nq --kbps $kbps
done

# MP3
for br in 8 16 24; do
    $EVAL --detections_csv $RECON \
          --condition mp3${br} --codec mp3 --bitrate $br --kbps $br
done

# Opus
for br in 8 14 24; do
    $EVAL --detections_csv $RECON \
          --condition opus${br} --codec opus --bitrate $br --kbps $br
done

echo "Finished: $(date)"
