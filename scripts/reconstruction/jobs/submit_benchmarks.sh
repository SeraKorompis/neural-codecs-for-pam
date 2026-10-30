#!/bin/bash
# Submit one independent PBS job per (dataset, codec, bitrate), so jobs run in
# parallel and each has its own walltime. Run from the repository root:
#   bash scripts/reconstruction/jobs/submit_benchmarks.sh
# Dry run (print the qsub commands only):
#   DRY_RUN=1 bash scripts/reconstruction/jobs/submit_benchmarks.sh
# Quick test on one dataset (3 files per condition):
#   DATASETS=anuraset MODE=test bash scripts/reconstruction/jobs/submit_benchmarks.sh
mkdir -p logs

declare -A BITRATES=(
  [encodec]="1.5 3.0 6.0 12.0 24.0"
  [dac]="2 3 6 9"
  [mp3]="8 16 24"
  [opus]="8 14 24"
)
DATASETS="${DATASETS:-anuraset northeastern_birds lemur_s4a lemur_swift1 lemur_swift2}"
MODE="${MODE:-full}"
WALLTIME="24:00:00"

for dataset in $DATASETS; do
  for codec in "${!BITRATES[@]}"; do
    if [[ $codec == encodec || $codec == dac ]]; then
      device=cuda
      res="select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S"
    else
      device=cpu
      res="select=1:ncpus=4:mem=16gb"
    fi
    for br in ${BITRATES[$codec]}; do
      name="bm_${dataset}_${codec}_${br}"
      cmd=(qsub -N "$name" -l "$res" -l "walltime=$WALLTIME"
           -o "logs/${name}.out" -e "logs/${name}.err"
           -v "CODEC=$codec,DATASET=$dataset,BITRATE=$br,DEVICE=$device,MODE=$MODE"
           scripts/reconstruction/jobs/run_benchmark.sh)
      if [[ -n "$DRY_RUN" ]]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
    done
  done
done