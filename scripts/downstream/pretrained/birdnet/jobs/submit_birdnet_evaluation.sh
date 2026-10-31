#!/bin/bash
# Submit the BirdNET evaluation job. Run from the repository root.
#
#   bash scripts/downstream/pretrained/birdnet/jobs/submit_birdnet_evaluation.sh
#
# Optional environment variables:
#   CODECS="original encodec dac mp3 opus"   conditions to evaluate
#   WALLTIME=12:00:00
#   CONDA_ENV=conda-birdnet-env
#   DRY_RUN=1                                print the qsub command without submitting

CODECS="${CODECS:-original encodec dac mp3 opus}"
WALLTIME="${WALLTIME:-12:00:00}"
JOB="scripts/downstream/pretrained/birdnet/jobs/run_birdnet_evaluation.sh"

mkdir -p logs
vars="CODECS=${CODECS// /:}"
[[ -n "$CONDA_ENV" ]] && vars+=",CONDA_ENV=$CONDA_ENV"

cmd=(qsub -N eval_birdnet -l "walltime=$WALLTIME" -l "select=1:ncpus=4:mem=16gb"
     -o logs/eval_birdnet.out -e logs/eval_birdnet.err -v "$vars" "$JOB")

if [[ -n "$DRY_RUN" ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
