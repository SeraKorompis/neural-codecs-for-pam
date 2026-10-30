#!/bin/bash
# Submit BirdNET detection jobs for Northeastern Birds. Run from the repository root.
#
#   bash scripts/downstream/pretrained/birdnet/jobs/submit_birdnet_detection.sh
#
# One job for the original audio, then one per (codec, bitrate) for the reconstructions.
# Jobs are chained (each starts only after the previous one succeeds), because they all
# append to the same CSV file and must not run at the same time. The script resumes
# where it stopped, so a failed chain can simply be submitted again.
#
# Optional environment variables:
#   STEPS="original reconstructed"   which parts to run
#   CODECS="encodec dac mp3 opus"
#   N_FILES=3                        limit to the first N files (quick tests only)
#   WALLTIME=16:00:00
#   CONDA_ENV=conda-birdnet-env
#   DRY_RUN=1                        print the qsub commands without submitting

STEPS="${STEPS:-original reconstructed}"
CODECS="${CODECS:-encodec dac mp3 opus}"
WALLTIME="${WALLTIME:-16:00:00}"
JOB="scripts/downstream/pretrained/birdnet/jobs/run_birdnet_detection.sh"

declare -A BITRATES=(
    [encodec]="1.5 3.0 6.0 12.0 24.0"
    [dac]="2 3 6 9"
    [mp3]="8 16 24"
    [opus]="8 14 24"
)

mkdir -p logs
prev=""
n=0

submit() {
    local name="$1" vars="$2"
    [[ -n "$N_FILES" ]]   && vars+=",N_FILES=$N_FILES"
    [[ -n "$CONDA_ENV" ]] && vars+=",CONDA_ENV=$CONDA_ENV"
    local cmd=(qsub -N "$name" -l "walltime=$WALLTIME" -l "select=1:ncpus=4:mem=32gb"
               -o "logs/$name.out" -e "logs/$name.err" -v "$vars")
    [[ -n "$prev" ]] && cmd+=(-W "depend=afterok:$prev")
    cmd+=("$JOB")
    if [[ -n "$DRY_RUN" ]]; then
        echo "${cmd[*]}"
        prev="<previous_job_id>"
    else
        prev=$("${cmd[@]}") || { echo "qsub failed for $name"; exit 1; }
        echo "$name -> $prev"
    fi
    n=$((n+1))
}

for step in $STEPS; do
    if [[ "$step" == "original" ]]; then
        submit "bn_original" "STEP=original"
    elif [[ "$step" == "reconstructed" ]]; then
        for codec in $CODECS; do
            for br in ${BITRATES[$codec]}; do
                submit "bn_${codec}_${br}" "STEP=reconstructed,CODECS=$codec,BITRATES=$br"
            done
        done
    fi
done
echo "$n jobs"
