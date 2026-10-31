#!/bin/bash
# Submit BirdNET embedding extraction jobs. Run from the repository root.
#
#   bash scripts/downstream/transfer_learned/jobs/submit_extract_embeddings.sh
#
# For each (dataset, split) the original-audio job runs first; the compressed
# jobs start only after it succeeds.
#
# Optional environment variables:
#   DATASETS="anuraset northeastern lemur"
#   SPLITS="train test"        overrides the default splits for every dataset
#   SOURCES="original encodec dac mp3 opus"
#   LIMIT=5                    first N clips/files (quick tests only)
#   WALLTIME=24:00:00   CONDA_ENV=conda-birdnet-env
#   DRY_RUN=1                  print the qsub commands without submitting

DATASETS="${DATASETS:-anuraset northeastern lemur}"
SOURCES="${SOURCES:-original encodec dac mp3 opus}"
WALLTIME="${WALLTIME:-24:00:00}"
JOB="scripts/downstream/transfer_learned/jobs/extract_embeddings.sh"

declare -A DEFAULT_SPLITS=([anuraset]="train test" [northeastern]="all" [lemur]="all")
declare -A BITRATES=([encodec]="1.5 3.0 6.0 12.0 24.0" [dac]="2 3 6 9" [mp3]="8 16 24" [opus]="8 14 24")

mkdir -p logs
n=0

submit() {   # submit NAME VARS DEPENDS_ON
    local name="$1" vars="$2" dep="$3"
    [[ -n "$LIMIT" ]]     && vars+=",LIMIT=$LIMIT"
    [[ -n "$CONDA_ENV" ]] && vars+=",CONDA_ENV=$CONDA_ENV"
    local cmd=(qsub -N "$name" -l "walltime=$WALLTIME" -l "select=1:ncpus=4:mem=32gb"
               -o "logs/$name.out" -e "logs/$name.err" -v "$vars")
    [[ -n "$dep" ]] && cmd+=(-W "depend=afterok:$dep")
    cmd+=("$JOB")
    if [[ -n "$DRY_RUN" ]]; then echo "${cmd[*]}"; JOBID="<original_job_id>"
    else JOBID=$("${cmd[@]}") || { echo "qsub failed for $name"; exit 1; }; echo "$name -> $JOBID"; fi
    n=$((n+1))
}

for ds in $DATASETS; do
    for split in ${SPLITS:-${DEFAULT_SPLITS[$ds]}}; do
        orig_id=""
        for src in $SOURCES; do
            if [[ "$src" == "original" ]]; then
                submit "bn_emb_${ds}_${split}_original" "DATASET=$ds,SPLIT=$split,SOURCE=original" ""
                orig_id="$JOBID"
            fi
        done
        for src in $SOURCES; do
            [[ "$src" == "original" ]] && continue
            for br in ${BITRATES[$src]}; do
                submit "bn_emb_${ds}_${split}_${src}${br}" "DATASET=$ds,SPLIT=$split,SOURCE=$src,BITRATE=$br" "$orig_id"
            done
        done
    done
done
echo "$n jobs"
