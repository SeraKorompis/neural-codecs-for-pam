#!/bin/bash
# Submit one job per (codec, bitrate, split) for the pre-processed AnuraSet clips.
# Run from the repository root.
#
#   bash scripts/reconstruction/jobs/submit_compress_anuraset.sh
#
# Optional environment variables:
#   SPLITS="train test"    which splits to compress
#   CODECS="encodec dac mp3 opus"
#   MAX_FILES=20           limit clips per job (for quick tests)
#   DRY_RUN=1              print the qsub commands without submitting

SPLITS="${SPLITS:-train test}"
CODECS="${CODECS:-encodec dac mp3 opus}"
WALLTIME="24:00:00"
JOB="scripts/reconstruction/jobs/compress_anuraset.sh"

declare -A BITRATES=(
    [encodec]="1.5 3.0 6.0 12.0 24.0"
    [dac]="2 3 6 9"
    [mp3]="8 16 24"
    [opus]="8 14 24"
)

mkdir -p logs
n=0
for codec in $CODECS; do
    if [[ "$codec" == "encodec" || "$codec" == "dac" ]]; then
        device=cuda
        res="select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S"
    else
        device=cpu
        res="select=1:ncpus=4:mem=16gb"
    fi
    for split in $SPLITS; do
        for br in ${BITRATES[$codec]}; do
            name="compress_anu_${codec}_${br}_${split}"
            cmd=(qsub -N "$name" -l "walltime=$WALLTIME" -l "$res"
                 -o "logs/$name.out" -e "logs/$name.err"
                 -v "CODEC=$codec,SPLIT=$split,BITRATE=$br,DEVICE=$device,MAX_FILES=${MAX_FILES:-}"
                 "$JOB")
            if [[ -n "$DRY_RUN" ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
            n=$((n+1))
        done
    done
done
echo "$n jobs"
