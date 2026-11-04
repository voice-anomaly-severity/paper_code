#!/bin/sh
# Cross-fitting: 5 random splits x 5 folds of the healthy speakers (25 models).
# Usage: ./run_all.sh <healthy clips> <patient clips> [GPU ids, default "0"] [workers per GPU, default 1]
set -e
HEALTHY="$(realpath "$1")"; PATIENTS="$(realpath "$2")"; GPUS="${3:-0}"; PER_GPU="${4:-1}"
cd "$(dirname "$0")"
set -- $(echo "$GPUS" | tr ',' ' ')
NGPU=$#; WORKERS=$((NGPU * PER_GPU)); GPU_LIST="$*"
mkdir -p runs
# build the shared cache of 16-kHz waveforms once, before starting parallel workers
python run_fold.py --healthy-clips "$HEALTHY" --patient-clips "$PATIENTS" --out runs --fold 0 --partition 0 > runs/log_p0_f0.txt 2>&1
w=0
while [ $w -lt $WORKERS ]; do
  gpu=$(echo $GPU_LIST | cut -d' ' -f$((w % NGPU + 1)))
  (j=0; for p in 0 1 2 3 4; do for f in 0 1 2 3 4; do
     if [ $j -gt 0 ] && [ $((j % WORKERS)) -eq $w ]; then
       CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python run_fold.py --healthy-clips "$HEALTHY" \
         --patient-clips "$PATIENTS" --out runs --fold $f --partition $p > runs/log_p${p}_f${f}.txt 2>&1
     fi; j=$((j + 1)); done; done) &
  w=$((w + 1))
done
wait
echo "all 25 models done"
