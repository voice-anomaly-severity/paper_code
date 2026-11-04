#!/usr/bin/env bash
# Start the full analysis pipeline detached from the terminal (survives logout).
# Usage: ./launch.sh [GPU ids as in nvidia-smi, default "0"] [extra run.py args]
# Progress: tail -f <output_dir>/logs/run.log ; GP workers: <output_dir>/logs/gp_worker_*.log
set -euo pipefail
cd "$(dirname "$0")"
GPUS="${1:-0}"
shift || true
LOG_DIR="$(.venv/bin/python -c 'from pipeline.config import load_config; print(load_config("config.yaml").path("output_dir"))')/logs"
mkdir -p "$LOG_DIR" .cache/mpl
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" MPLCONFIGDIR="$PWD/.cache/mpl" PYTHONWARNINGS=ignore \
  setsid nohup .venv/bin/python run.py "$@" > "$LOG_DIR/run.log" 2>&1 < /dev/null &
echo "started PID $! on GPUs $GPUS; log: $LOG_DIR/run.log"
