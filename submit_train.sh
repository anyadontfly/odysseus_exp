#!/bin/bash
#SBATCH -A bcjw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=odysseus-grpo
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /projects/bcjw/pyao3/odysseus_exp

mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

RUN_SCRIPT="${1:-./run_train.sh}"
if [[ $# -gt 0 ]]; then
    shift
fi

RUN_CMD=("$RUN_SCRIPT" "$@")
printf -v RUN_CMD_STR "%q " "${RUN_CMD[@]}"
RUN_CMD_STR="${RUN_CMD_STR% }"

echo "Container command: $RUN_CMD_STR"
./scripts/run_container.sh "$RUN_CMD_STR"

echo "Job finished at: $(date)"