#!/bin/bash
set -a
source key.env
set +a

BASE_DIR=/projects/bcjw/pyao3
PROJECT_DIR=$BASE_DIR/odysseus_exp
SIF=$BASE_DIR/trl-dev.sif

# Shared caches -- reusable across projects
export HF_HOME=$BASE_DIR/.hf_cache
export HF_HUB_CACHE=$BASE_DIR/.hf_cache/hub
export TRITON_HOME=$BASE_DIR/.triton
export TRITON_CACHE_DIR=$BASE_DIR/.triton/cache
export TRITON_AUTOTUNE_CACHE_DIR=$BASE_DIR/.triton/autotune
export TORCH_HOME=$BASE_DIR/.torch

# Override host CUDA_HOME -- DeltaAI's HPC SDK path leaks in via Apptainer's
# default env passthrough and points to a path that doesn't exist inside the
# container. nvcc lives at /usr/local/cuda in the nvidia/cuda base image.
export CUDA_HOME=/usr/local/cuda

mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TRITON_AUTOTUNE_CACHE_DIR"
mkdir -p "$TORCH_HOME"

echo "[env] SIF=$SIF"
echo "[env] CUDA_HOME=$CUDA_HOME"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] HF_HUB_CACHE=$HF_HUB_CACHE"
echo "[env] TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "[env] TRITON_AUTOTUNE_CACHE_DIR=$TRITON_AUTOTUNE_CACHE_DIR"
echo "[env] TORCH_HOME=$TORCH_HOME"


if [ "$#" -gt 0 ]; then
    apptainer exec --nv \
        --bind /projects/bcjw/pyao3 \
        --env HF_HOME=$HF_HOME \
        --env HF_HUB_CACHE=$HF_HUB_CACHE \
        --env HF_TOKEN=$HF_TOKEN \
        --env TRITON_HOME=$TRITON_HOME \
        --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
        --env TRITON_AUTOTUNE_CACHE_DIR=$TRITON_AUTOTUNE_CACHE_DIR \
        --env TORCH_HOME=$TORCH_HOME \
        --env CUDA_HOME=$CUDA_HOME \
        "$SIF" bash -lc "$*"
else
    apptainer exec --nv \
        --bind /projects/bcjw/pyao3 \
        --env HF_HOME=$HF_HOME \
        --env HF_HUB_CACHE=$HF_HUB_CACHE \
        --env HF_TOKEN=$HF_TOKEN \
        --env TRITON_HOME=$TRITON_HOME \
        --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
        --env TRITON_AUTOTUNE_CACHE_DIR=$TRITON_AUTOTUNE_CACHE_DIR \
        --env TORCH_HOME=$TORCH_HOME \
        --env CUDA_HOME=$CUDA_HOME \
        "$SIF" bash
fi