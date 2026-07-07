#!/usr/bin/env bash
set -euo pipefail

PORT=8000
VLLM_SERVER_HOST="0.0.0.0"
MAX_MODEL_LEN=16384
VLLM_GPUS=0,1
TRAIN_GPUS=2,3
TP_SIZE=2
NUM_TRAIN_GPUS=2
GPU_MEM_UTIL=0.85
HEALTH_TIMEOUT=900
ACCELERATE_CONFIG=fsdp_config.yaml
LOG_DIR=logs

MODEL="Qwen/Qwen3-VL-4B-Instruct"
MAX_TURNS=10
NUM_GENERATIONS=4
PER_DEVICE_TRAIN_BATCH_SIZE=2
STEPS_PER_GENERATION=2
MAX_STEPS=28
LEARNING_RATE=1e-6
MAX_COMPLETION_LENGTH=1024
GRADIENT_CHECKPOINTING=true
USE_LIGER_KERNEL=true

SCRIPT_NAME="$(basename "$0" .sh)"
RUN_LOG_DIR="${LOG_DIR}/${SCRIPT_NAME}"
LOG_STAMP="$(date +%H%M%S)"
VLLM_LOG="${RUN_LOG_DIR}/vllm_${LOG_STAMP}.log"
TRAIN_LOG="${RUN_LOG_DIR}/train_${LOG_STAMP}.log"
mkdir -p "$RUN_LOG_DIR"
echo "[run] server log -> $VLLM_LOG"
echo "[run] trainer log -> $TRAIN_LOG"

VLLM_PID=""
VLLM_PGID=""

cleanup() {
    # signal the whole vLLM process group (covers TP workers spawned by the server)
    if [[ -n "${VLLM_PGID:-}" ]]; then
        echo "[run] stopping vLLM process group (pgid $VLLM_PGID)..."
        kill -TERM -"$VLLM_PGID" 2>/dev/null || true
        # give workers a few seconds to release GPU memory, then force-kill
        for _ in $(seq 1 10); do
            kill -0 -"$VLLM_PGID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -"$VLLM_PGID" 2>/dev/null || true
    fi

    pkill -KILL -f "trl vllm-serve" 2>/dev/null || true
    pkill -KILL -f "VllmEngineCore" 2>/dev/null || true
    pkill -KILL -f "EngineCore_DP" 2>/dev/null || true
    pkill -KILL -f "from_engine_args" 2>/dev/null || true
    pkill -KILL -f "vllm.*worker" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[run] starting vLLM server: model=$MODEL gpus=[$VLLM_GPUS] tp=$TP_SIZE port=$PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPUS" trl vllm-serve \
    --model "$MODEL" \
    --tensor_parallel_size "$TP_SIZE" \
    --host "$VLLM_SERVER_HOST" --port "$PORT" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --trust_remote_code \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
VLLM_PGID=$(ps -o pgid= -p "$VLLM_PID" | tr -d ' ')
echo "[run] vLLM server pid=$VLLM_PID pgid=$VLLM_PGID"

echo "[run] waiting for vLLM server (timeout ${HEALTH_TIMEOUT}s)..."
elapsed=0
until curl -sfL "http://localhost:${PORT}/health/" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run] ERROR: vLLM server exited before becoming healthy."
        echo "----- last 40 lines of $VLLM_LOG -----"; tail -n 40 "$VLLM_LOG"
        exit 1
    fi
    if (( elapsed >= HEALTH_TIMEOUT )); then
        echo "[run] ERROR: vLLM server not healthy after ${HEALTH_TIMEOUT}s."
        echo "----- last 40 lines of $VLLM_LOG -----"; tail -n 40 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
    elapsed=$(( elapsed + 5 ))
done
echo "[run] vLLM server healthy after ${elapsed}s."

echo "[run] starting trainer: gpus=[$TRAIN_GPUS]"

TRAIN_ARGS=(
    --model "$MODEL"
    --max_turns "$MAX_TURNS"
    --num_generations "$NUM_GENERATIONS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --steps_per_generation "$STEPS_PER_GENERATION"
    --max_steps "$MAX_STEPS"
    --learning_rate "$LEARNING_RATE"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
)

if [[ "$GRADIENT_CHECKPOINTING" == true ]]; then
    TRAIN_ARGS+=(--gradient_checkpointing)
fi

if [[ "$USE_LIGER_KERNEL" == true ]]; then
    TRAIN_ARGS+=(--use_liger_kernel)
fi

status=0
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_TRAIN_GPUS" \
    --num_machines 1 \
    train.py \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$TRAIN_LOG"
status=${PIPESTATUS[0]}

echo "[run] trainer exited with status $status."
exit "$status"
