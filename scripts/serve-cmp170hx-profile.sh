#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VLLM_WORKDIR="${VLLM_WORKDIR:-$HOME/vllm}"
export QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP}"
nvrtc_lib="$VLLM_WORKDIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
cuda_headers="$VLLM_WORKDIR/.venv/lib/python3.12/site-packages/nvidia/cu13/include"
export LD_LIBRARY_PATH="$nvrtc_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$nvrtc_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$cuda_headers${CPATH:+:$CPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN_HOST="${QWEN_HOST:-0.0.0.0}" QWEN_PORT="${QWEN_PORT:-8002}"
export QWEN_CONTEXT="${QWEN_CONTEXT:-243200}" QWEN_SEQS="${QWEN_SEQS:-1}"
# Intentionally safe by default: enable FP8 explicitly after backport validation.
export QWEN_KV_CACHE_DTYPE="${QWEN_KV_CACHE_DTYPE:-auto}"
export QWEN_BATCH_TOKENS="${QWEN_BATCH_TOKENS:-1024}" QWEN_GPU_MEMORY="${QWEN_GPU_MEMORY:-0.90}"
export QWEN_MTP="${QWEN_MTP:-3}" QWEN_EAGER="${QWEN_EAGER:-0}"
export QWEN_GRAPH_MODE="${QWEN_GRAPH_MODE:-FULL_AND_PIECEWISE}" QWEN_CAPTURE_SIZES="${QWEN_CAPTURE_SIZES:-[4]}"
export QWEN_PROMPT_TOKENS_DETAILS="${QWEN_PROMPT_TOKENS_DETAILS:-1}"
if [[ ${QWEN_PRINT_COMMAND:-0} != 1 ]]; then
    shopt -s nullglob
    shards=("$QWEN_MODEL_DIR"/*.safetensors)
    (( ${#shards[@]} == 35 )) || { echo 'Expected 35 safetensors shards' >&2; exit 1; }
    [[ -f "$QWEN_MODEL_DIR/mtp-model-00001-of-00002.safetensors" && -f "$QWEN_MODEL_DIR/mtp-model-00002-of-00002.safetensors" ]] || { echo 'Quantized MTP shards missing' >&2; exit 1; }
    bytes=0
    for shard in "${shards[@]}"; do (( bytes += $(stat -c %s -- "$shard") )); done
    (( bytes == 153012759768 )) || { echo 'Unexpected total shard size' >&2; exit 1; }
fi
exec bash "$script_dir/serve.sh"
