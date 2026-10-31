#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KT_ROOT="$ROOT"
cd "$KT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUN_NAME="${RUN_NAME:-Qwen2_5_7B_Instruct}"
NPROC="${NPROC:-8}"
VLLM_MEM="${VLLM_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

RUN="runs/${RUN_NAME}"
QUERIES="$RUN/data/queries.jsonl"
STAGE1_DIR="$RUN/stage1"
ANS_DIR="$STAGE1_DIR/knowledge_answers_by_adapter"
mkdir -p "$ANS_DIR"

SCHEDULE=(base checkpoint-241 checkpoint-482 checkpoint-723 checkpoint-964 checkpoint-1205 final)

for ADAPTER in "${SCHEDULE[@]}"; do
  if [ "$ADAPTER" = base ]; then
    BASE_MODEL="$MODEL"
    KARGS=()
    OUT="$ANS_DIR/base.jsonl"
  elif [ "$ADAPTER" = final ]; then
    BASE_MODEL="$MODEL"
    KARGS=(--knowledge_lora "$STAGE1_DIR/knowledge_lora")
    OUT="$STAGE1_DIR/knowledge_answers.jsonl"
  else
    BASE_MODEL="$MODEL"
    KARGS=(--knowledge_lora "$STAGE1_DIR/knowledge_lora/$ADAPTER")
    OUT="$ANS_DIR/$ADAPTER.jsonl"
  fi

  echo "[stage1-eval] $ADAPTER -> $OUT"
  "$PY" -m src.inference.ours_inference_vllm \
    --base_model "$BASE_MODEL" \
    "${KARGS[@]}" \
    --queries_jsonl "$QUERIES" \
    --split all \
    --nprocs "$NPROC" \
    --dtype bfloat16 \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$VLLM_MEM" \
    --output "$OUT" \
    --adapter_arch lf_knowledge_lora_v2

  "$PY" -m src.inference.relabel_answer_containment \
    --input "$OUT" \
    --queries_jsonl "$QUERIES"
done

