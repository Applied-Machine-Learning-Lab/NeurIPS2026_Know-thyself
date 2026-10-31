#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KT_ROOT="$ROOT"
cd "$KT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-python}"
EASYR1_ROOT="${EASYR1_ROOT:-$ROOT/third_party/EasyR1}"

BASE_MODEL="${BASE_MODEL:-qwen25_7b}"
case "$BASE_MODEL" in
  qwen25_7b) _RUN_TAG="Qwen2_5_7B_Instruct" ;;
  qwen3_8b)  _RUN_TAG="Qwen3_8B" ;;
  *)
    echo "[error] unknown BASE_MODEL=$BASE_MODEL (expected qwen25_7b or qwen3_8b)" >&2
    exit 1
    ;;
esac

RUN_NAME="${RUN_NAME:-${_RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}"
NPROC="${NPROC:-8}"

TAG="${TAG:-fgpo}"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-1}"
STAGE3_LR="${STAGE3_LR:-5e-7}"
STAGE3_LORA_RANK="${STAGE3_LORA_RANK:-16}"
STAGE3_ROLLOUT_N="${STAGE3_ROLLOUT_N:-2}"
STAGE3_ROLLOUT_BS="${STAGE3_ROLLOUT_BS:-64}"
STAGE3_GLOBAL_BS="${STAGE3_GLOBAL_BS:-64}"
STAGE3_MAX_PROMPT_LEN="${STAGE3_MAX_PROMPT_LEN:-1024}"
STAGE3_MAX_RESPONSE_LEN="${STAGE3_MAX_RESPONSE_LEN:-160}"
STAGE3_ROLLOUT_TP="${STAGE3_ROLLOUT_TP:-1}"
STAGE3_GPU_MEM="${STAGE3_GPU_MEM:-0.70}"


RUN="runs/${RUN_NAME}"
QUERIES="${QUERIES:-$RUN/data/fgpo_queries.jsonl}"
INIT_SOURCE="${INIT_MODEL:-$RUN/stage2/merged_model}"

STAGE3_DIR="$RUN/stage3"
STAGE3_DATA="$STAGE3_DIR/data"
STAGE3_INIT="$STAGE3_DIR/init_model"
STAGE3_CKPT="$STAGE3_DIR/checkpoints"
STAGE3_FINAL="$STAGE3_DIR/final_model"
OUT_DIR="$RESULT_ROOT/${RUN_NAME}_stage3"
OUT_JSONL="$OUT_DIR/ours_stage3.jsonl"

if [ ! -s "$QUERIES" ]; then
  echo "[error] missing queries: $QUERIES" >&2
  exit 1
fi
if [ ! -s "$INIT_SOURCE/config.json" ]; then
  echo "[error] missing init model config: $INIT_SOURCE/config.json" >&2
  echo "        (expected the stage1+stage2 merged model)" >&2
  exit 1
fi

mkdir -p "$STAGE3_DIR" "$OUT_DIR"

if [ ! -e "$STAGE3_INIT" ]; then
  ln -s "$(cd "$INIT_SOURCE" && pwd)" "$STAGE3_INIT"
  echo "[stage3] linked init_model -> $INIT_SOURCE"
fi

echo "[stage3] run=$RUN_NAME tag=$TAG"
echo "[stage3] init_model=$STAGE3_INIT (source=$INIT_SOURCE)"
echo "[stage3] lr=$STAGE3_LR max_response_length=$STAGE3_MAX_RESPONSE_LEN"

"$PY" -m src.training.fgpo prepare \
  --queries_jsonl "$QUERIES" \
  --output_dir "$STAGE3_DATA"

"$PY" -m src.training.fgpo launch \
  --data_dir "$STAGE3_DATA" \
  --model "$STAGE3_INIT" \
  --easyr1_root "$EASYR1_ROOT" \
  --output_dir "$STAGE3_DIR" \
  --experiment_name "fgpo_${TAG}" \
  --epochs "$STAGE3_EPOCHS" \
  --rollout_n "$STAGE3_ROLLOUT_N" \
  --rollout_batch_size "$STAGE3_ROLLOUT_BS" \
  --global_batch_size "$STAGE3_GLOBAL_BS" \
  --lr "$STAGE3_LR" \
  --lora_rank "$STAGE3_LORA_RANK" \
  --max_prompt_length "$STAGE3_MAX_PROMPT_LEN" \
  --max_response_length "$STAGE3_MAX_RESPONSE_LEN" \
  --n_gpus_per_node "$NPROC" \
  --rollout_tensor_parallel_size "$STAGE3_ROLLOUT_TP" \
  --rollout_gpu_memory_utilization "$STAGE3_GPU_MEM" \
  --save_checkpoint_path "$STAGE3_CKPT" \
  --execute

has_model_weights() {
  local model_dir="$1"
  [ -s "$model_dir/config.json" ] || return 1
  [ -s "$model_dir/model.safetensors" ] && return 0
  [ -s "$model_dir/pytorch_model.bin" ] && return 0
  ls "$model_dir"/model-*.safetensors >/dev/null 2>&1 && return 0
  ls "$model_dir"/pytorch_model-*.bin >/dev/null 2>&1 && return 0
  return 1
}

ACTOR="$(find "$STAGE3_CKPT" -path '*/actor/model_world_size_*_rank_0.pt' -print 2>/dev/null | sort | tail -1 | sed 's#/model_world_size_.*_rank_0.pt$##')"
if [ -z "$ACTOR" ]; then
  echo "[error] no EasyR1 actor checkpoint under $STAGE3_CKPT" >&2
  exit 1
fi

ACTOR_LORA="$ACTOR/lora_adapter"
HF_MODEL="$ACTOR/huggingface"
if [ -s "$ACTOR_LORA/adapter_config.json" ]; then
  rm -rf "$STAGE3_FINAL"
  "$PY" -m src.training.merge_lora \
    --base_model "$STAGE3_INIT" \
    --adapter "$ACTOR_LORA" \
    --output_dir "$STAGE3_FINAL" \
    --force
else
  if ! has_model_weights "$HF_MODEL"; then
    if [ ! -s "$EASYR1_ROOT/scripts/model_merger.py" ]; then
      if [ -s "$ROOT/third_party/EasyR1/scripts/model_merger.py" ]; then
        EASYR1_ROOT="$ROOT/third_party/EasyR1"
      elif [ -s "/path/to/EasyR1/scripts/model_merger.py" ]; then
        EASYR1_ROOT="/path/to/EasyR1"
      fi
    fi
    rm -rf "$HF_MODEL"
    "$PY" "$EASYR1_ROOT/scripts/model_merger.py" --local_dir "$ACTOR"
  fi
  if ! has_model_weights "$HF_MODEL"; then
    echo "[error] merged EasyR1 model incomplete: $HF_MODEL" >&2
    exit 1
  fi

  rm -rf "$STAGE3_FINAL"
  ln -s "$(cd "$HF_MODEL" && pwd)" "$STAGE3_FINAL"
fi

if ! has_model_weights "$STAGE3_FINAL"; then
  echo "[error] final Stage3 model has no weights: $STAGE3_FINAL" >&2
  exit 1
fi

"$PY" - <<PY
import json
from pathlib import Path
stage3 = Path("$STAGE3_DIR")
meta = {
    "tag": "$TAG",
    "run_name": "$RUN_NAME",
    "init_source": "$INIT_SOURCE",
    "init_model": "$STAGE3_INIT",
    "checkpoint_actor": "$ACTOR",
    "checkpoint_lora": "$ACTOR_LORA",
    "final_model": "$STAGE3_FINAL",
    "stage3_lr": "$STAGE3_LR",
    "stage3_rollout_n": "$STAGE3_ROLLOUT_N",
    "stage3_rollout_batch_size": "$STAGE3_ROLLOUT_BS",
    "stage3_global_batch_size": "$STAGE3_GLOBAL_BS",
    "stage3_max_response_length": "$STAGE3_MAX_RESPONSE_LEN",
    "reward": "judgment_answer_think",
}
(stage3 / "stage3_selected_meta.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False) + "\\n")
PY

"$PY" -m src.inference.stage3_reasoning_inference_vllm \
  --base_model "$STAGE3_FINAL" \
  --queries_jsonl "$QUERIES" \
  --split all \
  --output "$OUT_JSONL" \
  --nprocs "$NPROC" \
  --dtype bfloat16 \
  --max_model_len 2048 \
  --gpu_memory_utilization 0.90 \
  --max_new_tokens "$STAGE3_MAX_RESPONSE_LEN"

echo "[done] stage3 result: $OUT_JSONL"
