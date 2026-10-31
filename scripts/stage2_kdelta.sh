#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KT_ROOT="$ROOT"
cd "$KT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

BASE_MODEL="${BASE_MODEL:-qwen25_7b}"
case "$BASE_MODEL" in
  qwen25_7b)
    _MODEL_DEFAULT="Qwen/Qwen2.5-7B-Instruct"
    _RUN_TAG="Qwen2_5_7B_Instruct"
    ;;
  qwen3_8b)
    _MODEL_DEFAULT="Qwen/Qwen3-8B"
    _RUN_TAG="Qwen3_8B"
    ;;
  *)
    echo "[error] unknown BASE_MODEL=$BASE_MODEL (expected qwen25_7b or qwen3_8b)" >&2
    exit 1
    ;;
esac

MODEL="${MODEL:-$_MODEL_DEFAULT}"
RUN_NAME="${RUN_NAME:-${_RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29751}"

STAGE2_LR="${STAGE2_LR:-1e-5}"
STAGE2_BS="${STAGE2_BS:-8}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
AUX_N="${AUX_N:-1000}"

RUN="runs/${RUN_NAME}"
AUX_ROOT="${AUX_ROOT:-$RUN/stage2/aux}"
OUT_ROOT="$RUN/stage2/data"
DATA_ROOT="$OUT_ROOT/by_adapter"
FOK_OUT="$RUN/stage2/fok_lora"
EVAL_ROOT="$RESULT_ROOT/${RUN_NAME}_stage2"
TRAJ_ROOT="$EVAL_ROOT/trajectory"
SCHEDULE=(base checkpoint-241 checkpoint-482 checkpoint-723 checkpoint-964 checkpoint-1205 final)

mkdir -p "$OUT_ROOT/combined_answers_by_adapter" "$DATA_ROOT" "$TRAJ_ROOT"

for ADAPTER in "${SCHEDULE[@]}"; do
  if [ "$ADAPTER" = base ]; then
    OLD="$RUN/stage1/knowledge_answers_by_adapter/base.jsonl"
  elif [ "$ADAPTER" = final ]; then
    OLD="$RUN/stage1/knowledge_answers.jsonl"
  else
    OLD="$RUN/stage1/knowledge_answers_by_adapter/$ADAPTER.jsonl"
  fi
  AUX="$AUX_ROOT/answers_by_adapter/$ADAPTER.jsonl"
  COMB="$OUT_ROOT/combined_answers_by_adapter/$ADAPTER.jsonl"
  DATA="$DATA_ROOT/$ADAPTER"
  if [ ! -s "$AUX" ]; then
    echo "[error] missing aux answers: $AUX" >&2
    exit 1
  fi
  "$PY" - <<PY
import json
from pathlib import Path
old = Path("$OLD")
aux = Path("$AUX")
out = Path("$COMB")
aux_n = int("$AUX_N")
out.parent.mkdir(parents=True, exist_ok=True)
n_old = n_aux = pos = 0
with out.open("w") as f:
    for line in old.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
        n_old += 1
        pos += int(row["is_correct"])
    for line in aux.open():
        if not line.strip() or n_aux >= aux_n:
            continue
        row = json.loads(line)
        row["split"] = "aux"
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
        n_aux += 1
        pos += int(row["is_correct"])
print(f"[combine] {out} train={n_old} aux={n_aux} pos={pos} neg={n_old+n_aux-pos}")
PY
  "$PY" -m src.training.fok_data \
    --answers_jsonl "$COMB" \
    --model "$MODEL" \
    --output_dir "$DATA" \
    --split all \
    --max_seq_len 512 \
    --balance none
done

"$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" -m src.training.kdelta_fok_lora_mixed \
  --model "$MODEL" \
  --knowledge_lora_root "$RUN/stage1/knowledge_lora" \
  --data_root "$DATA_ROOT" \
  --output_dir "$FOK_OUT" \
  --schedule "${SCHEDULE[@]}" \
  --method_version "kdelta_fok_lora_v1" \
  --epochs "$STAGE2_EPOCHS" \
  --per_device_train_batch_size "$STAGE2_BS" \
  --learning_rate "$STAGE2_LR" \
  --lora_rank "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --dtype bfloat16 \
  --logging_steps 20 \
  ${FORCE:+--force}

"$ROOT/scripts/eval_stage2.sh"
