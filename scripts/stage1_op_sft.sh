#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KT_ROOT="$ROOT"
cd "$KT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-python}"

BASE_MODEL="${BASE_MODEL:-qwen25_7b}"
case "$BASE_MODEL" in
  qwen25_7b)
    _MODEL_DEFAULT="Qwen/Qwen2.5-7B-Instruct"
    _RUN_TAG="Qwen2_5_7B_Instruct"
    _TEMPLATE="qwen"
    _OPSFT_DEFAULT="/path/to/opsft_samples/qwen25_7b"
    _NUM_LAYERS=28
    ;;
  qwen3_8b)
    _MODEL_DEFAULT="Qwen/Qwen3-8B"
    _RUN_TAG="Qwen3_8B"
    _TEMPLATE="qwen3"
    _OPSFT_DEFAULT="/path/to/opsft_samples/qwen3_8b"
    _NUM_LAYERS=36
    ;;
  *)
    echo "[error] unknown BASE_MODEL=$BASE_MODEL (expected qwen25_7b or qwen3_8b)" >&2
    exit 1
    ;;
esac

FOK_LAYER="${FOK_LAYER:-$(( _NUM_LAYERS * 2 / 3 ))}"
LORA_LAYERS="${LORA_LAYERS:-$(python3 -c "print(','.join(str(i) for i in range(0, $FOK_LAYER + 1)))")}"

MODEL="${MODEL:-$_MODEL_DEFAULT}"
RUN_NAME="${RUN_NAME:-${_RUN_TAG}}"
TEMPLATE="${TEMPLATE:-$_TEMPLATE}"
OPSFT_SOURCE_DIR="${OPSFT_SOURCE_DIR:-$_OPSFT_DEFAULT}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-third_party/LlamaFactory}"
NPROC="${NPROC:-8}"

RUN="runs/${RUN_NAME}"
DATA_DIR="$RUN/data"
STAGE1_DIR="$RUN/stage1"
SOURCE_CORRECTED="${OPSFT_CORRECTED_FILE:-$OPSFT_SOURCE_DIR/step2_corrected/corrected_all.jsonl}"
SOURCE_ANSWERS="${OPSFT_ANSWERS_FILE:-$OPSFT_SOURCE_DIR/step1_responses/responses_all.jsonl}"
TRAIN_EVAL="${OPSFT_TRAIN_EVAL_FILE:-$OPSFT_SOURCE_DIR/step4_eval_train/responses_all.jsonl}"
VAL_EVAL="${OPSFT_VAL_EVAL_FILE:-$OPSFT_SOURCE_DIR/step4_eval_val/responses_all.jsonl}"

TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-5000}"
VAL_QUESTIONS="${VAL_QUESTIONS:-1000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
STAGE1_LR="${STAGE1_LR:-2e-5}"
STAGE1_BS="${STAGE1_BS:-8}"
STAGE1_GA="${STAGE1_GA:-1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-241}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
DEEPSPEED="${DEEPSPEED:-src/configs/ds_config_zero2.json}"
METHOD_VERSION="${METHOD_VERSION:-opsft_lora_v1}"

mkdir -p "$DATA_DIR" "$STAGE1_DIR"

"$PY" - <<PY
import json
from pathlib import Path

train_eval = Path("$TRAIN_EVAL")
val_eval = Path("$VAL_EVAL")
data_dir = Path("$DATA_DIR")
train_n = int("$TRAIN_QUESTIONS")
val_n = int("$VAL_QUESTIONS")

def row_from_source(row, split):
    answers = row.get("answers") or []
    primary = row.get("primary_answer") or ""
    if not primary:
        for answer in answers:
            if str(answer or "").strip():
                primary = answer
                break
    return {
        "id": str(row.get("id", row.get("qid", ""))),
        "question": row.get("question", ""),
        "answers": answers,
        "primary_answer": primary,
        "split": split,
    }

def read_rows(path, split, limit):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(row_from_source(json.loads(line), split))
            if len(rows) >= limit:
                break
    if len(rows) != limit:
        raise SystemExit(f"[error] {path} yielded {len(rows)} rows, expected {limit}")
    return rows

train = read_rows(train_eval, "train", train_n)
val = read_rows(val_eval, "val", val_n)
data_dir.mkdir(parents=True, exist_ok=True)
with (data_dir / "queries.jsonl").open("w") as all_f, \
        (data_dir / "train_queries.jsonl").open("w") as train_f, \
        (data_dir / "val_queries.jsonl").open("w") as val_f:
    for row in train:
        text = json.dumps(row, ensure_ascii=False) + "\n"
        all_f.write(text)
        train_f.write(text)
    for row in val:
        text = json.dumps(row, ensure_ascii=False) + "\n"
        all_f.write(text)
        val_f.write(text)
print(f"[stage1] queries train={len(train)} val={len(val)} -> {data_dir / 'queries.jsonl'}")
PY

"$PY" -m src.training.op_sft_lora \
  --model "$MODEL" \
  --corrected_file "$SOURCE_CORRECTED" \
  --answers_file "$SOURCE_ANSWERS" \
  --dataset_json "$DATA_DIR/opsft_lf_data/opsft_replacement.json" \
  --data_meta "$DATA_DIR/opsft_meta.json" \
  --output_dir "$STAGE1_DIR/knowledge_lora" \
  --method_version "$METHOD_VERSION" \
  --llamafactory_root "$LLAMAFACTORY_ROOT" \
  --nproc "$NPROC" \
  --template "$TEMPLATE" \
  --cutoff_len "$MAX_SEQ_LEN" \
  --lora_rank "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout 0.0 \
  --learning_rate "$STAGE1_LR" \
  --num_train_epochs "$STAGE1_EPOCHS" \
  --per_device_train_batch_size "$STAGE1_BS" \
  --gradient_accumulation_steps "$STAGE1_GA" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --deepspeed "$DEEPSPEED" \
  --drop_correct \
  --lora_layers_to_transform "$LORA_LAYERS" \
  ${FORCE:+--force}

echo "[stage1] knowledge LoRA: $KT_ROOT/$STAGE1_DIR/knowledge_lora"
echo "[stage1] lora_layers=$LORA_LAYERS"

