#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KT_ROOT="$ROOT"
cd "$KT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PY="${PY:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUN_NAME="${RUN_NAME:-Qwen2_5_7B_Instruct}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}"
NPROC="${NPROC:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29760}"

RUN="runs/${RUN_NAME}"
FOK="$RUN/stage2/fok_lora"
EVAL_ROOT="$RESULT_ROOT/${RUN_NAME}_stage2"
TRAJ_ROOT="$EVAL_ROOT/trajectory"
mkdir -p "$TRAJ_ROOT"

SCHEDULE=(base checkpoint-241 checkpoint-482 checkpoint-723 checkpoint-964 checkpoint-1205 final)

idx=0
for ADAPTER in "${SCHEDULE[@]}"; do
  idx=$((idx + 1))
  ADIR="$TRAJ_ROOT/$ADAPTER"
  mkdir -p "$ADIR"
  if [ "$ADAPTER" = base ]; then
    INPUT="$RUN/stage1/knowledge_answers_by_adapter/base.jsonl"
    KARGS=()
  elif [ "$ADAPTER" = final ]; then
    INPUT="$RUN/stage1/knowledge_answers.jsonl"
    KARGS=(--knowledge_lora "$RUN/stage1/knowledge_lora")
  else
    INPUT="$RUN/stage1/knowledge_answers_by_adapter/$ADAPTER.jsonl"
    KARGS=(--knowledge_lora "$RUN/stage1/knowledge_lora/$ADAPTER")
  fi
  echo "[score] $ADAPTER input=$INPUT"
  "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$((MASTER_PORT_BASE + idx))" -m src.inference.fok_yesno_score \
    --base_model "$MODEL" \
    "${KARGS[@]}" \
    --fok_lora "$FOK" \
    --input_jsonl "$INPUT" \
    --output_jsonl "$ADIR/scores.jsonl" \
    --metrics_json "$ADIR/metrics_all.json" \
    --batch_size 16 \
    --max_prompt_len 512 \
    --dtype bfloat16
done

"$PY" - <<PY
import json
from pathlib import Path
from src.baselines.pipeline.common import compute_auroc, compute_threshold_metrics

run = Path("$RUN")
out = Path("$TRAJ_ROOT")
schedule = ["base", "checkpoint-241", "checkpoint-482", "checkpoint-723", "checkpoint-964", "checkpoint-1205", "final"]
qid_split = {}
for line in (run / "data/queries.jsonl").open():
    if line.strip():
        row = json.loads(line)
        qid_split[str(row.get("id", row.get("qid")))] = row.get("split", "train")

def calc(rows):
    labels = [int(row["is_correct"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    tm = compute_threshold_metrics(scores, labels, 0.5)
    return {
        "n": len(rows),
        "answer_acc": sum(labels) / max(len(labels), 1),
        "auroc": compute_auroc(scores, labels),
        "threshold_acc": tm["accuracy"],
        "illusion": 1.0 - tm["tnr"],
        "omission": 1.0 - tm["tpr"],
        "bias": ((1.0 - tm["tnr"]) + (1.0 - tm["tpr"])) / 2 * (1 if (1.0 - tm["tnr"]) >= (1.0 - tm["tpr"]) else -1),
        "pos": sum(labels),
        "neg": len(labels) - sum(labels),
    }

summary = []
for adapter in schedule:
    rows = []
    for line in (out / adapter / "scores.jsonl").open():
        if not line.strip():
            continue
        row = json.loads(line)
        row["split"] = qid_split[str(row["qid"])]
        rows.append(row)
    item = {"adapter": adapter}
    for split in ["train", "val", "all"]:
        sub = rows if split == "all" else [row for row in rows if row["split"] == split]
        metrics = calc(sub)
        item[split] = metrics
        (out / adapter / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2) + "\\n")
    summary.append(item)

(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
lines = []
headers = ["adapter", "train_acc", "train_auroc", "train_illu", "train_om", "val_acc", "val_auroc", "val_illu", "val_om"]
lines.append("| " + " | ".join(headers) + " |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for item in summary:
    tr = item["train"]
    va = item["val"]
    lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
        item["adapter"], tr["answer_acc"], tr["auroc"], tr["illusion"], tr["omission"],
        va["answer_acc"], va["auroc"], va["illusion"], va["omission"]))
(out / "summary.md").write_text("\\n".join(lines) + "\\n")
print("\\n".join(lines))
print(f"[done] {out}")
PY
