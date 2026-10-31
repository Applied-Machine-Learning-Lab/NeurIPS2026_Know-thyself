from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines.utils.answer_containment import (
    ANSWER_CONTAINMENT_LABEL_VERSION,
)


LF_KNOWLEDGE_LORA_ARCH_VERSION = "lf_knowledge_lora_v2"
LF_KNOWLEDGE_FULL_ARCH_VERSION = "lf_knowledge_full_corrected_suffix_eos_v10"
LF_STANDARD_KNOWLEDGE_LORA_ARCH_VERSION = "lf_standard_knowledge_lora_v1"
LF_TWO_LORA_ARCH_VERSION = "lf_two_lora_v2"
LF_FULL_KNOWLEDGE_FOK_LORA_ARCH_VERSION = "lf_full_knowledge_fok_lora_v2"
FOK_DATA_VERSION = "fok_yesno_lf_containment_v1"
FOK_SCORE_VERSION = "yes_no_token_prob_v2"


FOK_PROMPT = (
    "Can you answer the following question correctly? "
    "Answer with exactly Yes or No.\n\n"
    "Question: {question}"
)

ANSWERABILITY_PROMPT = (
    "Is the following question answerable? "
    "Answer with exactly Yes or No.\n\n"
    "Question: {question}"
)

PROMPT_STYLES = {"fok": FOK_PROMPT, "answerability": ANSWERABILITY_PROMPT}


def build_fok_prompt(tokenizer, question: str, style: str = "fok") -> str:
    body = PROMPT_STYLES[style].format(question=question)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_yes_no_sample(
    tokenizer,
    question: str,
    label: int,
    max_seq_len: int,
) -> Optional[Dict]:
    prompt = build_fok_prompt(tokenizer, question)
    answer = "Yes" if label else "No"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not answer_ids:
        return None
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_seq_len:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def _load_rows(path: Path, split: str) -> List[Dict]:
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        if split != "all" and row.get("split", "train") != split:
            continue
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _save_dataset(path: Path, rows: List[Dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list([
        {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }
        for row in rows
    ])
    DatasetDict({"train": ds}).save_to_disk(str(path))


def _balance_rows(rows: List[Dict], mode: str, seed: int) -> tuple[List[Dict], Dict]:
    positives = [row for row in rows if int(row["is_correct"]) == 1]
    negatives = [row for row in rows if int(row["is_correct"]) == 0]
    meta = {
        "balance": mode,
        "balance_seed": seed,
        "n_pos_before": len(positives),
        "n_neg_before": len(negatives),
    }
    if mode == "none" or not positives or not negatives:
        meta.update({
            "n_pos_after": len(positives),
            "n_neg_after": len(negatives),
        })
        return rows, meta

    rng = random.Random(seed)
    if mode == "downsample":
        n = min(len(positives), len(negatives))
        balanced = rng.sample(positives, n) + rng.sample(negatives, n)
    elif mode == "oversample":
        n = max(len(positives), len(negatives))
        pos = positives if len(positives) == n else positives + [
            rng.choice(positives) for _ in range(n - len(positives))
        ]
        neg = negatives if len(negatives) == n else negatives + [
            rng.choice(negatives) for _ in range(n - len(negatives))
        ]
        balanced = pos + neg
    else:
        raise ValueError(f"unknown balance mode: {mode}")

    rng.shuffle(balanced)
    meta.update({
        "n_pos_after": sum(int(row["is_correct"]) == 1 for row in balanced),
        "n_neg_after": sum(int(row["is_correct"]) == 0 for row in balanced),
    })
    return balanced, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_jsonl", required=True,
                        help="Knowledge-model answer JSONL after containment relabeling.")
    parser.add_argument("--model", required=True,
                        help="Tokenizer source.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--balance", choices=["none", "downsample", "oversample"],
                        default="none")
    parser.add_argument("--balance_seed", type=int, default=0)
    args = parser.parse_args()

    rows = _load_rows(Path(args.answers_jsonl), args.split)
    if not rows:
        raise SystemExit(f"[error] no rows for split={args.split} in {args.answers_jsonl}")
    stale = [
        row.get("qid") for row in rows
        if row.get("label_version") != ANSWER_CONTAINMENT_LABEL_VERSION
    ]
    if stale:
        raise SystemExit(
            f"[error] answers_jsonl has stale/missing containment labels; "
            f"first qid={stale[0]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    out_rows = []
    dropped = 0
    for row in rows:
        sample = build_yes_no_sample(
            tokenizer=tokenizer,
            question=row["question"],
            label=int(row["is_correct"]),
            max_seq_len=args.max_seq_len,
        )
        if sample is None:
            dropped += 1
            continue
        sample.update({
            "qid": row["qid"],
            "split": row.get("split", "train"),
            "target": "Yes" if int(row["is_correct"]) else "No",
            "is_correct": int(row["is_correct"]),
            "data_version": FOK_DATA_VERSION,
        })
        out_rows.append(sample)

    if not out_rows:
        raise SystemExit("[error] no FoK training rows survived tokenization")
    out_rows, balance_meta = _balance_rows(
        out_rows, mode=args.balance, seed=args.balance_seed)

    out_dir = Path(args.output_dir)
    jsonl_path = out_dir / "fok_tokenized.jsonl"
    dataset_dir = out_dir / "fok_tokenized_dataset"
    _write_jsonl(jsonl_path, out_rows)
    _save_dataset(dataset_dir, out_rows)
    meta = {
        "data_version": FOK_DATA_VERSION,
        "label_version": ANSWER_CONTAINMENT_LABEL_VERSION,
        "answers_jsonl": str(Path(args.answers_jsonl)),
        "split": args.split,
        "n": len(out_rows),
        "dropped": dropped,
        **balance_meta,
    }
    (out_dir / "fok_data_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] FoK data n={len(out_rows)} dropped={dropped} "
          f"balance={args.balance} "
          f"pos={balance_meta['n_pos_after']} neg={balance_meta['n_neg_after']}")
    print(f"       jsonl={jsonl_path}")
    print(f"       tokenized_path={dataset_dir}")


if __name__ == "__main__":
    main()
