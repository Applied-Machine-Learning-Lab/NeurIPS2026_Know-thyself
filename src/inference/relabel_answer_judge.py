from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines.utils.answer_judge import (
    ANSWER_JUDGE_LABEL_VERSION,
    judge_answers,
)
from src.baselines.utils.judge_runner import SingleVLLMJudge
from src.baselines.utils.prompt import QA_PROMPT_VERSION


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _current(rows: list[dict], tag: str) -> bool:
    ic_field = f"is_correct_{tag}" if tag else "is_correct"
    return bool(rows) and all(
        row.get("answer_prompt_version") == QA_PROMPT_VERSION and
        row.get(f"label_version_{tag}" if tag else "label_version")
            == ANSWER_JUDGE_LABEL_VERSION and
        ic_field in row
        for row in rows
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _acc_line(name: str, rows: list[dict], ic_field: str) -> str:
    if not rows:
        return f"{name}=nan (0/0)"
    correct = sum(int(row[ic_field]) for row in rows)
    return f"{name}={correct / len(rows):.3f} ({correct}/{len(rows)})"


def _acc_summary(rows: list[dict], ic_field: str) -> str:
    parts = [_acc_line("all", rows, ic_field)]
    if any("split" in row for row in rows):
        for split in ("train", "val"):
            split_rows = [row for row in rows if row.get("split") == split]
            if split_rows:
                parts.append(_acc_line(split, split_rows, ic_field))
    return "; ".join(parts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--judge_model", required=True)
    p.add_argument("--judge_tag", default="",
                   help="If set, store results under suffixed keys: "
                        "is_correct_<tag>, judge_yes_prob_<tag>, etc. "
                        "Empty string (default) uses unsuffixed keys and "
                        "overwrites is_correct in-place.")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.input)
    rows = _read_jsonl(path)
    tag = args.judge_tag.strip()
    suffix = f"_{tag}" if tag else ""
    ic_field = f"is_correct{suffix}"
    yes_field = f"judge_yes_prob{suffix}"
    no_field = f"judge_no_prob{suffix}"
    lv_field = f"label_version{suffix}"

    if _current(rows, tag):
        print(f"[skip] {path} already has {ANSWER_JUDGE_LABEL_VERSION} "
              f"(tag={tag!r})")
        return
    stale_prompt = [
        row.get("qid") for row in rows
        if row.get("answer_prompt_version") != QA_PROMPT_VERSION
    ]
    if stale_prompt:
        raise SystemExit(
            f"[error] {path} uses a stale answer prompt; rerun inference first. "
            f"first stale qid={stale_prompt[0]}")

    judge = SingleVLLMJudge(
        model_path=args.judge_model,
        tp=args.tp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        judged = judge_answers(
            judge,
            [row["question"] for row in rows],
            [row["answers"] for row in rows],
            [row["answer"] for row in rows],
        )
    finally:
        judge.shutdown()

    for row, label in zip(rows, judged):
        row[ic_field] = int(label.is_correct)
        row[yes_field] = label.yes_prob
        row[no_field] = label.no_prob
        row[lv_field] = ANSWER_JUDGE_LABEL_VERSION

    _write_jsonl(path, rows)
    print(f"[done] relabeled {path} with {ANSWER_JUDGE_LABEL_VERSION} "
          f"(tag={tag!r}); {_acc_summary(rows, ic_field)}")


if __name__ == "__main__":
    main()
