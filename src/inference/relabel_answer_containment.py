from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines.utils.answer_containment import (
    ANSWER_CONTAINMENT_LABEL_VERSION,
    label_answers_by_containment,
)
from src.baselines.pipeline.common import compute_auroc, compute_threshold_metrics
from src.baselines.utils.prompt import QA_PROMPT_VERSION


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"[error] invalid JSONL in {path}:{lineno}: {exc.msg}"
                ) from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _meta_by_qid(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    out = {}
    for row in _read_jsonl(path):
        qid = str(row.get("id", row.get("qid")))
        out[qid] = row
    return out


def _prediction(row: dict) -> str:
    return str(row.get("answer", row.get("response", "")))


def _answers(row: dict, meta: dict[str, dict]) -> object:
    if row.get("answers") is not None:
        return row["answers"]
    m = meta.get(str(row.get("qid")), {})
    if m.get("answers") is not None:
        return m["answers"]
    raise SystemExit(f"[error] row qid={row.get('qid')} has no answers")


def _fill_meta(row: dict, meta: dict[str, dict]) -> None:
    m = meta.get(str(row.get("qid")), {})
    if row.get("question") is None and m.get("question") is not None:
        row["question"] = m["question"]
    if row.get("answers") is None and m.get("answers") is not None:
        row["answers"] = m["answers"]
    if row.get("split") is None and m.get("split") is not None:
        row["split"] = m["split"]


def _current(rows: list[dict]) -> bool:
    return bool(rows) and all(
        row.get("answer_prompt_version") == QA_PROMPT_VERSION and
        row.get("label_version") == ANSWER_CONTAINMENT_LABEL_VERSION
        for row in rows
    )


def _acc_line(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"{name}=nan (0/0)"
    correct = sum(int(row["is_correct"]) for row in rows)
    return f"{name}={correct / len(rows):.3f} ({correct}/{len(rows)})"


def _acc_summary(rows: list[dict]) -> str:
    parts = [_acc_line("all", rows)]
    if any("split" in row for row in rows):
        for split in ("train", "val"):
            split_rows = [row for row in rows if row.get("split") == split]
            if split_rows:
                parts.append(_acc_line(split, split_rows))
    return "; ".join(parts)


def _maybe_update_metrics(path: Path, rows: list[dict]) -> None:
    if not path.name.endswith("_scores.jsonl"):
        return
    if not rows or not all("score" in row for row in rows):
        return
    labels = [int(row["is_correct"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    auroc = compute_auroc(scores, labels)
    tm = compute_threshold_metrics(scores, labels)
    metrics_path = path.with_name(path.name.replace("_scores.jsonl",
                                                     "_metrics.json"))
    extra = {}
    for key, value in rows[0].items():
        if key.endswith("_path") or key.endswith("_mode") or key in {
            "eval_model_path",
            "answer_prompt_version",
            "label_version",
        }:
            extra[key] = value
    metrics_path.write_text(json.dumps(dict(
        auroc=auroc,
        **tm,
        n=len(labels),
        n_pos=int(sum(labels)),
        n_neg=int(len(labels) - sum(labels)),
        **extra,
    ), ensure_ascii=False, indent=2) + "\n")
    print(f"[done] updated metrics {metrics_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--queries_jsonl", default=None,
                   help="Optional qid -> question/answers/split source for "
                        "score files that do not carry answers.")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.input)
    rows = _read_jsonl(path)
    if _current(rows):
        print(f"[skip] {path} already has {ANSWER_CONTAINMENT_LABEL_VERSION}")
        return
    meta = _meta_by_qid(Path(args.queries_jsonl) if args.queries_jsonl else None)

    stale_prompt = [
        row.get("qid") for row in rows
        if row.get("answer_prompt_version") not in (None, QA_PROMPT_VERSION)
    ]
    if stale_prompt:
        raise SystemExit(
            f"[error] {path} uses a stale answer prompt; rerun inference first. "
            f"first stale qid={stale_prompt[0]}")

    answers = []
    predictions = []
    for row in rows:
        _fill_meta(row, meta)
        answers.append(_answers(row, meta))
        predictions.append(_prediction(row))

    labels = label_answers_by_containment(answers, predictions)
    for row, label in zip(rows, labels):
        row["is_correct"] = int(label.is_correct)
        row["em"] = int(label.em)
        row["f1"] = float(label.f1)
        row["answer_prompt_version"] = QA_PROMPT_VERSION
        row["label_version"] = ANSWER_CONTAINMENT_LABEL_VERSION
        row.pop("judge_yes_prob", None)
        row.pop("judge_no_prob", None)

    _write_jsonl(path, rows)
    _maybe_update_metrics(path, rows)
    print(f"[done] relabeled {path} with "
          f"{ANSWER_CONTAINMENT_LABEL_VERSION}; {_acc_summary(rows)}")


if __name__ == "__main__":
    main()
