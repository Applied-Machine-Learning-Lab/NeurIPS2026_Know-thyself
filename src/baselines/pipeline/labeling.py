from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.baselines.utils.answer_containment import (
    ANSWER_CONTAINMENT_LABEL_VERSION,
    label_answers_by_containment,
)
from src.baselines.utils.prompt import QA_PROMPT_VERSION


def greedy_current(rows: list[dict], questions: list[str],
                   expected_model: str | None = None) -> bool:
    if len(rows) != len(questions):
        return False
    if all("question" in row for row in rows) and \
            [row["question"] for row in rows] != questions:
        return False
    if not all("answer" in row and "is_correct" in row for row in rows):
        return False
    if not all(row.get("answer_prompt_version") == QA_PROMPT_VERSION
               for row in rows):
        return False
    if expected_model is not None and not all(
            row.get("eval_model_path") == expected_model for row in rows):
        return False
    return all(row.get("label_version") == ANSWER_CONTAINMENT_LABEL_VERSION
               for row in rows)


def load_greedy(path: Path, questions: list[str],
                expected_model: str | None = None):
    if not path.exists():
        return None
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
    except json.JSONDecodeError as exc:
        print(f"[warn] ignoring {path}: invalid JSON cache ({exc})")
        return None
    if not greedy_current(rows, questions, expected_model):
        return None
    print(f"[skip] shared greedy decode — existing {path}")
    return [row["answer"] for row in rows], \
        [int(bool(row["is_correct"])) for row in rows]


def label_answers_by_gold_containment(args, questions: List[str], gold: List,
                                      answers: List[str]):
    print("[label] correctness by normalized answer containment")
    labeled = label_answers_by_containment(gold, answers)
    labels = [row.is_correct for row in labeled]
    return labels, labeled


def write_greedy(path: Path, samples_meta: list, questions: List[str],
                 answers: List[str], labels: List[int], labeled,
                 extra_fields: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for s, q, a, lbl, label in zip(samples_meta, questions, answers, labels,
                                       labeled):
            f.write(json.dumps(dict(
                qid=s.get("id"),
                question=q,
                answers=s.get("answers"),
                split=s.get("split"),
                answer=a,
                is_correct=bool(lbl),
                em=int(label.em),
                f1=float(label.f1),
                answer_prompt_version=QA_PROMPT_VERSION,
                label_version=ANSWER_CONTAINMENT_LABEL_VERSION,
                **(extra_fields or {}),
            ), ensure_ascii=False) + "\n")
    tmp.replace(path)
