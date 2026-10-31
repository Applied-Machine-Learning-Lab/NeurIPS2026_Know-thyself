from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.baselines.utils.answer_containment import (
    ANSWER_CONTAINMENT_LABEL_VERSION,
)
from src.baselines.utils.pre_answer import PRE_ANSWER_SCORE_VERSION
from src.baselines.utils.prompt import QA_PROMPT_VERSION


VLLM_METHODS = {
    "verbalized", "verbal_10",
    "prob", "n_prob",
    "consis_lex", "consis_sem", "semantic_entropy",
}
HF_DP_METHODS = {"elical"}
ALL_METHODS = sorted(VLLM_METHODS | HF_DP_METHODS)

METHOD_SCORE_VERSIONS = {
    "verbalized": PRE_ANSWER_SCORE_VERSION,
    "verbal_10": PRE_ANSWER_SCORE_VERSION,
    "prob": PRE_ANSWER_SCORE_VERSION,
    "n_prob": PRE_ANSWER_SCORE_VERSION,
    "consis_lex": PRE_ANSWER_SCORE_VERSION,
    "consis_sem": PRE_ANSWER_SCORE_VERSION,
    "semantic_entropy": PRE_ANSWER_SCORE_VERSION,
    "elical": "elical_official_pre_answer_vector_head_v1",
}


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    return float(np.trapezoid(tp / n_pos, fp / n_neg))


def compute_threshold_metrics(scores: List[float], labels: List[int],
                                threshold: float = 0.5) -> Dict[str, float]:
    s = np.asarray(scores); y = np.asarray(labels)
    pred = (s > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return dict(
        accuracy=(tp + tn) / max(len(y), 1),
        tpr=tp / max(tp + fn, 1),
        tnr=tn / max(tn + fp, 1),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def compute_ece(scores: List[float], labels: List[int],
                 n_bins: int = 15) -> float:
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=int)
    if len(s) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s < hi) if hi < 1.0 else (s >= lo) & (s <= hi)
        if not mask.any():
            continue
        conf = s[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / len(s)) * abs(conf - acc)
    return float(ece)


def best_threshold_youden(scores: List[float], labels: List[int]) -> float:
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
    if len(set(y.tolist())) < 2:
        return 0.5
    order = np.argsort(-s)
    s_sorted = s[order]; y_sorted = y[order]
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos; tnr = 1.0 - fp / n_neg
    j = tpr + tnr - 1.0
    idx = int(np.argmax(j))
    return float(s_sorted[idx])


def _safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return compute_auroc(scores.tolist(), labels.tolist()) \
        if len(labels) and len(set(labels.tolist())) >= 2 else float("nan")


def save_result(output_dir: Path, name: str, samples_meta: list,
                greedy: list, scores: list, labels: list,
                extra_fields: dict | None = None) -> Dict[str, float]:
    auroc = compute_auroc(scores, labels)
    tm = compute_threshold_metrics(scores, labels)
    print(f"[{name}] AUROC={auroc:.4f} Acc={tm['accuracy']:.3f} "
          f"TPR={tm['tpr']:.3f} TNR={tm['tnr']:.3f} "
          f"(n={len(labels)}, pos={sum(labels)}, neg={len(labels)-sum(labels)})")
    per_sample = [
        dict(qid=s.get("id", i), score=float(sc), answer=a,
             is_correct=bool(lbl),
             answer_prompt_version=QA_PROMPT_VERSION,
             label_version=ANSWER_CONTAINMENT_LABEL_VERSION,
             answers=s.get("answers"),
             split=s.get("split"),
             **(extra_fields or {}))
        for i, (s, sc, a, lbl) in enumerate(
            zip(samples_meta, scores, greedy, labels))
    ]
    with open(output_dir / f"{name}_scores.jsonl", "w") as f:
        for r in per_sample:
            f.write(json.dumps(r) + "\n")
    with open(output_dir / f"{name}_metrics.json", "w") as f:
        json.dump(dict(auroc=auroc, **tm, n=len(labels),
                       n_pos=int(sum(labels)),
                       n_neg=int(len(labels) - sum(labels)),
                       **(extra_fields or {})), f, indent=2)
    return dict(auroc=auroc, accuracy=tm["accuracy"])
