from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from src.data.evaluator import compute_f1, exact_match, normalize_answer


ANSWER_CONTAINMENT_LABEL_VERSION = "answer_containment_v1"


@dataclass
class ContainmentLabel:
    is_correct: int
    em: int
    f1: float


def _as_list(answers: object) -> list[str]:
    if isinstance(answers, list):
        return [str(answer) for answer in answers]
    return [str(answers)]


def contains_answer(prediction: str, gold_answers: object) -> bool:
    norm_prediction = normalize_answer(str(prediction or ""))
    if not norm_prediction:
        return False
    padded_prediction = f" {norm_prediction} "
    for answer in _as_list(gold_answers):
        norm_answer = normalize_answer(answer)
        if not norm_answer:
            continue
        if f" {norm_answer} " in padded_prediction:
            return True
    return False


def label_answers_by_containment(
    gold_answers: Iterable[object],
    predictions: Iterable[str],
) -> List[ContainmentLabel]:
    labels: list[ContainmentLabel] = []
    for answers, prediction in zip(gold_answers, predictions):
        answer_list = _as_list(answers)
        labels.append(ContainmentLabel(
            is_correct=int(contains_answer(prediction, answer_list)),
            em=int(exact_match(str(prediction), answer_list)),
            f1=float(compute_f1(str(prediction), answer_list)),
        ))
    return labels
