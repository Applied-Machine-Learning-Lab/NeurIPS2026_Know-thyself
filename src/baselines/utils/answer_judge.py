from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


ANSWER_JUDGE_LABEL_VERSION = "llm_judge_yes_no_v2"
YES_CHOICES = ["yes", "Yes", " yes", " Yes"]
NO_CHOICES = ["no", "No", " no", " No"]


JUDGE_TEMPLATE = (
    "You are judging whether a generated answer is correct.\n"
    "Question: {question}\n"
    "Gold answers: {gold_answers}\n"
    "Generated answer: {prediction}\n\n"
    "Treat the gold answers as authoritative. Do not use outside knowledge "
    "to reject an exact or equivalent gold answer. Does the generated answer "
    "mean the same thing as any gold answer? Answer with exactly yes or no."
)


@dataclass
class JudgeLabel:
    is_correct: int
    yes_prob: float
    no_prob: float


def build_answer_judge_prompts(tokenizer, questions: Iterable[str],
                               gold_answers: Iterable[object],
                               predictions: Iterable[str]) -> List[str]:
    prompts = []
    for question, answers, prediction in zip(
            questions, gold_answers, predictions):
        if isinstance(answers, list):
            gold = " / ".join(str(a) for a in answers)
        else:
            gold = str(answers)
        body = JUDGE_TEMPLATE.format(
            question=question,
            gold_answers=gold,
            prediction=prediction,
        )
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True))
    return prompts


def judge_answers(judge_runner, questions: List[str],
                  gold_answers: List[object],
                  predictions: List[str]) -> List[JudgeLabel]:
    from src.data.evaluator import exact_match

    labels: List[JudgeLabel | None] = [None] * len(questions)
    pending_indices = []
    pending_questions = []
    pending_answers = []
    pending_predictions = []
    for idx, (question, answers, prediction) in enumerate(zip(
            questions, gold_answers, predictions)):
        answer_list = answers if isinstance(answers, list) else [answers]
        if exact_match(str(prediction), [str(answer) for answer in answer_list]):
            labels[idx] = JudgeLabel(is_correct=1, yes_prob=1.0, no_prob=0.0)
            continue
        pending_indices.append(idx)
        pending_questions.append(question)
        pending_answers.append(answers)
        pending_predictions.append(prediction)

    if pending_indices:
        prompts = build_answer_judge_prompts(
            judge_runner.tokenizer,
            pending_questions,
            pending_answers,
            pending_predictions)
        probs = judge_runner.first_token_group_softmax(
            prompts, [YES_CHOICES, NO_CHOICES])
        for idx, (yes_prob, no_prob) in zip(pending_indices, probs):
            labels[idx] = JudgeLabel(
                is_correct=int(yes_prob >= no_prob),
                yes_prob=float(yes_prob),
                no_prob=float(no_prob),
            )
    assert all(label is not None for label in labels)
    return [label for label in labels if label is not None]
