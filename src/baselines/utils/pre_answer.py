from __future__ import annotations

import re
from typing import List


PRE_ANSWER_SCORE_VERSION = "pre_answer_question_only_v1"


AB_BODY = (
    "Question: {question}\n"
    "Before answering, judge whether you can answer this question correctly.\n"
    "Choose one option:\n"
    " (A) Yes, I can answer it correctly.\n"
    " (B) No, I cannot answer it correctly.\n"
    "The correct option is: ("
)

YES_NO_BODY = (
    "Question: {question}\n"
    "Before answering, judge whether you can answer this question correctly.\n"
    "Answer only Yes or No.\n"
    "Answer:"
)

NUMERIC_BODY = (
    "Question: {question}\n"
    "Before answering, give a confidence score between 0 and 100 for whether "
    "you can answer this question correctly. Output only the number."
)


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_YES_RE = re.compile(r"\b(?:yes|can|correctly|confident)\b", re.I)
_NO_RE = re.compile(r"\b(?:no|cannot|can't|unable|not sure|uncertain)\b", re.I)


def chat_wrap(tokenizer, body: str) -> str:
    msgs = [{"role": "user", "content": body}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def ab_prompts(tokenizer, questions: List[str]) -> List[str]:
    return [chat_wrap(tokenizer, AB_BODY.format(question=q))
            for q in questions]


def yes_no_prompts(tokenizer, questions: List[str]) -> List[str]:
    return [chat_wrap(tokenizer, YES_NO_BODY.format(question=q))
            for q in questions]


def numeric_prompts(tokenizer, questions: List[str]) -> List[str]:
    return [chat_wrap(tokenizer, NUMERIC_BODY.format(question=q))
            for q in questions]


def parse_numeric_confidence(text: str) -> float:
    match = _NUMBER_RE.search(text)
    if not match:
        return 0.5
    try:
        value = float(match.group(0))
    except ValueError:
        return 0.5
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def parse_yes_probability(text: str) -> float:
    stripped = text.strip().lower()
    if not stripped:
        return 0.5
    if stripped.startswith(("yes", "a", "(a")):
        return 1.0
    if stripped.startswith(("no", "b", "(b")):
        return 0.0
    yes = bool(_YES_RE.search(stripped))
    no = bool(_NO_RE.search(stripped))
    if yes and not no:
        return 1.0
    if no and not yes:
        return 0.0
    return 0.5
