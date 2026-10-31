from typing import List, Optional

from src.baselines.base import VLLMBaselineScorer
from src.baselines.utils.pre_answer import (PRE_ANSWER_SCORE_VERSION,
                                            numeric_prompts,
                                            parse_numeric_confidence)


def _chat_wrap(tokenizer, body: str) -> str:
    msgs = [{"role": "user", "content": body}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


_VERBAL10_FEWSHOTS = [
    ("What is the capital of France?", 98),
    ("Who wrote the novel '1984'?", 95),
    ("What is the chemical symbol for gold?", 97),
    ("In what year did the Titanic sink?", 92),
    ("Who painted the Mona Lisa?", 96),
    ("What is the smallest prime number?", 99),
    ("Which planet is known as the Red Planet?", 98),
    ("What is the population of Reykjavik?", 45),
    ("Who invented the adhesive bandage?", 30),
    ("What is the third-longest river in Africa?", 35),
]

_VERBAL10_PREFIX = "\n\n".join(
    f"Question: {q}\nConfidence: {c}"
    for q, c in _VERBAL10_FEWSHOTS
)

VERBAL10_BODY = (
    _VERBAL10_PREFIX
    + "\n\nQuestion: {question}\n"
    "Before answering, give a confidence score between 0 and 100 for whether "
    "you can answer this question correctly. Output only the number."
)


_VERBAL_MIN_TOKENS = 16


class VerbalizedScorer(VLLMBaselineScorer):
    name = "verbalized"
    score_version = PRE_ANSWER_SCORE_VERSION

    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        del greedy_answers
        tok = self.runner.tokenizer
        prompts = numeric_prompts(tok, questions)
        mnt = max(self.max_new_tokens, _VERBAL_MIN_TOKENS)
        texts = self.runner.generate_greedy(prompts, max_new_tokens=mnt)
        return [parse_numeric_confidence(t) for t in texts]


class Verbalized10Scorer(VLLMBaselineScorer):
    name = "verbal_10"
    score_version = PRE_ANSWER_SCORE_VERSION

    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        del greedy_answers
        tok = self.runner.tokenizer
        prompts = [_chat_wrap(tok, VERBAL10_BODY.format(question=q))
                   for q in questions]
        mnt = max(self.max_new_tokens, _VERBAL_MIN_TOKENS)
        texts = self.runner.generate_greedy(prompts, max_new_tokens=mnt)
        return [parse_numeric_confidence(t.split("\n\n")[0]) for t in texts]
