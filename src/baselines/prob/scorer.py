import math
from typing import List, Optional

from src.baselines.base import VLLMBaselineScorer
from src.baselines.utils.pre_answer import (PRE_ANSWER_SCORE_VERSION,
                                            yes_no_prompts)


class _SeqProbBase(VLLMBaselineScorer):
    normalize: bool = False
    score_version = PRE_ANSWER_SCORE_VERSION

    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        del greedy_answers
        prompts = yes_no_prompts(self.runner.tokenizer, questions)
        yes_lps = self.runner.teacher_force_logprobs(prompts, [" Yes"] * len(prompts))
        no_lps = self.runner.teacher_force_logprobs(prompts, [" No"] * len(prompts))
        scores = []
        for pos_lps, neg_lps in zip(yes_lps, no_lps):
            if not pos_lps or not neg_lps:
                scores.append(0.5)
                continue
            pos = sum(pos_lps)
            neg = sum(neg_lps)
            if self.normalize:
                pos /= len(pos_lps)
                neg /= len(neg_lps)
            m = max(pos, neg)
            p_yes = math.exp(pos - m)
            p_no = math.exp(neg - m)
            scores.append(p_yes / max(p_yes + p_no, 1e-12))
        return scores


class ProbScorer(_SeqProbBase):
    name = "prob"
    normalize = False


class NProbScorer(_SeqProbBase):
    name = "n_prob"
    normalize = True
