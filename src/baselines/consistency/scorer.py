from typing import List, Optional

from src.baselines.base import VLLMBaselineScorer
from src.baselines.utils.pre_answer import (PRE_ANSWER_SCORE_VERSION,
                                            parse_yes_probability,
                                            yes_no_prompts)


def _rouge_l_f1(hyp: str, ref: str) -> float:
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if h[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    rr = lcs / n
    return 2 * p * rr / (p + rr)


class _ConsisBase(VLLMBaselineScorer):
    def __init__(self, runner, num_samples: int = 10, temperature: float = 1.0,
                 max_new_tokens: int = 128):
        super().__init__(runner, max_new_tokens=max_new_tokens)
        self.num_samples = num_samples
        self.temperature = temperature
        self.score_version = PRE_ANSWER_SCORE_VERSION

    def _sample_answerability(self, questions: List[str]):
        prompts = yes_no_prompts(self.runner.tokenizer, questions)
        return self.runner.generate_samples(
            prompts, num_samples=self.num_samples,
            temperature=self.temperature,
            max_new_tokens=min(self.max_new_tokens, 16),
            strip_newline=True)

    def _yes_fraction(self, questions: List[str]) -> List[float]:
        samples = self._sample_answerability(questions)
        out = []
        for sam_list in samples:
            if not sam_list:
                out.append(0.5)
                continue
            vals = [parse_yes_probability(s.text) for s in sam_list]
            out.append(sum(vals) / len(vals))
        return out


class ConsisLexScorer(_ConsisBase):
    name = "consis_lex"

    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        del greedy_answers
        return self._yes_fraction(questions)


class ConsisSemScorer(_ConsisBase):
    name = "consis_sem"

    NLI_ENTAIL = 2

    def __init__(self, runner, nli_model_name: str = "microsoft/deberta-large-mnli",
                 nli_device: str = "cuda:0", nli_nprocs: int = 1, **kw):
        super().__init__(runner, **kw)
        self.nli_model_name = nli_model_name
        self.nli_device = nli_device
        self.nli_nprocs = nli_nprocs
        self._nli_loaded = False

    def _load_nli(self):
        if self._nli_loaded:
            return
        import os
        from pathlib import Path
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        local_only = Path(self.nli_model_name).exists() or \
            os.environ.get("HF_HUB_OFFLINE") == "1" or \
            os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        self.nli_tok = AutoTokenizer.from_pretrained(
            self.nli_model_name, local_files_only=local_only)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_name,
            local_files_only=local_only).to(self.nli_device).eval()
        self._nli_loaded = True

    def _bidir_entail(self, a: str, b: str) -> bool:
        import torch
        with torch.no_grad():
            pairs = [(a, b), (b, a)]
            enc = self.nli_tok(pairs, padding=True, truncation=True,
                               max_length=256, return_tensors="pt"
                               ).to(self.nli_device)
            preds = self.nli_model(**enc).logits.argmax(dim=-1).tolist()
        return preds[0] == self.NLI_ENTAIL and preds[1] == self.NLI_ENTAIL

    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        del greedy_answers
        return self._yes_fraction(questions)
