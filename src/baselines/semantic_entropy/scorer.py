import math
from typing import List, Optional

from src.baselines.base import VLLMBaselineScorer
from src.baselines.utils.pre_answer import (PRE_ANSWER_SCORE_VERSION,
                                            parse_yes_probability,
                                            yes_no_prompts)


class SemanticEntropyScorer(VLLMBaselineScorer):
    name = "semantic_entropy"

    NLI_ENTAIL = 2

    def __init__(self, runner, num_samples: int = 10, temperature: float = 1.0,
                 variant: str = "discrete",
                 nli_model_name: str = "microsoft/deberta-large-mnli",
                 nli_device: str = "cuda:0",
                 max_new_tokens: int = 128,
                 nli_nprocs: int = 1):
        super().__init__(runner, max_new_tokens=max_new_tokens)
        self.num_samples = num_samples
        self.temperature = temperature
        assert variant in {"discrete", "se"}
        self.variant = variant
        self.nli_model_name = nli_model_name
        self.nli_device = nli_device
        self.nli_nprocs = nli_nprocs
        self._nli_loaded = False
        self.score_version = PRE_ANSWER_SCORE_VERSION

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

    def _cluster(self, texts: List[str]) -> List[int]:
        cid = [-1] * len(texts)
        reps: List[int] = []
        for i, s in enumerate(texts):
            placed = False
            for c, rep in enumerate(reps):
                if s.strip() == texts[rep].strip() or \
                   self._bidir_entail(s, texts[rep]):
                    cid[i] = c
                    placed = True
                    break
            if not placed:
                cid[i] = len(reps)
                reps.append(i)
        return cid

    def _entropy_discrete(self, cids: List[int]) -> float:
        from collections import Counter
        c = Counter(cids)
        k = sum(c.values())
        h = 0.0
        for v in c.values():
            p = v / k
            if p > 0:
                h -= p * math.log(p)
        return h

    def _entropy_se(self, cids: List[int], lps: List[float]) -> float:
        from collections import defaultdict
        cl = defaultdict(list)
        for c, lp in zip(cids, lps):
            cl[c].append(lp)
        cluster_lp = []
        for members in cl.values():
            m = max(members)
            cluster_lp.append(m + math.log(sum(math.exp(x - m) for x in members)))
        m = max(cluster_lp)
        exps = [math.exp(x - m) for x in cluster_lp]
        z = sum(exps)
        probs = [e / z for e in exps]
        return -sum(p * math.log(p) for p in probs if p > 0)

    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        del greedy_answers
        prompts = yes_no_prompts(self.runner.tokenizer, questions)
        samples = self.runner.generate_samples(
            prompts, num_samples=self.num_samples,
            temperature=self.temperature,
            max_new_tokens=min(self.max_new_tokens, 16),
            strip_newline=True)
        out = []
        for sam_list in samples:
            if not sam_list:
                out.append(0.5)
                continue
            vals = [parse_yes_probability(s.text) for s in sam_list]
            out.append(sum(vals) / len(vals))
        return out
