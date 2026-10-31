from abc import ABC, abstractmethod
from typing import List, Optional


class BaselineScorer(ABC):
    name: str = "unnamed"

    @abstractmethod
    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        raise NotImplementedError

    def __call__(self, questions, **kw):
        return self.score(questions, **kw)


class VLLMBaselineScorer(BaselineScorer):

    def __init__(self, runner, max_new_tokens: int = 128):
        self.runner = runner
        self.max_new_tokens = max_new_tokens


class HFDDPBaselineScorer(BaselineScorer):

    def __init__(self, model_path: str, nprocs: int = 8,
                 dtype: str = "bfloat16", max_new_tokens: int = 128,
                 batch_size: int = 8):
        self.model_path = model_path
        self.nprocs = nprocs
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
