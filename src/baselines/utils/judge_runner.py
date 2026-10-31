from __future__ import annotations

import gc
import math
import os
from typing import List


class SingleVLLMJudge:
    def __init__(self, model_path: str, tp: int = 8,
                 dtype: str = "bfloat16", max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.90):
        from transformers import AutoTokenizer
        from vllm import LLM
        cache_root = os.environ.get("VLLM_CACHE_ROOT")
        if not cache_root:
            cache_root = os.path.abspath(os.path.join(
                "tmp", "vllm_cache", f"judge_{os.getpid()}"))
        os.makedirs(cache_root, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = cache_root
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
            cache_root, "torchinductor")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")

        self.model_path = model_path
        self.tp = tp
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        print(f"[judge_runner] loaded {model_path} (tp={tp})")

    def generate_greedy(self, prompts: List[str],
                         max_new_tokens: int = 16,
                         top_p: float = 0.95, top_k: int = 50,
                         repetition_penalty: float = 1.05) -> List[str]:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=0.0, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
        )
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    def first_token_group_softmax(self, prompts: List[str],
                                  choice_groups: List[List[str]]
                                  ) -> List[List[float]]:
        from vllm import SamplingParams
        group_token_ids = []
        for group in choice_groups:
            ids = set()
            for choice in group:
                toks = self.tokenizer(choice, add_special_tokens=False).input_ids
                if toks:
                    ids.add(toks[0])
            if not ids:
                raise ValueError(f"choice group tokenizes to empty: {group}")
            group_token_ids.append(sorted(ids))

        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outs = self.llm.generate(prompts, sp)
        rows = []
        for out in outs:
            lp = out.outputs[0].logprobs
            if not lp or lp[0] is None:
                rows.append([1.0 / len(choice_groups)] * len(choice_groups))
                continue
            token_logprobs = {
                tid: item.logprob for tid, item in lp[0].items()
            }
            raw = []
            for ids in group_token_ids:
                vals = [token_logprobs.get(tid, -1e9) for tid in ids]
                m = max(vals)
                raw.append(m + math.log(sum(math.exp(v - m) for v in vals)))
            m = max(raw)
            exps = [math.exp(v - m) for v in raw]
            z = sum(exps)
            rows.append([v / z for v in exps])
        return rows

    def first_token_softmax(self, prompts: List[str],
                            choices: List[str]) -> List[List[float]]:
        return self.first_token_group_softmax(
            prompts, [[choice] for choice in choices])

    def shutdown(self):
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
