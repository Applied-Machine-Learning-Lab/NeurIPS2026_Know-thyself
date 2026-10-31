from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch.multiprocessing as mp


@dataclass
class SampleRecord:
    text: str
    log_prob: float
    finish_reason: str | None = None
    stop_reason: str | None = None
    token_ids: List[int] | None = None


def _visible_gpu_ids(nprocs: int) -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if len(ids) < nprocs:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES has {len(ids)} GPUs but nprocs={nprocs}: "
                f"{visible}")
        return ids[:nprocs]
    return [str(i) for i in range(nprocs)]


def _generate_with_optional_lora(llm, prompts, sampling_params, lora_request):
    if lora_request is None:
        return llm.generate(prompts, sampling_params)
    return llm.generate(prompts, sampling_params, lora_request=lora_request)


def _infer_max_lora_rank(lora_path: str | None) -> int | None:
    if not lora_path:
        return None
    cfg_path = Path(lora_path) / "adapter_config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
        rank = int(cfg.get("r", 16))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        rank = 16
    return max(16, rank)


def _worker_loop(rank: int, gpu_id: str, model_path: str,
                 lora_path: str | None, dtype: str,
                 max_model_len: int, gpu_memory_utilization: float,
                 max_lora_rank: int | None,
                 cache_root: str,
                 req_q: mp.Queue, resp_q: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if cache_root:
        rank_cache = os.path.join(cache_root, f"worker_{rank}")
        os.makedirs(rank_cache, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = rank_cache
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            os.path.join(rank_cache, "torchinductor"),
        )
    try:
        from vllm import LLM, SamplingParams
        lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("eval_adapter", 1, lora_path)
        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=1,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        if lora_path:
            llm_kwargs.update(enable_lora=True, max_loras=1)
            if max_lora_rank is not None:
                llm_kwargs["max_lora_rank"] = max_lora_rank
        llm = LLM(**llm_kwargs)
        tok = llm.get_tokenizer()
    except Exception as e:
        resp_q.put(("error", rank, repr(e)))
        return
    resp_q.put(("ready", rank, None))

    while True:
        msg = req_q.get()
        if msg is None:
            break
        op, prompts, extra = msg
        try:
            if op == "generate_greedy":
                sp = SamplingParams(
                    temperature=0.0,
                    top_p=extra.get("top_p", 0.95),
                    top_k=extra.get("top_k", 50),
                    repetition_penalty=extra.get("repetition_penalty", 1.05),
                    max_tokens=extra["max_new_tokens"],
                )
                outs = _generate_with_optional_lora(
                    llm, prompts, sp, lora_request)
                if extra.get("strip_newline", False):
                    res = [o.outputs[0].text.split("\n")[0].strip()
                           for o in outs]
                else:
                    res = [o.outputs[0].text for o in outs]

            elif op == "generate_samples":
                sp = SamplingParams(
                    temperature=extra["temperature"],
                    top_p=extra.get("top_p", 0.95),
                    top_k=extra.get("top_k", 50),
                    repetition_penalty=extra.get("repetition_penalty", 1.05),
                    max_tokens=extra["max_new_tokens"],
                    n=extra["num_samples"],
                    logprobs=1,
                )
                outs = _generate_with_optional_lora(
                    llm, prompts, sp, lora_request)
                res = []
                for o in outs:
                    per = []
                    for s in o.outputs:
                        text = s.text if not extra.get(
                            "strip_newline", False) else \
                            s.text.split("\n")[0].strip()
                        if s.logprobs:
                            total = 0.0
                            n = 0
                            for step, tid in zip(s.logprobs, s.token_ids):
                                if step is None or tid not in step:
                                    continue
                                total += step[tid].logprob
                                n += 1
                            lp = total / max(n, 1)
                        else:
                            lp = 0.0
                        per.append((
                            text,
                            lp,
                            getattr(s, "finish_reason", None),
                            getattr(s, "stop_reason", None),
                            list(getattr(s, "token_ids", []) or []),
                        ))
                    res.append(per)

            elif op == "teacher_force_logprobs":
                completions = extra["completions"]
                full_texts = [f"{p}{c}" for p, c in zip(prompts, completions)]
                prompt_lens = [
                    len(tok(p, add_special_tokens=False).input_ids)
                    for p in prompts
                ]
                sp = SamplingParams(temperature=0.0, max_tokens=1,
                                     prompt_logprobs=1)
                outs = _generate_with_optional_lora(
                    llm, full_texts, sp, lora_request)
                res = []
                for i, o in enumerate(outs):
                    pl = prompt_lens[i]
                    pl_lp = o.prompt_logprobs
                    if pl_lp is None:
                        res.append([])
                        continue
                    full_ids = tok(full_texts[i],
                                    add_special_tokens=False).input_ids
                    comp_lps: List[float] = []
                    for t in range(pl, len(full_ids)):
                        if t >= len(pl_lp) or pl_lp[t] is None:
                            continue
                        tid = full_ids[t]
                        if tid in pl_lp[t]:
                            comp_lps.append(pl_lp[t][tid].logprob)
                    res.append(comp_lps)

            elif op == "first_token_softmax":
                choices = extra["choices"]
                choice_ids = []
                for c in choices:
                    ids = tok(c, add_special_tokens=False).input_ids
                    if not ids:
                        raise ValueError(f"choice {c!r} tokenizes to empty")
                    choice_ids.append(ids[0])
                k = min(20, getattr(tok, "vocab_size", 200000))
                sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=k)
                outs = _generate_with_optional_lora(
                    llm, prompts, sp, lora_request)
                res = []
                for o in outs:
                    lp = o.outputs[0].logprobs
                    if not lp or lp[0] is None:
                        res.append([1.0 / len(choices)] * len(choices))
                        continue
                    d: Dict[int, float] = {
                        tid: lg.logprob for tid, lg in lp[0].items()
                    }
                    raw = [d.get(ci, -1e9) for ci in choice_ids]
                    m = max(raw)
                    exps = [math.exp(x - m) for x in raw]
                    z = sum(exps)
                    res.append([e / z for e in exps])

            elif op == "first_token_group_softmax":
                choice_groups = extra["choice_groups"]
                group_ids = []
                for group in choice_groups:
                    ids = []
                    for c in group:
                        toks = tok(c, add_special_tokens=False).input_ids
                        if toks:
                            ids.append(toks[0])
                    if not ids:
                        raise ValueError(f"choice group {group!r} tokenizes empty")
                    group_ids.append(sorted(set(ids)))
                k = min(20, getattr(tok, "vocab_size", 200000))
                sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=k)
                outs = _generate_with_optional_lora(
                    llm, prompts, sp, lora_request)
                res = []
                for o in outs:
                    lp = o.outputs[0].logprobs
                    if not lp or lp[0] is None:
                        res.append([1.0 / len(choice_groups)] * len(choice_groups))
                        continue
                    d: Dict[int, float] = {
                        tid: lg.logprob for tid, lg in lp[0].items()
                    }
                    raw = []
                    for ids in group_ids:
                        vals = [d.get(tid, -1e9) for tid in ids]
                        m_group = max(vals)
                        raw.append(m_group + math.log(sum(
                            math.exp(v - m_group) for v in vals)))
                    m = max(raw)
                    exps = [math.exp(x - m) for x in raw]
                    z = sum(exps)
                    res.append([e / z for e in exps])

            else:
                raise ValueError(f"unknown op {op!r}")

            resp_q.put(("ok", rank, res))
        except Exception as e:
            import traceback
            resp_q.put(("error", rank,
                        f"{e!r}\n{traceback.format_exc()}"))


class VLLMRunner:

    def __init__(self, model_path: str, nprocs: int = 8,
                 lora_path: str | None = None,
                 dtype: str = "bfloat16", max_model_len: int = 2048,
                 gpu_memory_utilization: float = 0.90):
        from transformers import AutoTokenizer
        self.model_path = model_path
        self.nprocs = nprocs
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        gpu_ids = _visible_gpu_ids(nprocs)
        cache_root = os.environ.get("VLLM_CACHE_ROOT")
        if not cache_root:
            cache_root = os.path.abspath(os.path.join(
                "tmp", "vllm_cache", f"runner_{os.getpid()}"))
        os.makedirs(cache_root, exist_ok=True)
        max_lora_rank = _infer_max_lora_rank(lora_path)

        ctx = mp.get_context("spawn")
        self._req_qs = [ctx.Queue() for _ in range(nprocs)]
        self._resp_q = ctx.Queue()
        self._procs = []
        for rank in range(nprocs):
            p = ctx.Process(
                target=_worker_loop,
                args=(rank, gpu_ids[rank], model_path, lora_path, dtype,
                      max_model_len, gpu_memory_utilization, max_lora_rank,
                      cache_root, self._req_qs[rank], self._resp_q),
            )
            p.daemon = False
            p.start()
            self._procs.append(p)

        ready = 0
        while ready < nprocs:
            status, rk, info = self._resp_q.get()
            if status == "ready":
                ready += 1
            elif status == "error":
                self._terminate_all()
                raise RuntimeError(f"vLLM worker rank={rk} failed: {info}")
        print(f"[vllm_runner] {nprocs} worker(s) ready "
              f"(model={model_path}, gpus={','.join(gpu_ids)}, "
              f"tp=1 per worker, lora={lora_path or 'none'}, "
              f"max_lora_rank={max_lora_rank or 'default'})")

    def _terminate_all(self):
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        self._procs = []

    def shutdown(self):
        for q in self._req_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        self._procs = []


    def _dispatch(self, op: str, prompts: List[str], **extra) -> List:
        n = len(prompts)
        slices: List[List[int]] = [[] for _ in range(self.nprocs)]
        for i in range(n):
            slices[i % self.nprocs].append(i)

        for rank in range(self.nprocs):
            idx = slices[rank]
            sub_prompts = [prompts[i] for i in idx]
            sub_extra = dict(extra)
            if "completions" in extra:
                sub_extra["completions"] = [extra["completions"][i] for i in idx]
            self._req_qs[rank].put((op, sub_prompts, sub_extra))

        results: List = [None] * n
        pending = self.nprocs
        while pending > 0:
            status, rank, payload = self._resp_q.get()
            if status == "error":
                self._terminate_all()
                raise RuntimeError(
                    f"vLLM worker rank={rank} failed during {op}: {payload}")
            for orig_i, r in zip(slices[rank], payload):
                results[orig_i] = r
            pending -= 1
        return results


    def generate_greedy(self, prompts: List[str],
                         max_new_tokens: int = 256,
                         strip_newline: bool = False,
                         top_p: float = 0.95,
                         top_k: int = 50,
                         repetition_penalty: float = 1.05) -> List[str]:
        return self._dispatch("generate_greedy", prompts,
                              max_new_tokens=max_new_tokens,
                              strip_newline=strip_newline,
                              top_p=top_p, top_k=top_k,
                              repetition_penalty=repetition_penalty)

    def generate_samples(self, prompts: List[str], num_samples: int = 10,
                          temperature: float = 1.0,
                          max_new_tokens: int = 256,
                          top_p: float = 0.95,
                          top_k: int = 50,
                          repetition_penalty: float = 1.05,
                          strip_newline: bool = False,
                          ) -> List[List[SampleRecord]]:
        raw = self._dispatch(
            "generate_samples", prompts,
            num_samples=num_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
            strip_newline=strip_newline,
        )
        return [[SampleRecord(
            text=t,
            log_prob=lp,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            token_ids=token_ids,
        ) for t, lp, finish_reason, stop_reason, token_ids in lst]
                for lst in raw]

    def teacher_force_logprobs(self, prompts: List[str],
                                completions: List[str],
                                ) -> List[List[float]]:
        return self._dispatch("teacher_force_logprobs", prompts,
                              completions=completions)

    def first_token_softmax(self, prompts: List[str],
                             choices: List[str]) -> List[List[float]]:
        return self._dispatch("first_token_softmax", prompts,
                              choices=choices)

    def first_token_group_softmax(
        self,
        prompts: List[str],
        choice_groups: List[List[str]],
    ) -> List[List[float]]:
        return self._dispatch("first_token_group_softmax", prompts,
                              choice_groups=choice_groups)
