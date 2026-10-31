from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn

from src.baselines.base import HFDDPBaselineScorer
from src.baselines.utils import run_hf_dp
from src.baselines.utils.prompt import build_qa_prompt


DEFAULT_ELICAL_REPO = "Shiyunee/Honest-Qwen2.5-7B-Instruct"
DEFAULT_ELICAL_PRESETS = {
    "1k": {
        "lora": (
            "lora/hybrid_answer_conf/long_qa/"
            "batchsize16_accumulation8_epochs50_weightdecay0.1_"
            "r8_alpha16_loradrpout0.0_1k_training_samples/"
            "best-checkpoint/lora_epoch_best"
        ),
        "head": (
            "lora/hybrid_answer_conf/long_qa/"
            "batchsize16_accumulation8_epochs50_weightdecay0.1_"
            "r8_alpha16_loradrpout0.0_1k_training_samples/"
            "best-checkpoint/vector_head_epoch_best.pt"
        ),
    },
    "560k": {
        "lora": (
            "lora/hybrid_answer_conf/long_qa/"
            "batchsize16_accumulation8_epochs10_weightdecay0.1_"
            "r8_alpha16_loradrpout0.0/"
            "best-checkpoint/lora_epoch_best"
        ),
        "head": (
            "lora/hybrid_answer_conf/long_qa/"
            "batchsize16_accumulation8_epochs10_weightdecay0.1_"
            "r8_alpha16_loradrpout0.0/"
            "best-checkpoint/vector_head_epoch_best.pt"
        ),
    },
}


def _adapter_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    return p if (p / "adapter_config.json").exists() else None


def _parent_model(path: str) -> Optional[str]:
    parent = Path(path) / "parent_model.txt"
    return parent.read_text().strip() if parent.exists() else None


def _resolve_paths(repo_id: str, preset: str, lora_path: Optional[str],
                   vector_head_path: Optional[str]) -> Tuple[str, str]:
    if lora_path and vector_head_path:
        return lora_path, vector_head_path
    if preset not in DEFAULT_ELICAL_PRESETS:
        raise ValueError(
            f"unknown EliCal preset {preset}; expected one of "
            f"{sorted(DEFAULT_ELICAL_PRESETS)}"
        )
    spec = DEFAULT_ELICAL_PRESETS[preset]
    from huggingface_hub import snapshot_download

    allow = [spec["lora"] + "/*", spec["head"]]
    root = snapshot_download(repo_id=repo_id, allow_patterns=allow)
    return str(Path(root) / spec["lora"]), str(Path(root) / spec["head"])


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip().split("\n")[0]


def _load_combined_lm(score_base_model: str, pre_adapter_path: Optional[str],
                      elical_lora_path: str, device: str,
                      dtype: torch.dtype):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        score_base_model, torch_dtype=dtype, trust_remote_code=True)
    if pre_adapter_path:
        model = PeftModel.from_pretrained(model, pre_adapter_path)
        model = model.merge_and_unload()

    if not hasattr(model, "model"):
        raise ValueError(
            "EliCal adapter merge expects a CausalLM with a `.model` backbone"
        )
    backbone = PeftModel.from_pretrained(model.model, elical_lora_path)
    model.model = backbone.merge_and_unload()
    return model.to(device).eval()


def _load_elical_model(score_base_model: str, pre_adapter_path: Optional[str],
                       elical_lora_path: str, vector_head_path: str,
                       device: str, dtype: torch.dtype):
    model = _load_combined_lm(
        score_base_model=score_base_model,
        pre_adapter_path=pre_adapter_path,
        elical_lora_path=elical_lora_path,
        device=device,
        dtype=dtype,
    )
    hidden_size = model.config.hidden_size
    head = nn.Linear(hidden_size, 1).to(device)
    state = torch.load(vector_head_path, map_location="cpu")
    head.load_state_dict(state)
    head = head.to(dtype=torch.float32).eval()
    return model.eval(), head


def _score_worker(rank, device, questions, indices, extras):
    from transformers import AutoTokenizer

    dtype = getattr(torch, extras["dtype"])
    score_base_model = extras["score_base_model"]
    pre_adapter_path = extras.get("pre_adapter_path")
    elical_lora_path = extras["elical_lora_path"]
    vector_head_path = extras["vector_head_path"]
    batch_size = extras["batch_size"]
    max_length = extras["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(
        score_base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model, head = _load_elical_model(
        score_base_model=score_base_model,
        pre_adapter_path=pre_adapter_path,
        elical_lora_path=elical_lora_path,
        vector_head_path=vector_head_path,
        device=device,
        dtype=dtype,
    )

    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            prompts = [build_qa_prompt(tokenizer, q) for q in batch]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length).to(device)
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            hidden = out.hidden_states[-1][
                torch.arange(len(batch), device=device), last_idx
            ].float()
            prob = torch.sigmoid(head(hidden).squeeze(-1))
            scores.extend(prob.cpu().tolist())
    return scores


def _answer_worker(rank, device, questions, indices, extras):
    from transformers import AutoTokenizer

    dtype = getattr(torch, extras["dtype"])
    score_base_model = extras["score_base_model"]
    pre_adapter_path = extras.get("pre_adapter_path")
    elical_lora_path = extras["elical_lora_path"]
    batch_size = extras["batch_size"]
    max_new_tokens = extras["max_new_tokens"]

    tokenizer = AutoTokenizer.from_pretrained(
        score_base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = _load_combined_lm(
        score_base_model=score_base_model,
        pre_adapter_path=pre_adapter_path,
        elical_lora_path=elical_lora_path,
        device=device,
        dtype=dtype,
    )

    answers: List[str] = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            prompts = [build_qa_prompt(tokenizer, q) for q in batch]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=1024).to(device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
            new = gen[:, enc.input_ids.shape[1]:]
            for row in new:
                text = tokenizer.decode(row, skip_special_tokens=True)
                answers.append(extract_answer(text))
    return answers


class EliCalScorer(HFDDPBaselineScorer):
    name = "elical"

    def __init__(self, model_path: str, score_base_model: Optional[str] = None,
                 pre_adapter_path: Optional[str] = None,
                 elical_repo: str = DEFAULT_ELICAL_REPO,
                 elical_preset: str = "1k",
                 elical_lora_path: Optional[str] = None,
                 vector_head_path: Optional[str] = None,
                 nprocs: int = 8, dtype: str = "bfloat16",
                 batch_size: int = 8, max_length: int = 1024,
                 max_new_tokens: int = 256,
                 **_):
        super().__init__(model_path=model_path, nprocs=nprocs, dtype=dtype,
                         batch_size=batch_size)
        self.elical_repo = elical_repo
        self.elical_preset = elical_preset
        self.elical_lora_path, self.vector_head_path = _resolve_paths(
            elical_repo, elical_preset, elical_lora_path, vector_head_path)
        model_adapter = _adapter_dir(model_path)
        self.score_base_model = (
            score_base_model or _parent_model(model_path) or model_path
        )
        self.pre_adapter_path = pre_adapter_path or (
            str(model_adapter) if model_adapter else None
        )
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        if _adapter_dir(self.score_base_model) and not score_base_model:
            raise ValueError(
                "EliCal scoring needs a base model directory or HF id. "
                "Pass --elical_score_base_model when --model is a PEFT adapter."
            )

    def score(self, questions: List[str],
              greedy_answers: Optional[List[str]] = None) -> List[float]:
        del greedy_answers
        extras = dict(
            score_base_model=self.score_base_model,
            pre_adapter_path=self.pre_adapter_path,
            elical_lora_path=self.elical_lora_path,
            vector_head_path=self.vector_head_path,
            dtype=self.dtype,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        return run_hf_dp(
            _score_worker, questions, nprocs=self.nprocs, extras=extras)

    def generate_answers(self, questions: List[str]) -> List[str]:
        extras = dict(
            score_base_model=self.score_base_model,
            pre_adapter_path=self.pre_adapter_path,
            elical_lora_path=self.elical_lora_path,
            dtype=self.dtype,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
        )
        return run_hf_dp(
            _answer_worker, questions, nprocs=self.nprocs, extras=extras)
