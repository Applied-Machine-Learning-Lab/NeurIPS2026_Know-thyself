from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baselines.pipeline.common import (
    compute_auroc,
    compute_threshold_metrics,
)
from src.training.fok_data import FOK_SCORE_VERSION, build_fok_prompt


YES_CHOICES = ["Yes", " Yes", "yes", " yes", "YES"]
NO_CHOICES = ["No", " No", "no", " no", "NO"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--knowledge_lora", default=None)
    ap.add_argument("--fok_lora", default=None)
    ap.add_argument("--input_jsonl", required=True,
                    help="greedy.jsonl with qid/question/is_correct")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt_style", choices=["fok", "answerability"], default="fok",
                    help="fok = self-capability question (paper); answerability = "
                         "asks whether the question itself is answerable (SelfAware-aligned)")
    return ap.parse_args()


def _init_dp() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, local_rank, world


def _read_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    tmp.replace(path)


def _dtype(name: str):
    return getattr(torch, name)


def _load_model(args: argparse.Namespace, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=_dtype(args.dtype),
        device_map={"": device},
        trust_remote_code=True,
    )
    if args.knowledge_lora:
        model = PeftModel.from_pretrained(
            model, args.knowledge_lora, is_trainable=False)
        if args.fok_lora:
            model = model.merge_and_unload()
            model = PeftModel.from_pretrained(
                model, args.fok_lora, is_trainable=False)
    elif args.fok_lora:
        model = PeftModel.from_pretrained(
            model, args.fok_lora, is_trainable=False)
    model.eval()
    return model


@contextlib.contextmanager
def _adapter_context(model):
    if hasattr(model, "disable_adapter"):
        yield
    else:
        yield


def _choice_ids(tokenizer, choices: Iterable[str]) -> List[int]:
    ids = []
    for choice in choices:
        toks = tokenizer(choice, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks[0])
    return sorted(set(ids))


def _logsumexp(values: List[torch.Tensor]) -> torch.Tensor:
    return torch.logsumexp(torch.stack(values), dim=0)


def _last_non_pad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    return (
        attention_mask.size(1)
        - 1
        - attention_mask.flip(dims=[1]).long().argmax(dim=1)
    )


@torch.no_grad()
def _p_yes(model, tokenizer, prompts: List[str], device: torch.device,
           batch_size: int, max_prompt_len: int) -> List[float]:
    yes_ids = _choice_ids(tokenizer, YES_CHOICES)
    no_ids = _choice_ids(tokenizer, NO_CHOICES)
    if not yes_ids or not no_ids:
        raise RuntimeError("Yes/No token groups are empty")

    scores = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)
        last = _last_non_pad_indices(enc["attention_mask"])
        logits = model(**enc).logits[torch.arange(len(batch), device=device), last]
        yes_log = _logsumexp([logits[:, tid] for tid in yes_ids])
        no_log = _logsumexp([logits[:, tid] for tid in no_ids])
        probs = torch.stack([no_log, yes_log], dim=-1).float().softmax(-1)[:, 1]
        scores.extend(float(x) for x in probs.cpu().tolist())
    return scores


def _shard_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.name}.rank{rank}.tmp")


def _merge_shards(output_jsonl: Path, world: int) -> List[dict]:
    rows = []
    for rank in range(world):
        shard = _shard_path(output_jsonl, rank)
        if not shard.exists():
            raise FileNotFoundError(f"missing shard: {shard}")
        rows.extend(_read_jsonl(shard))
        shard.unlink()
    rows.sort(key=lambda row: row["_idx"])
    for row in rows:
        row.pop("_idx", None)
    _write_jsonl(rows, output_jsonl)
    return rows


def _write_metrics(rows: List[dict], path: Path) -> dict:
    labels = [int(bool(row["is_correct"])) for row in rows]
    scores = [float(row["score"]) for row in rows]
    tm = compute_threshold_metrics(scores, labels, 0.5)
    metrics = {
        "n": len(rows),
        "answer_acc": sum(labels) / max(len(labels), 1),
        "auroc": compute_auroc(scores, labels),
        "threshold_acc": tm["accuracy"],
        "tpr": tm["tpr"],
        "tnr": tm["tnr"],
        "illusion": 1.0 - tm["tnr"],
        "omission": 1.0 - tm["tpr"],
        "score_version": FOK_SCORE_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    args = parse_args()
    rank, local_rank, world = _init_dp()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    all_rows = _read_jsonl(Path(args.input_jsonl))
    indexed = list(enumerate(all_rows))
    shard = indexed[rank::world]
    rows = [row for _, row in shard]
    if rank == 0:
        print(f"[info] n={len(all_rows)} world={world}")
    print(f"[rank {rank}] shard={len(rows)} device={device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = _load_model(args, device)
    prompts = [build_fok_prompt(tokenizer, row["question"], style=args.prompt_style) for row in rows]
    with _adapter_context(model):
        scores = _p_yes(
            model, tokenizer, prompts, device,
            args.batch_size, args.max_prompt_len)

    out_rows = []
    for (orig_idx, row), score in zip(shard, scores):
        out_rows.append({
            "qid": row.get("qid") or row.get("id"),
            "question": row["question"],
            "is_correct": int(bool(row["is_correct"])),
            "score": float(score),
            "answer": row.get("answer"),
            "judge_yes_prob": row.get("judge_yes_prob"),
            "judge_no_prob": row.get("judge_no_prob"),
            "_idx": orig_idx,
        })

    output_jsonl = Path(args.output_jsonl)
    if world > 1:
        _write_jsonl(out_rows, _shard_path(output_jsonl, rank))
        dist.barrier()
        if rank == 0:
            out_rows = _merge_shards(output_jsonl, world)
            metrics = _write_metrics(out_rows, Path(args.metrics_json))
            print(f"[done] {metrics}")
        dist.barrier()
        dist.destroy_process_group()
    else:
        for row in out_rows:
            row.pop("_idx", None)
        _write_jsonl(out_rows, output_jsonl)
        metrics = _write_metrics(out_rows, Path(args.metrics_json))
        print(f"[done] {metrics}")


if __name__ == "__main__":
    main()
