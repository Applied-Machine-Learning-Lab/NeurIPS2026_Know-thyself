from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.training.fok_data import FOK_DATA_VERSION


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _init_dist() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, local_rank, world, device


def _main(rank: int) -> bool:
    return rank == 0


def _safe_adapter_name(adapter_id: str) -> str:
    return "k_" + "".join(ch if ch.isalnum() else "_" for ch in adapter_id)


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _linear_targets(model) -> List[str]:
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf != "lm_head":
                targets.add(leaf)
    if not targets:
        raise RuntimeError("no Linear modules found for LoRA target discovery")
    return sorted(targets)


def _freeze_except_fok(model, fok_name: str) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = (
            f"lora_A.{fok_name}" in name
            or f"lora_B.{fok_name}" in name
            or f"lora_embedding_A.{fok_name}" in name
            or f"lora_embedding_B.{fok_name}" in name
            or f"lora_magnitude_vector.{fok_name}" in name
        )


def _active_adapters(model, adapter_names: List[str], fok_name: str) -> None:
    model.module.base_model.set_adapter(adapter_names) if isinstance(model, DDP) \
        else model.base_model.set_adapter(adapter_names)
    base = model.module if isinstance(model, DDP) else model
    _freeze_except_fok(base, fok_name)


def _load_model(args: argparse.Namespace, schedule: List[str], device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
    ).to(device)

    knowledge_map: Dict[str, str] = {}
    first_adapter_id = next((item for item in schedule if item != "base"), None)
    if first_adapter_id is None:
        targets = _linear_targets(model)
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        from peft import get_peft_model

        model = get_peft_model(model, lora_cfg, adapter_name=args.fok_adapter_name)
    else:
        first_path = _knowledge_adapter_path(args, first_adapter_id)
        first_name = _safe_adapter_name(first_adapter_id)
        model = PeftModel.from_pretrained(
            model,
            first_path,
            adapter_name=first_name,
            is_trainable=False,
        )
        knowledge_map[first_adapter_id] = first_name
        for adapter_id in schedule:
            if adapter_id in ("base", first_adapter_id):
                continue
            path = _knowledge_adapter_path(args, adapter_id)
            adapter_name = _safe_adapter_name(adapter_id)
            model.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
            knowledge_map[adapter_id] = adapter_name

        targets = sorted(model.peft_config[first_name].target_modules)
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model.add_adapter(args.fok_adapter_name, lora_cfg)

    model.config.use_cache = False
    model.train()
    _freeze_except_fok(model, args.fok_adapter_name)
    return model, knowledge_map


def _knowledge_adapter_path(args: argparse.Namespace, adapter_id: str) -> Path:
    if adapter_id == "base":
        raise ValueError("base has no knowledge adapter")
    if adapter_id == "final":
        return _repo_path(args.knowledge_lora_root)
    return _repo_path(args.knowledge_lora_root) / adapter_id


def _load_grouped_rows(args: argparse.Namespace,
                       schedule: List[str]) -> Dict[str, List[dict]]:
    data_root = _repo_path(args.data_root)
    grouped = {}
    for adapter_id in schedule:
        path = data_root / adapter_id / "fok_tokenized.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing FoK data for {adapter_id}: {path}")
        rows = _load_jsonl(path)
        if args.max_rows_per_checkpoint > 0:
            rows = rows[:args.max_rows_per_checkpoint]
        if not rows:
            raise RuntimeError(f"empty FoK data for {adapter_id}: {path}")
        bad = [
            row.get("qid") for row in rows
            if row.get("data_version") != FOK_DATA_VERSION
        ]
        if bad:
            raise RuntimeError(
                f"stale FoK data_version for {adapter_id}, first qid={bad[0]}"
            )
        grouped[adapter_id] = rows
    return grouped


def _make_batch_plan(grouped: Dict[str, List[dict]], schedule: List[str],
                     global_batch: int, seed: int, epoch: int) -> List[tuple[str, List[int]]]:
    rng = random.Random(seed + epoch)
    batches: List[tuple[str, List[int]]] = []
    for adapter_id in schedule:
        rows = grouped[adapter_id]
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        usable = len(indices) - (len(indices) % global_batch)
        indices = indices[:usable]
        for start in range(0, len(indices), global_batch):
            batches.append((adapter_id, indices[start:start + global_batch]))
    rng.shuffle(batches)
    return batches


def _pad_batch(rows: List[dict], pad_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention_mask = []
    labels = []
    class_ids = []
    for row in rows:
        n = len(row["input_ids"])
        pad = max_len - n
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
        class_ids.append(int(row["is_correct"]))
    return {
        "input_ids": torch.tensor(input_ids, device=device, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, device=device, dtype=torch.long),
        "labels": torch.tensor(labels, device=device, dtype=torch.long),
        "class_ids": torch.tensor(class_ids, device=device, dtype=torch.long),
    }


def _class_weights(rows: Iterable[dict]) -> Dict[int, float]:
    counts = {0: 0, 1: 0}
    for row in rows:
        counts[int(row["is_correct"])] += 1
    total = counts[0] + counts[1]
    weights = {}
    for cls in (0, 1):
        weights[cls] = total / max(2 * counts[cls], 1)
    return weights


def _loss(logits: torch.Tensor, labels: torch.Tensor,
          class_ids: torch.Tensor, weights: Dict[int, float]) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    per_sample = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    sample_weights = torch.tensor(
        [weights[int(cls)] for cls in class_ids.detach().cpu().tolist()],
        device=per_sample.device,
        dtype=per_sample.dtype,
    )
    return (per_sample * sample_weights).mean()


def _save_adapter(model, output_dir: Path, rank: int, adapter_name: str) -> None:
    if not _main(rank):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.save_pretrained(
        str(output_dir),
        selected_adapters=[adapter_name],
        safe_serialization=True,
    )
    nested = output_dir / adapter_name
    if nested.exists() and not (output_dir / "adapter_config.json").exists():
        for item in nested.iterdir():
            dst = output_dir / item.name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(item), str(dst))
        nested.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--knowledge_lora_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--schedule", nargs="+", required=True)
    parser.add_argument("--method_version", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--max_rows_per_checkpoint", type=int, default=0)
    parser.add_argument("--fok_adapter_name", default="fok")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world, device = _init_dist()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    output_dir = _repo_path(args.output_dir)
    meta_path = output_dir / "mixed_train_meta.json"
    if (not args.force and (output_dir / "adapter_config.json").exists()
            and meta_path.exists()):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
        if meta.get("method_version") == args.method_version:
            if _main(rank):
                print(f"[skip] current mixed FoK adapter exists: {output_dir}")
            return

    schedule = list(args.schedule)
    grouped = _load_grouped_rows(args, schedule)
    if _main(rank):
        print("[stage2-mixed] schedule=" + ",".join(schedule), flush=True)
        for adapter_id in schedule:
            rows = grouped[adapter_id]
            pos = sum(int(row["is_correct"]) == 1 for row in rows)
            neg = len(rows) - pos
            print(f"[stage2-mixed] data {adapter_id}: n={len(rows)} pos={pos} neg={neg}",
                  flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, knowledge_map = _load_model(args, schedule, device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    trainable = [
        param for param in model.parameters()
        if param.requires_grad
    ]
    if not trainable:
        raise RuntimeError("no trainable FoK LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    global_batch = args.per_device_train_batch_size * world
    total_steps = 0
    for epoch in range(args.epochs):
        plan = _make_batch_plan(
            grouped=grouped,
            schedule=schedule,
            global_batch=global_batch,
            seed=args.seed,
            epoch=epoch,
        )
        if _main(rank):
            print(f"[stage2-mixed] epoch={epoch + 1}/{args.epochs} steps={len(plan)}",
                  flush=True)
        for step, (adapter_id, indices) in enumerate(plan, start=1):
            start = rank * args.per_device_train_batch_size
            end = start + args.per_device_train_batch_size
            local_indices = indices[start:end]
            rows = [grouped[adapter_id][idx] for idx in local_indices]
            if adapter_id == "base":
                active = [args.fok_adapter_name]
            else:
                active = [knowledge_map[adapter_id], args.fok_adapter_name]
            _active_adapters(model, active, args.fok_adapter_name)
            batch = _pad_batch(rows, tokenizer.pad_token_id, device)
            weights = _class_weights(grouped[adapter_id])
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = _loss(
                outputs.logits,
                batch["labels"],
                batch["class_ids"],
                weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            total_steps += 1
            if _main(rank) and (
                    total_steps == 1 or total_steps % args.logging_steps == 0):
                print(
                    f"[stage2-mixed] step={total_steps} "
                    f"epoch_step={step}/{len(plan)} ckpt={adapter_id} "
                    f"loss={float(loss.detach().cpu()):.4f}",
                    flush=True,
                )

    if world > 1:
        dist.barrier()
    _save_adapter(model, output_dir, rank, args.fok_adapter_name)
    if _main(rank):
        meta = {
            "method_version": args.method_version,
            "trainer": "kdelta_fok_lora_mixed",
            "batch_policy": "one_knowledge_checkpoint_per_batch_interleaved",
            "loss_correction": "inverse_class_frequency_per_checkpoint",
            "model": args.model,
            "knowledge_lora_root": str(_repo_path(args.knowledge_lora_root)),
            "data_root": str(_repo_path(args.data_root)),
            "schedule": schedule,
            "epochs": args.epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "world_size": world,
            "global_batch_size": global_batch,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
            "total_steps": total_steps,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        print(f"[done] mixed FoK LoRA -> {output_dir}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
