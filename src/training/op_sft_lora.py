from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml

from src.training.op_sft_full import (
    OPSFT_L2_RAW_DATA_VERSION,
    _build_env,
    _find_llamafactory,
    _prepare_data,
    _repo_path,
)


def _write_yaml(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=10**9))


def _adapter_current(output_dir: Path, train_meta: Path,
                     expected_version: str,
                     expected_train_config: Dict) -> bool:
    if not (output_dir / "adapter_config.json").exists():
        return False
    if not train_meta.exists():
        return False
    try:
        meta = json.loads(train_meta.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get("method_version") == expected_version and \
        meta.get("train_config") == expected_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corrected_file", required=True)
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--data_meta", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method_version", required=True)
    parser.add_argument("--prepare_script", default=None)
    parser.add_argument("--llamafactory_root", default=None)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="AdamW weight decay (decoupled L2) applied to LoRA params. "
                             "LLaMA-Factory default is 0; set >0 to softly pull LoRA "
                             "weights toward zero at unsupervised positions.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"],
                        default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None,
                        help="Training seed passed to LLaMA Factory. Only recorded "
                             "in train_config when explicitly set, so existing "
                             "adapters trained at the default seed still skip.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--drop_correct", action="store_true",
                        help="Drop is_correct=True samples entirely (prepare_ksft_data).")
    parser.add_argument("--full_supervision", action="store_true",
                        help="Supervise the full assistant_content (prefix+replacement) "
                             "instead of only the replacement span. Plain answer-only SFT.")
    parser.add_argument("--unify_replacement", action="store_true",
                        help="Unify replacement token to primary_answer for all wrong "
                             "samples of the same question.")
    parser.add_argument("--lora_layers_to_transform", default=None,
                        help="Comma-separated layer indices to apply LoRA to "
                             "(e.g. '0,1,2,...,17'). Empty / 'all' = all layers.")
    parser.add_argument("--unlikelihood", action="store_true",
                        help="Welleck unlikelihood (form B) training: wrong samples "
                             "get -log(1-p) at first-wrong-token only; correct samples "
                             "get normal full SFT. Requires prepare_ksft_data.py to "
                             "emit ul_wrong_prefix (mode=unlikelihood).")
    parser.add_argument("--unlikelihood_alpha", type=float, default=1.0,
                        help="Scalar multiplier on the UL loss term.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _repo_path(args.output_dir)
    dataset_json = _repo_path(args.dataset_json)
    train_meta = output_dir / "lf_train_meta.json"

    if args.dry:
        print(f"[dry] OP-SFT LoRA data: {dataset_json}")
        print(f"[dry] output: {output_dir}")
        return

    sample_count = _prepare_data(args)
    llamafactory_root = _find_llamafactory(args.llamafactory_root)
    if llamafactory_root is None:
        raise SystemExit("[error] LlamaFactory not found")

    layers_to_transform = None
    if args.lora_layers_to_transform and args.lora_layers_to_transform.lower() not in ("all", "none", ""):
        layers_to_transform = sorted({
            int(x) for x in args.lora_layers_to_transform.split(",") if x.strip()
        })

    train_config: Dict = {
        "model": args.model,
        "dataset_json": str(dataset_json),
        "data_version": OPSFT_L2_RAW_DATA_VERSION,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_layers_to_transform": layers_to_transform,
        "drop_correct": bool(args.drop_correct),
        "full_supervision": bool(args.full_supervision),
        "unify_replacement": bool(args.unify_replacement),
        "unlikelihood": bool(args.unlikelihood),
        "unlikelihood_alpha": float(args.unlikelihood_alpha),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "deepspeed": str(_repo_path(args.deepspeed)) if args.deepspeed else None,
        "training_samples": sample_count,
    }
    if args.seed is not None:
        train_config["seed"] = int(args.seed)
    if not args.force and _adapter_current(
            output_dir, train_meta, args.method_version, train_config):
        print(f"[skip] current OP-SFT LoRA adapter exists: {output_dir}")
        return

    dataset_name = dataset_json.stem
    cfg: Dict = {
        "model_name_or_path": args.model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dataset": dataset_name,
        "dataset_dir": str(dataset_json.parent),
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "output_dir": str(output_dir),
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": "none",
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "ddp_timeout": 180000000,
    }
    if args.deepspeed:
        cfg["deepspeed"] = str(_repo_path(args.deepspeed))
    if layers_to_transform is not None:
        cfg["lora_layers_to_transform"] = layers_to_transform
    if args.unlikelihood:
        cfg["use_unlikelihood"] = True
        cfg["unlikelihood_alpha"] = float(args.unlikelihood_alpha)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    yaml_path = output_dir.parent / f"{output_dir.name}_opsft_lora_train.yaml"
    _write_yaml(yaml_path, cfg)

    cmd = [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)]
    print("\n$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=_build_env(llamafactory_root, args.nproc),
    )
    if proc.returncode != 0:
        raise SystemExit(f"[error] LLaMA Factory failed with code {proc.returncode}")
    if not (output_dir / "adapter_config.json").exists():
        raise SystemExit(f"[error] adapter_config.json missing after training: {output_dir}")

    meta = {
        "method_version": args.method_version,
        "llamafactory_root": str(llamafactory_root),
        "yaml": str(yaml_path),
        "train_config": train_config,
    }
    train_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] OP-SFT LoRA adapter -> {output_dir}")


if __name__ == "__main__":
    main()
