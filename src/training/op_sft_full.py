from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
OPSFT_L2_RAW_DATA_VERSION = "opsft_l2_32sample_raw_replacement_v1"
OPSFT_L2_RAW_DATA_VERSION_DROPPED = "opsft_l2_32sample_raw_replacement_drop_correct_v1"
OPSFT_L2_RAW_DATA_VERSION_FULLSUP = "opsft_l2_32sample_raw_full_supervision_drop_correct_v1"
OPSFT_L2_FULL_STAGE1_VERSION = "opsft_l2_replacement_full_sft_v1"


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _find_llamafactory(root: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if root:
        candidates.append(Path(root))
    candidates.extend([
        _REPO_ROOT / "third_party" / "LlamaFactory",
        _REPO_ROOT.parent / "LlamaFactory",
    ])
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "src" / "llamafactory").exists():
            return candidate.resolve()
    return None


def _find_prepare_script(script: Optional[str]) -> Path:
    if script:
        path = _repo_path(script)
        if path.exists():
            return path
    return (_REPO_ROOT / "src" / "data" / "prepare_ksft_data.py").resolve()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_env(llamafactory_root: Optional[Path], nproc: int) -> Dict[str, str]:
    env = os.environ.copy()
    current_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = current_bin + os.pathsep + env.get("PATH", "")
    if llamafactory_root is not None:
        paths = [str(llamafactory_root / "src")]
        existing = env.get("PYTHONPATH")
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["NNODES"] = "1"
    env["NODE_RANK"] = "0"
    env["NPROC_PER_NODE"] = str(nproc)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(_free_port())
    if nproc > 1:
        env["FORCE_TORCHRUN"] = "1"
    return env


def _write_yaml(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _patch_llamafactory_l2(llamafactory_root: Path) -> None:
    from src.training.llamafactory_full_sft import _patch_llamafactory_l2

    _patch_llamafactory_l2(llamafactory_root)


def _count_json_array(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except (OSError, json.JSONDecodeError):
        return 0


def _prepare_data(args: argparse.Namespace) -> int:
    output_json = _repo_path(args.dataset_json)
    meta_path = _repo_path(args.data_meta)
    corrected = _repo_path(args.corrected_file)
    answers = _repo_path(args.answers_file)
    drop_correct = bool(getattr(args, "drop_correct", False))
    full_supervision = bool(getattr(args, "full_supervision", False))
    unify_replacement = bool(getattr(args, "unify_replacement", False))
    unlikelihood = bool(getattr(args, "unlikelihood", False))
    if unlikelihood:
        version_parts = ["opsft_l2_32sample_unlikelihood"]
    else:
        version_parts = ["opsft_l2_32sample_raw_replacement"]
    if drop_correct:
        version_parts.append("drop_correct")
    if full_supervision:
        version_parts.append("fullsup")
    if unify_replacement:
        version_parts.append("unify")
    version_parts.append("v1")
    expected_version = "_".join(version_parts)
    current = False
    if output_json.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            current = (
                meta.get("data_version") == expected_version and
                Path(meta.get("corrected_file", "")).resolve() == corrected.resolve() and
                Path(meta.get("answers_file", "")).resolve() == answers.resolve() and
                bool(meta.get("drop_correct", False)) == drop_correct and
                bool(meta.get("full_supervision", False)) == full_supervision and
                bool(meta.get("unify_replacement", False)) == unify_replacement and
                bool(meta.get("unlikelihood", False)) == unlikelihood
            )
        except (json.JSONDecodeError, OSError):
            current = False
    if current and not args.force:
        count = _count_json_array(output_json)
        print(f"[skip] current OP-SFT raw dataset exists: {output_json} ({count})")
        return count

    if not corrected.exists():
        raise SystemExit(f"[error] corrected_file not found: {corrected}")
    if not answers.exists():
        raise SystemExit(f"[error] answers_file not found: {answers}")

    prepare_script = _find_prepare_script(args.prepare_script)
    mode = "unlikelihood" if unlikelihood else "replacement"
    cmd = [
        sys.executable, str(prepare_script),
        "--corrected_file", str(corrected),
        "--answers_file", str(answers),
        "--output", str(output_json),
        "--mode", mode,
    ]
    if drop_correct:
        cmd.append("--drop_correct")
    if full_supervision:
        cmd.append("--full_supervision")
    if unify_replacement:
        cmd.append("--unify_replacement")
    print("\n$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"[error] OP-SFT data preparation failed: {proc.returncode}")

    count = _count_json_array(output_json)
    meta = {
        "data_version": expected_version,
        "corrected_file": str(corrected),
        "answers_file": str(answers),
        "dataset_json": str(output_json),
        "training_samples": count,
        "drop_correct": drop_correct,
        "full_supervision": full_supervision,
        "unify_replacement": unify_replacement,
        "unlikelihood": unlikelihood,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] OP-SFT raw dataset -> {output_json} ({count})")
    return count


def _model_current(output_dir: Path, train_meta: Path,
                   expected_version: str, train_config: Dict) -> bool:
    if not (output_dir / "config.json").exists() or not train_meta.exists():
        return False
    try:
        meta = json.loads(train_meta.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get("method_version") == expected_version and \
        meta.get("train_config") == train_config


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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"],
                        default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--l2_lambda", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--pure_bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _repo_path(args.output_dir)
    dataset_json = _repo_path(args.dataset_json)
    train_meta = output_dir / "lf_train_meta.json"

    if args.dry:
        print(f"[dry] OP-SFT data: {dataset_json}")
        print(f"[dry] output: {output_dir}")
        return

    sample_count = _prepare_data(args)
    if args.prepare_only:
        return

    llamafactory_root = _find_llamafactory(args.llamafactory_root)
    if llamafactory_root is None:
        raise SystemExit("[error] LlamaFactory not found")
    _patch_llamafactory_l2(llamafactory_root)

    train_config: Dict = {
        "model": args.model,
        "dataset_json": str(dataset_json),
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "deepspeed": str(_repo_path(args.deepspeed)) if args.deepspeed else None,
        "l2_lambda": args.l2_lambda,
        "weight_decay": args.weight_decay,
        "pure_bf16": bool(args.pure_bf16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "training_samples": sample_count,
    }
    if not args.force and _model_current(
            output_dir, train_meta, args.method_version, train_config):
        print(f"[skip] current OP-SFT L2 full model exists: {output_dir}")
        return

    dataset_name = dataset_json.stem
    cfg: Dict = {
        "model_name_or_path": args.model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "use_l2_reg": True,
        "l2_lambda": args.l2_lambda,
        "l2_base_model": args.model,
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
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": args.weight_decay,
        "bf16": True,
        "ddp_timeout": 180000000,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "disable_gradient_checkpointing": not bool(args.gradient_checkpointing),
    }
    if args.pure_bf16:
        cfg["pure_bf16"] = True
    if args.deepspeed:
        cfg["deepspeed"] = str(_repo_path(args.deepspeed))
    yaml_path = output_dir.parent / f"{output_dir.name}_opsft_l2_train.yaml"
    _write_yaml(yaml_path, cfg)

    cmd = [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)]
    print("\n$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=_build_env(llamafactory_root, args.nproc),
    )
    if proc.returncode != 0:
        raise SystemExit(f"[error] LLaMA Factory failed with code {proc.returncode}")
    if not (output_dir / "config.json").exists():
        raise SystemExit(f"[error] config.json missing after training: {output_dir}")

    meta = {
        "method_version": args.method_version,
        "llamafactory_root": str(llamafactory_root),
        "yaml": str(yaml_path),
        "train_config": train_config,
    }
    train_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] OP-SFT L2 full model -> {output_dir}")


if __name__ == "__main__":
    main()
