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


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _find_llamafactory(root: Optional[str]) -> Optional[Path]:
    candidates = []
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


def _patch_llamafactory_top1_miss_loss(llamafactory_root: Path) -> None:
    hparams = llamafactory_root / "src" / "llamafactory" / "hparams" / "finetuning_args.py"
    trainer = llamafactory_root / "src" / "llamafactory" / "train" / "sft" / "trainer.py"

    hparams_text = hparams.read_text()
    if "use_top1_miss_loss" not in hparams_text:
        marker = "    use_sample_weight: bool = field(\n"
        insert = (
            "    use_top1_miss_loss: bool = field(\n"
            "        default=False,\n"
            "        metadata={\"help\": \"Only backpropagate label tokens whose current top-1 prediction is not the target token.\"},\n"
            "    )\n"
        )
        if marker not in hparams_text:
            raise RuntimeError(f"cannot patch LLaMA Factory hparams: marker missing in {hparams}")
        hparams.write_text(hparams_text.replace(marker, insert + marker))

    trainer_text = trainer.read_text()
    if "use_top1_miss_loss" in trainer_text:
        return
    marker = '''        elif sample_weights is not None and self.finetuning_args.use_sample_weight:
            # Per-sample weighted loss for class balancing
'''
    insert = '''        elif self.finetuning_args.use_top1_miss_loss:
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            label_mask = shift_labels != IGNORE_INDEX
            with torch.no_grad():
                top1 = shift_logits.argmax(dim=-1)
                miss_mask = label_mask & top1.ne(shift_labels)

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            denom = miss_mask.sum().clamp(min=1)
            loss = (per_token_loss * miss_mask.to(per_token_loss.dtype)).sum() / denom

            if self.state.global_step % 20 == 0:
                total = int(label_mask.sum().item())
                missed = int(miss_mask.sum().item())
                ratio = (missed / total) if total else 0.0
                logger.info_rank0(
                    f"top1_miss_loss step={self.state.global_step} "
                    f"missed={missed} total={total} ratio={ratio:.4f}"
                )

            return loss

''' + marker
    if marker not in trainer_text:
        raise RuntimeError(f"cannot patch LLaMA Factory trainer: marker missing in {trainer}")
    trainer.write_text(trainer_text.replace(marker, insert))


def _patch_llamafactory_l2(llamafactory_root: Path) -> None:
    hparams = llamafactory_root / "src" / "llamafactory" / "hparams" / "finetuning_args.py"
    trainer = llamafactory_root / "src" / "llamafactory" / "train" / "sft" / "trainer.py"
    hparams_text = hparams.read_text()
    if "use_l2_reg" not in hparams_text:
        marker = "    freeze_vision_tower: bool = field(\n"
        insert = (
            "    use_l2_reg: bool = field(\n"
            "        default=False,\n"
            "        metadata={\"help\": \"Whether to use L2 parameter regularization against a base model.\"},\n"
            "    )\n"
            "    l2_lambda: float = field(\n"
            "        default=1e-5,\n"
            "        metadata={\"help\": \"Lambda for L2 regularization. Gradient injection: param.grad += 2λ(θ − θ_base).\"},\n"
            "    )\n"
            "    l2_base_model: str | None = field(\n"
            "        default=None,\n"
            "        metadata={\"help\": \"Path to the base model for L2 regularization reference parameters.\"},\n"
            "    )\n"
        )
        if marker not in hparams_text:
            raise RuntimeError(f"cannot patch LLaMA Factory hparams: marker missing in {hparams}")
        hparams.write_text(hparams_text.replace(marker, insert + marker))

    trainer_text = trainer.read_text()
    if "self.l2_base_params" not in trainer_text:
        raise RuntimeError(
            f"cannot patch LLaMA Factory trainer: L2 parameter-loading block missing in {trainer}"
        )

    loss_block = '''        # L2 regularization: add λ × Σ||θ - θ_base||² to loss.
        if self.l2_base_params:
            l2_val = torch.tensor(0.0, device=ce_loss.device)
            n_params = 0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                key = name
                while key.startswith("module."):
                    key = key[len("module."):]
                base = self.l2_base_params.get(key)
                if base is None:
                    continue
                base = base.to(param.device, dtype=param.dtype, non_blocking=True)
                l2_val = l2_val + ((param - base) ** 2).sum()
                n_params += 1
            l2_loss = self.l2_lambda * l2_val

            if self.state.global_step % 20 == 0:
                logger.info_rank0(
                    f"step={self.state.global_step} ce={ce_loss.item():.4f} "
                    f"l2={l2_val.item():.4f} l2_scaled={l2_loss.item():.2e} params={n_params}"
                )

            return ce_loss + l2_loss

        return ce_loss
'''

    if loss_block in trainer_text:
        trainer_text = trainer_text.replace(loss_block, "        return ce_loss\n", 1)

    inject_block = '''    def _base_param_key(self, name: str) -> str:
        while name.startswith("module."):
            name = name[len("module."):]
        return name

    def _inject_l2_grads(self, model) -> None:
        if not self.l2_base_params:
            return

        scale = 1.0 / max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        log_this_step = self.state.global_step % 20 == 0
        l2_val = 0.0
        n_params = 0

        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            key = self._base_param_key(name)
            base = self.l2_base_params.get(key)
            if base is None:
                continue
            base = base.to(param.device, dtype=param.dtype, non_blocking=True)
            param.grad.data.add_(2.0 * self.l2_lambda * scale * (param.data - base))
            if log_this_step:
                l2_val += ((param.detach() - base) ** 2).sum().item()
                n_params += 1

        if log_this_step and n_params:
            logger.info_rank0(
                f"l2_grad step={self.state.global_step} "
                f"l2={l2_val:.4f} l2_scaled={self.l2_lambda * l2_val:.2e} "
                f"params={n_params}"
            )

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        result = super().training_step(model, inputs, *args, **kwargs)
        self._inject_l2_grads(model)
        return result
'''

    pred_marker = "\n    @override\n    def prediction_step"
    if pred_marker not in trainer_text:
        raise RuntimeError(
            f"cannot patch LLaMA Factory trainer: prediction_step marker missing in {trainer}"
        )
    pred_start = trainer_text.index(pred_marker)
    if "    def _base_param_key" in trainer_text:
        start = trainer_text.index("    def _base_param_key")
        trainer_text = trainer_text[:start] + inject_block + trainer_text[pred_start:]
    else:
        step_marker = "\n    @override\n    def training_step"
        if step_marker in trainer_text:
            start = trainer_text.index(step_marker) + 1
            trainer_text = trainer_text[:start] + inject_block + trainer_text[pred_start:]
        else:
            trainer_text = trainer_text[:pred_start + 1] + inject_block + trainer_text[pred_start + 1:]

    trainer.write_text(trainer_text)


def _model_current(output_dir: Path, train_meta: Path,
                   expected_version: str,
                   expected_train_config: Dict | None = None) -> bool:
    if not (output_dir / "config.json").exists():
        return False
    if not train_meta.exists():
        return False
    try:
        meta = json.loads(train_meta.read_text())
    except json.JSONDecodeError:
        return False
    if meta.get("method_version") != expected_version:
        return False
    if expected_train_config is None:
        return True
    return meta.get("train_config") == expected_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenized_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method_version", required=True)
    parser.add_argument("--llamafactory_root", default=None)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"],
                        default="steps")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--l2_lambda", type=float, default=0.0)
    parser.add_argument("--l2_base_model", default=None)
    parser.add_argument("--top1_miss_loss", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        default=True)
    parser.add_argument("--no_gradient_checkpointing",
                        dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--pure_bf16", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _repo_path(args.output_dir)
    train_meta = output_dir / "lf_train_meta.json"

    tokenized_path = _repo_path(args.tokenized_path)
    if not args.dry and not tokenized_path.exists():
        raise SystemExit(f"[error] tokenized_path not found: {tokenized_path}")

    llamafactory_root = _find_llamafactory(args.llamafactory_root)
    if llamafactory_root is None and not args.dry:
        raise SystemExit(
            "[error] LlamaFactory not found. Pass --llamafactory_root or install "
            "llamafactory in the current environment.")
    if args.l2_lambda > 0 and llamafactory_root is not None and not args.dry:
        _patch_llamafactory_l2(llamafactory_root)
    if args.top1_miss_loss and llamafactory_root is not None and not args.dry:
        _patch_llamafactory_top1_miss_loss(llamafactory_root)

    train_config: Dict = {
        "model": args.model,
        "tokenized_path": str(tokenized_path),
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
        "l2_base_model": args.l2_base_model or args.model,
        "top1_miss_loss": bool(args.top1_miss_loss),
        "weight_decay": args.weight_decay,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "pure_bf16": bool(args.pure_bf16),
    }
    if not args.force and _model_current(
            output_dir, train_meta, args.method_version, train_config):
        print(f"[skip] current LLaMA Factory full model exists: {output_dir}")
        return

    cfg: Dict = {
        "model_name_or_path": args.model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset": "dummy_tokenized",
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "tokenized_path": str(tokenized_path),
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": "none",
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "plot_loss": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": bool(args.bf16),
        "ddp_timeout": 180000000,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "disable_gradient_checkpointing": not bool(args.gradient_checkpointing),
    }
    if args.pure_bf16:
        cfg["pure_bf16"] = True
    if args.deepspeed:
        cfg["deepspeed"] = str(_repo_path(args.deepspeed))
    if args.l2_lambda > 0:
        cfg["use_l2_reg"] = True
        cfg["l2_lambda"] = args.l2_lambda
        cfg["l2_base_model"] = args.l2_base_model or args.model
    if args.top1_miss_loss:
        cfg["use_top1_miss_loss"] = True

    yaml_path = output_dir.parent / f"{output_dir.name}_lf_train.yaml"
    _write_yaml(yaml_path, cfg)

    cmd = [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)]
    print("\n$", " ".join(shlex.quote(x) for x in cmd), flush=True)
    if args.dry:
        return
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
        "llamafactory_root": str(llamafactory_root) if llamafactory_root else None,
        "yaml": str(yaml_path),
        "train_config": train_config,
    }
    train_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] full model -> {output_dir}")


if __name__ == "__main__":
    main()
