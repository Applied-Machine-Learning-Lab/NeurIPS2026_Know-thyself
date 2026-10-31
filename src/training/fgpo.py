from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import string
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


_ABILITY_KEYS = ("ability", "label", "y", "is_correct", "correct",
                 "answerable", "can_answer")


def _ability(q: dict):
    for key in _ABILITY_KEYS:
        if key not in q or q[key] is None:
            continue
        v = q[key]
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("can", "cannot"):
                return v
            v = v in ("1", "true", "yes")
        return "can" if bool(v) else "cannot"
    return None


def _row(q: dict) -> dict:
    answers = q.get("answers", []) or []
    primary = q.get("primary_answer") or (answers[0] if answers else "")
    norm = [_normalize(a) for a in answers if a]
    gt = {
        "answers": answers,
        "normalized_answers": norm,
        "primary_answer": primary,
        "question": q["question"],
    }
    ability = _ability(q)
    if ability is not None:
        gt["ability"] = ability
    gt = json.dumps(gt, ensure_ascii=False)
    return {"prompt": q["question"], "ground_truth": gt,
            "qid": q.get("id", "")}


def cmd_prepare(args):
    import pandas as pd

    rows = [json.loads(l) for l in open(args.queries_jsonl)]
    train = [r for r in rows if r.get("split", "train") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    if not val:
        n = max(1, len(rows) // 10)
        val = rows[-n:]; train = rows[:-n]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_row(r) for r in train]).to_parquet(
        out / "train.parquet", index=False)
    pd.DataFrame([_row(r) for r in val]).to_parquet(
        out / "val.parquet", index=False)
    n_missing = sum(_ability(r) is None for r in train)
    print(f"[prep] train={len(train)} val={len(val)} -> {out}")
    if n_missing:
        print(f"[warn] {n_missing}/{len(train)} train rows have no answerability "
              f"label (one of {', '.join(_ABILITY_KEYS)}); the "
              f"FGPO reward needs it")


def _rel(path: Path, cwd: Path) -> str:
    return os.path.relpath(path, cwd)


def _model_arg(model: str, cwd: Path) -> str:
    p = Path(model)
    if p.is_absolute() or p.exists() or model.startswith("."):
        return _rel(p.resolve(), cwd)
    return model


def _has_easyr1_source(path: Path) -> bool:
    return (path / "verl" / "trainer" / "main.py").exists()


def _resolve_easyr1_root(path: str) -> Path:
    requested = Path(path).resolve()
    if _has_easyr1_source(requested):
        return requested

    candidates: list[Path] = []
    for base in [Path.cwd().resolve(), _REPO_ROOT, *_REPO_ROOT.parents]:
        candidates.extend([
            base / "EasyR1",
            base / "projects" / "EasyR1",
            base / "Experiments" / "EasyR1",
        ])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _has_easyr1_source(resolved):
            print(f"[warn] EasyR1 root {requested} has no verl source; "
                  f"using {resolved}")
            return resolved

    return requested


REWARD = "judgment_answer_think"


def _ensure_easyr1_patches(er: Path) -> None:
    patch_root = _REPO_ROOT / "easyr1_patches"
    copies = [
        (patch_root / "format_prompt" / f"{REWARD}.jinja",
         er / "examples" / "format_prompt" / f"{REWARD}.jinja"),
        (patch_root / "reward_function" / f"{REWARD}.py",
         er / "examples" / "reward_function" / f"{REWARD}.py"),
        (patch_root / "config.yaml",
         er / "examples" / "config.yaml"),
    ]
    for src, dst in copies:
        if not src.exists():
            sys.exit(f"[error] missing EasyR1 patch source: {src}")
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[patch] copied {src} -> {dst}")


def _parse_env_overrides(items: list[str]) -> dict[str, str]:
    env = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"[error] --reward_env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            sys.exit(f"[error] empty env key in --reward_env item: {item}")
        env[key] = value
    return env


def cmd_launch(args):
    if args.rollout_n < 2:
        sys.exit("[error] GRPO requires --rollout_n >= 2")
    er = _resolve_easyr1_root(args.easyr1_root)
    _ensure_easyr1_patches(er)
    fmt = er / "examples" / "format_prompt" / f"{REWARD}.jinja"
    rew = er / "examples" / "reward_function" / f"{REWARD}.py"
    cfg = er / "examples" / "config.yaml"
    if not fmt.exists() or not rew.exists() or not cfg.exists():
        sys.exit(f"[error] EasyR1 root {er} missing one of "
                 f"{REWARD}.jinja / {REWARD}.py / config.yaml")
    out = Path(args.output_dir).resolve()
    data = Path(args.data_dir).resolve()
    import pandas as pd
    gts = pd.read_parquet(data / "train.parquet")["ground_truth"]
    if not any('"ability"' in g for g in gts):
        sys.exit("[error] the FGPO reward needs an answerability "
                 "label per train query; see `prepare`")

    save_path = Path(args.save_checkpoint_path).resolve() \
        if args.save_checkpoint_path else out / "checkpoints"
    overrides: List[str] = [
        f"data.train_files={_rel(data / 'train.parquet', er)}@train",
        f"data.val_files={_rel(data / 'val.parquet', er)}@train",
        "data.prompt_key=prompt",
        "data.answer_key=ground_truth",
        f"data.format_prompt={_rel(fmt, er)}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"worker.actor.model.model_path={_model_arg(args.model, er)}",
        f"worker.actor.model.lora.rank={args.lora_rank}",
        "worker.actor.padding_free=false",
        "worker.actor.dynamic_batching=false",
        f"worker.reward.reward_function={_rel(rew, er)}:compute_score",
        f"worker.rollout.n={args.rollout_n}",
        f"worker.rollout.tensor_parallel_size={args.rollout_tensor_parallel_size}",
        f"worker.rollout.gpu_memory_utilization={args.rollout_gpu_memory_utilization}",
        "worker.ref.padding_free=false",
        "worker.ref.dynamic_batching=false",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        f"trainer.save_checkpoint_path={_rel(save_path, er)}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.val_freq={args.val_freq}",
        f"trainer.val_before_train={str(args.val_before_train).lower()}",
        f"trainer.save_model_only={str(args.save_model_only).lower()}",
        "trainer.find_last_checkpoint=false",
        "trainer.logger=[file]",
        f"trainer.total_epochs={args.epochs}",
        f"data.rollout_batch_size={args.rollout_batch_size}",
        f"worker.actor.global_batch_size={args.global_batch_size}",
        f"worker.actor.optim.lr={args.lr}",
    ]
    if args.seed is not None:
        overrides += [
            f"data.seed={args.seed}",
            f"worker.rollout.seed={args.seed}",
        ]

    cmd = [
        sys.executable, "-m", "verl.trainer.main",
        f"config={_rel(cfg, er)}",
        *overrides,
    ]
    reward_env = _parse_env_overrides(args.reward_env)
    env_prefix = [f"{k}={v}" for k, v in sorted(reward_env.items())]
    line = " \\\n    ".join(shlex.quote(x) for x in cmd)
    print("# Run from", _rel(er, Path.cwd()))
    print("cd", _rel(er, Path.cwd()))
    if env_prefix:
        print(" ".join(shlex.quote(x) for x in env_prefix), line)
    else:
        print(line)
    if args.execute:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(er) + os.pathsep + env.get("PYTHONPATH", "")
        env.update(reward_env)
        proc = subprocess.run(cmd, cwd=str(er), env=env)
        if proc.returncode != 0:
            raise SystemExit(f"[error] EasyR1 failed with code {proc.returncode}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--queries_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("launch")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model", required=True,
                    help="merged Stage 2 model (or HF id) used as FGPO init")
    p.add_argument("--easyr1_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--experiment_name", default="fgpo")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--rollout_n", type=int, default=8)
    p.add_argument("--rollout_batch_size", type=int, default=256)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_response_length", type=int, default=512)
    p.add_argument("--n_gpus_per_node", type=int, default=8)
    p.add_argument("--rollout_tensor_parallel_size", type=int, default=1)
    p.add_argument("--rollout_gpu_memory_utilization", type=float, default=0.60)
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--seed", type=int, default=None,
                   help="data shuffle + vLLM rollout seed (EasyR1 default 1)")
    p.add_argument("--reward_env", nargs="*", default=[],
                   help="reward-function environment overrides as KEY=VALUE")
    p.add_argument("--save_checkpoint_path", default=None)
    p.add_argument("--save_freq", type=int, default=-1)
    p.add_argument("--val_freq", type=int, default=-1)
    p.add_argument("--val_before_train", action="store_true")
    p.add_argument("--save_model_only", action="store_true", default=True)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=cmd_launch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
