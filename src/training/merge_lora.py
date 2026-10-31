from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _dtype(name: str):
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if (out / "config.json").exists() and not args.force:
        print(f"[skip] merged model exists: {out}")
        return
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=_dtype(args.dtype),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(out))
    (out / "merged_from.txt").write_text(
        f"base_model={args.base_model}\nadapter={args.adapter}\n")
    print(f"[done] merged model -> {out}")


if __name__ == "__main__":
    main()
