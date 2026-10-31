from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines.utils.prompt import QA_PROMPT_VERSION, build_qa_prompts
from src.baselines.utils.vllm_runner import VLLMRunner
from src.data.evaluator import compute_f1, exact_match, is_correct
from src.training.fok_data import (
    FOK_SCORE_VERSION,
    LF_FULL_KNOWLEDGE_FOK_LORA_ARCH_VERSION,
    LF_KNOWLEDGE_FULL_ARCH_VERSION,
    LF_KNOWLEDGE_LORA_ARCH_VERSION,
    build_fok_prompt,
)


YES_CHOICES = ["Yes", " Yes", "yes", " yes", "YES"]
NO_CHOICES = ["No", " No", "no", " no", "NO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--knowledge_lora", default=None)
    parser.add_argument("--fok_lora", default=None)
    parser.add_argument("--adapter_arch", default=None)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return parser.parse_args()


def _read_queries(path: Path, split: str) -> List[dict]:
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        if split == "all" or row.get("split", "train") == split:
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    queries = _read_queries(Path(args.queries_jsonl), args.split)
    print(
        f"[info] {len(queries)} queries split={args.split} "
        f"backend=vllm nprocs={args.nprocs} "
        f"knowledge_lora={args.knowledge_lora} fok_lora={args.fok_lora}",
        flush=True,
    )
    if args.knowledge_lora and args.fok_lora:
        raise SystemExit(
            "[error] vLLM inference supports at most one LoRA")

    runner = VLLMRunner(
        model_path=args.base_model,
        nprocs=args.nprocs,
        lora_path=args.knowledge_lora or args.fok_lora,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        prompts = build_qa_prompts(
            runner.tokenizer, [row["question"] for row in queries])
        responses = runner.generate_greedy(
            prompts,
            max_new_tokens=args.max_new_tokens,
            strip_newline=False,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
        )
        fok_prompts = [
            build_fok_prompt(runner.tokenizer, row["question"])
            for row in queries
        ]
        fok_probs = runner.first_token_group_softmax(
            fok_prompts,
            [NO_CHOICES, YES_CHOICES],
        )
    finally:
        runner.shutdown()

    rows = []
    adapter_arch = args.adapter_arch
    if adapter_arch is None:
        if args.knowledge_lora:
            adapter_arch = LF_KNOWLEDGE_LORA_ARCH_VERSION
        elif args.fok_lora:
            adapter_arch = LF_FULL_KNOWLEDGE_FOK_LORA_ARCH_VERSION
        else:
            adapter_arch = LF_KNOWLEDGE_FULL_ARCH_VERSION
    for query, response, fok_prob in zip(queries, responses, fok_probs):
        response = response.strip()
        j_score = float(fok_prob[1])
        rows.append({
            "qid": query["id"],
            "question": query["question"],
            "answers": query["answers"],
            "primary_answer": query.get("primary_answer"),
            "split": query.get("split", "train"),
            "response": response,
            "answer": response,
            "j_score": j_score,
            "fok_score": j_score,
            "fok_score_version": FOK_SCORE_VERSION,
            "abstain": bool(j_score < 0.5),
            "em": int(exact_match(response, query["answers"])),
            "f1": compute_f1(response, query["answers"]),
            "is_correct": int(is_correct(response, query["answers"])),
            "answer_prompt_version": QA_PROMPT_VERSION,
            "adapter_arch": adapter_arch,
            "knowledge_model": args.base_model if not args.knowledge_lora else None,
            "knowledge_lora": args.knowledge_lora,
            "fok_lora": args.fok_lora,
        })

    output = Path(args.output)
    _write_jsonl(output, rows)
    n_em = sum(int(row["em"]) for row in rows)
    mean_f1 = sum(float(row["f1"]) for row in rows) / max(len(rows), 1)
    print(
        f"[done] n={len(rows)} local_em={n_em / max(len(rows), 1):.3f} "
        f"({n_em}/{len(rows)}) local_f1={mean_f1:.3f} -> {output}")


if __name__ == "__main__":
    main()
