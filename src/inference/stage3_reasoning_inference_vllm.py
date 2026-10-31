from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

from jinja2 import Template

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines.utils.vllm_runner import VLLMRunner
from src.baselines.utils.answer_containment import (
    ANSWER_CONTAINMENT_LABEL_VERSION,
    contains_answer,
)
from src.data.evaluator import compute_f1, exact_match


STAGE3_REASONING_PROMPT_VERSION = "fgpo_judgment_answer_think_prompt_v1"
STAGE3_JUDGE_SCORE_VERSION = "generated_judge_yes_prob_v1"

FORMAT_PROMPT = "judgment_answer_think.jinja"
JUDGE_TAG = "answer"
ANSWER_TAG = "response"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format_prompt", default=None)
    parser.add_argument("--adapter_arch", default="easy_r1_stage3_full_model_v1")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument(
        "--enable_thinking",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Pass enable_thinking to apply_chat_template for Qwen3 models. "
            "'auto' enables it when the tokenizer's model path contains 'qwen3'."
        ),
    )
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


def _default_format_prompt() -> Path:
    return _REPO_ROOT / "easyr1_patches" / "format_prompt" / FORMAT_PROMPT


def _build_stage3_prompt(
    tokenizer,
    template: Template,
    question: str,
    enable_thinking: bool | None = None,
) -> str:
    content = template.render(content=question)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        **kwargs,
    )


def _parse_stage3_response(text: str) -> dict:
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    judge_match = re.search(
        rf"<{JUDGE_TAG}>\s*(yes|no)\s*</{JUDGE_TAG}>", text,
        re.DOTALL | re.IGNORECASE)
    answer_match = re.search(rf"<{ANSWER_TAG}>(.*?)</{ANSWER_TAG}>", text, re.DOTALL)
    judge = judge_match.group(1).strip().lower() if judge_match else None
    answer = answer_match.group(1).strip() if answer_match else text.strip()
    think = think_match.group(1).strip() if think_match else ""
    format_ok = bool(think_match and judge_match and answer_match)
    return {
        "think": think,
        "judge": judge,
        "answer": answer,
        "format_ok": format_ok,
    }


def _is_abstain(answer: str, judge: str | None) -> bool:
    norm = re.sub(r"[^\w\s]", "", answer.lower()).strip()
    norm = " ".join(norm.split())
    return judge == "no" or norm in {
        "abstain",
        "i dont know",
        "idk",
        "unknown",
        "i cannot answer",
        "cannot answer",
    }


def main() -> None:
    args = parse_args()
    queries = _read_queries(Path(args.queries_jsonl), args.split)
    prompt_path = Path(args.format_prompt) if args.format_prompt else \
        _default_format_prompt()
    if not prompt_path.exists():
        raise SystemExit(f"[error] missing Stage3 format prompt: {prompt_path}")
    template = Template(prompt_path.read_text(encoding="utf-8").strip())

    if args.enable_thinking == "auto":
        thinking_flag = "qwen3" in args.base_model.lower()
    else:
        thinking_flag = args.enable_thinking == "true"

    print(
        f"[info] {len(queries)} queries split={args.split} "
        f"backend=vllm nprocs={args.nprocs} "
        f"stage3_prompt={prompt_path} enable_thinking={thinking_flag}",
        flush=True,
    )
    runner = VLLMRunner(
        model_path=args.base_model,
        nprocs=args.nprocs,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        prompts = [
            _build_stage3_prompt(
                runner.tokenizer, template, row["question"],
                enable_thinking=thinking_flag,
            )
            for row in queries
        ]
        responses = runner.generate_greedy(
            prompts,
            max_new_tokens=args.max_new_tokens,
            strip_newline=False,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
        )
        judge_prompts: List[str] = []
        parsed_list = [_parse_stage3_response(r.strip())
                       for r in responses]
        for base_prompt, parsed in zip(prompts, parsed_list):
            think_body = parsed["think"] if parsed["format_ok"] else ""
            judge_prompts.append(
                f"{base_prompt}<think>{think_body}</think>\n<{JUDGE_TAG}>"
            )
        yes_probs = runner.first_token_softmax(judge_prompts, choices=["yes", "no"])
    finally:
        runner.shutdown()

    rows = []
    for query, raw_response, yes_no_probs in zip(queries, responses, yes_probs):
        parsed = _parse_stage3_response(raw_response.strip())
        answer = parsed["answer"]
        judge = parsed["judge"]
        abstain = _is_abstain(answer, judge)
        judged_yes = judge == "yes"
        if isinstance(yes_no_probs, (list, tuple)) and len(yes_no_probs) == 2:
            p_yes = float(yes_no_probs[0])
            if not (0.0 <= p_yes <= 1.0) or p_yes != p_yes:
                p_yes = 1.0 if judged_yes else (0.0 if judge == "no" else 0.5)
        else:
            p_yes = 1.0 if judged_yes else (0.0 if judge == "no" else 0.5)
        j_score = 1.0 if judged_yes else 0.0 if judge == "no" else 0.5
        fok_score = p_yes
        answer_for_eval = "" if abstain else answer
        answer_correct = int(contains_answer(answer_for_eval, query["answers"]))
        rows.append({
            "qid": query["id"],
            "question": query["question"],
            "answers": query["answers"],
            "primary_answer": query.get("primary_answer"),
            "split": query.get("split", "train"),
            "response": answer_for_eval,
            "answer": answer_for_eval,
            "raw_response": raw_response.strip(),
            "stage3_judge": judge,
            "stage3_answer": answer,
            "stage3_format_ok": bool(parsed["format_ok"]),
            "j_score": j_score,
            "fok_score": fok_score,
            "fok_score_version": STAGE3_JUDGE_SCORE_VERSION,
            "abstain": abstain,
            "em": int(exact_match(answer_for_eval, query["answers"])),
            "f1": compute_f1(answer_for_eval, query["answers"]),
            "is_correct": answer_correct,
            "correctness_label_version": ANSWER_CONTAINMENT_LABEL_VERSION,
            "answer_prompt_version": STAGE3_REASONING_PROMPT_VERSION,
            "adapter_arch": args.adapter_arch,
            "knowledge_model": args.base_model,
            "knowledge_lora": None,
            "fok_lora": None,
        })

    output = Path(args.output)
    _write_jsonl(output, rows)
    n = max(len(rows), 1)
    n_correct = sum(int(row["is_correct"]) for row in rows)
    n_yes = sum(row["stage3_judge"] == "yes" for row in rows)
    n_no = sum(row["stage3_judge"] == "no" for row in rows)
    n_fmt = sum(bool(row["stage3_format_ok"]) for row in rows)
    fok_scores = [row["fok_score"] for row in rows]
    fok_mean = sum(fok_scores) / n
    fok_unique = len(set(round(s, 6) for s in fok_scores))
    print(
        f"[done] n={len(rows)} correct={n_correct / n:.3f} "
        f"judge_yes={n_yes / n:.3f} judge_no={n_no / n:.3f} "
        f"format_ok={n_fmt / n:.3f} fok_mean={fok_mean:.3f} "
        f"fok_unique={fok_unique} -> {output}")


if __name__ == "__main__":
    main()
