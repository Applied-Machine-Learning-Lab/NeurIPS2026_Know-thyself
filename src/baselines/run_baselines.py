import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.baselines.pipeline import (ALL_METHODS, HF_DP_METHODS,
                                     METHOD_SCORE_VERSIONS, VLLM_METHODS,
                                     run_hfdp_phase, run_vllm_phase)
from src.baselines.pipeline.labeling import load_greedy
from src.baselines.utils.answer_containment import (
    ANSWER_CONTAINMENT_LABEL_VERSION,
)
from src.data.dataset_builder import load_from_jsonl
from src.data.data_loader import SUPPORTED_DATASETS, load_dataset_by_name
from src.baselines.utils.prompt import QA_PROMPT_VERSION


def _load_summary(output_dir: Path) -> Dict[str, dict]:
    path = output_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _existing_result(output_dir: Path, name: str,
                     expected_qids: list[str] | None = None,
                     args=None) -> dict | None:
    metrics_path = output_dir / f"{name}_metrics.json"
    scores_path = output_dir / f"{name}_scores.jsonl"
    if not metrics_path.exists() or not scores_path.exists():
        return None
    if expected_qids is not None:
        try:
            rows = [json.loads(line)
                    for line in scores_path.read_text().splitlines()
                    if line.strip()]
        except json.JSONDecodeError as exc:
            print(f"[warn] recomputing {name}: {scores_path} invalid JSON "
                  f"cache ({exc})")
            return None
        found = [str(row.get("qid")) for row in rows]
        if found != expected_qids:
            print(f"[warn] recomputing {name}: {scores_path} qids do not "
                  "match current samples")
            return None
        if not all(row.get("answer_prompt_version") == QA_PROMPT_VERSION
                   for row in rows):
            print(f"[warn] recomputing {name}: {scores_path} answer prompt "
                  "version differs")
            return None
        if not all(row.get("label_version") == ANSWER_CONTAINMENT_LABEL_VERSION
                   for row in rows):
            print(f"[warn] recomputing {name}: {scores_path} label version "
                  "differs")
            return None
        expected_model = getattr(args, "result_context", {}).get(
            "eval_model_path") if args is not None else None
        if expected_model is not None and not all(
                row.get("eval_model_path") == expected_model for row in rows):
            print(f"[warn] recomputing {name}: {scores_path} eval model "
                  "differs")
            return None
        expected_score_version = METHOD_SCORE_VERSIONS.get(name)
        if expected_score_version is not None and not all(
                row.get("score_version") == expected_score_version
                for row in rows):
            print(f"[warn] recomputing {name}: {scores_path} score version "
                  "differs")
            return None
        if name == "elical" and args is not None:
            found_modes = {row.get("elical_answer_mode") for row in rows}
            if found_modes != {args.elical_answer_mode}:
                print(f"[warn] recomputing {name}: {scores_path} EliCal "
                      "answer mode differs")
                return None
    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"[skip] {name} — existing {metrics_path} and {scores_path}")
    return {
        "auroc": metrics.get("auroc", float("nan")),
        "accuracy": metrics.get("accuracy", float("nan")),
    }


def _model_context(path: str) -> str:
    p = Path(path).expanduser()
    if p.exists():
        return str(p.resolve())
    return path


def _eval_model_context(model_path: str, adapter_path: str | None) -> str:
    context = _model_context(model_path)
    if adapter_path:
        context += f"::adapter={_model_context(adapter_path)}"
    return context


def _load_existing_greedy(output_dir: Path, questions: list[str], args):
    loaded = load_greedy(
        output_dir / "greedy.jsonl", questions,
        args.result_context.get("eval_model_path"))
    return loaded if loaded is not None else ([], [])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="main model (eval target), e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model_adapter", default=None,
                    help="optional PEFT LoRA adapter loaded on top of --model "
                         "for vLLM-backed eval-target baselines")
    ap.add_argument("--dataset", default="triviaqa", choices=SUPPORTED_DATASETS)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--input_jsonl", default=None,
                    help="optional explicit eval samples; overrides dataset/split")
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--baselines", nargs="+", required=True,
                    choices=ALL_METHODS)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--nprocs", type=int, default=8,
                    help="GPUs for main model DP (one replica per GPU)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="EliCal long_qa default = 256")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="per-GPU batch for HF DP phase")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--se_num_samples", type=int, default=10)
    ap.add_argument("--se_temperature", type=float, default=1.0)
    ap.add_argument("--se_variant", choices=["discrete", "se"],
                    default="discrete")
    ap.add_argument("--nli_model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--elical_repo",
                    default="Shiyunee/Honest-Qwen2.5-7B-Instruct")
    ap.add_argument("--elical_preset", choices=["1k", "560k"], default="1k")
    ap.add_argument("--elical_lora_path", default=None)
    ap.add_argument("--elical_vector_head_path", default=None)
    ap.add_argument("--elical_score_base_model", default=None,
                    help="base model used to host EliCal's AutoModel adapter")
    ap.add_argument("--elical_pre_adapter", default=None,
                    help="optional knowledge PEFT adapter merged before EliCal")
    ap.add_argument("--elical_answer_mode", choices=["target", "combined"],
                    default="combined",
                    help="target uses --model for answers; combined also "
                         "loads EliCal adapter for answer generation")
    ap.add_argument("--elical_batch_size", type=int, default=4)
    ap.add_argument("--elical_max_length", type=int, default=1024)
    return ap.parse_args()


def main():
    args = parse_args()
    args.result_context = {
        "eval_model_path": _eval_model_context(args.model, args.model_adapter)
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl:
        print(f"[info] loading explicit samples: {args.input_jsonl}")
        samples_meta = load_from_jsonl(args.input_jsonl)
        if args.num_samples:
            samples_meta = samples_meta[:args.num_samples]
    else:
        print(f"[info] loading {args.dataset}/{args.split} ({args.num_samples})")
        samples_meta = load_dataset_by_name(
            args.dataset, split=args.split, num_samples=args.num_samples)
    questions = [s["question"] for s in samples_meta]
    gold = [s["answers"] for s in samples_meta]
    expected_qids = [str(s.get("id", i)) for i, s in enumerate(samples_meta)]

    requested = list(args.baselines)
    summary: Dict[str, dict] = _load_summary(output_dir)
    remaining = []
    for name in requested:
        existing = _existing_result(output_dir, name, expected_qids, args)
        if existing is None:
            remaining.append(name)
        else:
            summary[name] = existing

    if not remaining:
        print("[info] all requested baselines already exist; nothing to run")

    vllm_baselines = [b for b in remaining if b in VLLM_METHODS]
    hfdp_baselines = [b for b in remaining if b in HF_DP_METHODS]

    greedy_answers, labels = _load_existing_greedy(output_dir, questions, args)

    if vllm_baselines:
        greedy_answers, labels, s1 = run_vllm_phase(
            args, questions, samples_meta, gold, output_dir,
            vllm_baselines)
        summary.update(s1)

    if hfdp_baselines:
        s2 = run_hfdp_phase(
            args, questions, samples_meta, gold, output_dir,
            greedy_answers, labels, hfdp_baselines)
        summary.update(s2)

    print("\n=== SUMMARY ===")
    for name in requested:
        m = summary.get(name)
        if m is not None:
            print(f"{name:>18} : AUROC={m['auroc']:.4f}  "
                  f"Acc={m['accuracy']:.3f}")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
