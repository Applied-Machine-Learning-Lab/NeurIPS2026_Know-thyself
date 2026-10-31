from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.baselines.pipeline.common import METHOD_SCORE_VERSIONS, save_result
from src.baselines.pipeline.labeling import (label_answers_by_gold_containment,
                                             load_greedy, write_greedy)
from src.baselines.utils.prompt import QA_PROMPT_VERSION


def _load_existing_greedy(output_dir: Path, questions: List[str], args):
    path = output_dir / "greedy.jsonl"
    loaded = load_greedy(
        path, questions,
        getattr(args, "result_context", {}).get("eval_model_path"))
    if loaded is None and path.exists():
        print(f"[warn] ignoring {path}: greedy labels are stale")
    return loaded


def _build_vllm_scorers(runner, args) -> Dict:
    from src.baselines import (ConsisLexScorer, ConsisSemScorer, NProbScorer,
                                ProbScorer, SemanticEntropyScorer,
                                Verbalized10Scorer, VerbalizedScorer)
    mnt = args.max_new_tokens
    return {
        "verbalized": lambda: VerbalizedScorer(runner, max_new_tokens=mnt),
        "verbal_10":  lambda: Verbalized10Scorer(runner, max_new_tokens=mnt),
        "prob":       lambda: ProbScorer(runner, max_new_tokens=mnt),
        "n_prob":     lambda: NProbScorer(runner, max_new_tokens=mnt),
        "consis_lex": lambda: ConsisLexScorer(
            runner, num_samples=args.se_num_samples,
            temperature=args.se_temperature, max_new_tokens=mnt),
        "consis_sem": lambda: ConsisSemScorer(
            runner, num_samples=args.se_num_samples,
            temperature=args.se_temperature,
            nli_model_name=args.nli_model, max_new_tokens=mnt,
            nli_nprocs=args.nprocs),
        "semantic_entropy": lambda: SemanticEntropyScorer(
            runner, num_samples=args.se_num_samples,
            temperature=args.se_temperature, variant=args.se_variant,
            nli_model_name=args.nli_model, max_new_tokens=mnt,
            nli_nprocs=args.nprocs),
    }


def run_vllm_phase(args, questions: List[str], samples_meta: list,
                    gold: List, output_dir: Path,
                    vllm_baselines: List[str],
                    ) -> Tuple[List[str], List[int], Dict[str, dict]]:
    print(f"\n=== vLLM phase: {vllm_baselines} ===")
    from src.baselines.utils import VLLMRunner, build_qa_prompts

    runner = VLLMRunner(model_path=args.model, nprocs=args.nprocs,
                         lora_path=args.model_adapter,
                         dtype=args.dtype,
                         gpu_memory_utilization=(
                             args.vllm_gpu_memory_utilization))

    greedy_answers: List[str] = []
    labels: List[int] = []
    existing = _load_existing_greedy(output_dir, questions, args)
    labels_current = False
    if existing:
        greedy_answers, labels = existing
        labels_current = True
        print(f"[info] greedy EM on eval split: "
              f"{sum(labels)/len(labels):.3f} "
              f"({sum(labels)}/{len(labels)})")

    summary: Dict[str, dict] = {}
    pending_scores = {}
    if vllm_baselines:
        builders = _build_vllm_scorers(runner, args)
        for name in vllm_baselines:
            print(f"\n--- {name} ---")
            scorer = builders[name]()
            scores = scorer.score(questions, greedy_answers=None)
            pending_scores[name] = scores

    if not labels_current:
        print("[info] running shared greedy decode for answer labels...")
        prompts = build_qa_prompts(runner.tokenizer, questions)
        greedy_answers = runner.generate_greedy(
            prompts, max_new_tokens=args.max_new_tokens,
            strip_newline=True)
        labels = []

    runner.shutdown()
    print("[info] main vLLM shut down, GPUs freed.")

    if not labels_current:
        labels, labeled = label_answers_by_gold_containment(
            args, questions, gold, greedy_answers)
        write_greedy(output_dir / "greedy.jsonl", samples_meta, questions,
                     greedy_answers, labels, labeled,
                     extra_fields=getattr(args, "result_context", {}))
        print(f"[info] containment answer Acc: "
              f"{sum(labels)/len(labels):.3f} "
              f"({sum(labels)}/{len(labels)})")

    for name, scores in pending_scores.items():
        context = dict(getattr(args, "result_context", {}))
        context["score_version"] = METHOD_SCORE_VERSIONS.get(name)
        summary[name] = save_result(
            output_dir, name, samples_meta, greedy_answers, scores, labels,
            extra_fields=context)
    return greedy_answers, labels, summary
