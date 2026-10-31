from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.baselines.pipeline.common import METHOD_SCORE_VERSIONS, save_result
from src.baselines.pipeline.labeling import (
    label_answers_by_gold_containment,
    write_greedy,
)
from src.baselines.utils.prompt import build_qa_prompt


def load_model_auto(model_path, torch_dtype):
    from transformers import AutoModelForCausalLM
    adapter_dir = Path(model_path)
    if (adapter_dir / "adapter_config.json").exists():
        parent_file = adapter_dir / "parent_model.txt"
        if not parent_file.exists():
            raise ValueError(
                f"Found adapter_config.json but no parent_model.txt in "
                f"{model_path}. Cannot determine parent model for adapter "
                "loading.")
        from peft import PeftModel
        model = load_model_auto(parent_file.read_text().strip(), torch_dtype)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        return model.merge_and_unload()
    return AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch_dtype, trust_remote_code=True)


def _greedy_worker(rank, device, questions, indices, extras):
    import torch
    from transformers import AutoTokenizer
    model_path = extras["model_path"]
    max_new = extras["max_new_tokens"]
    bs = extras["batch_size"]
    dtype = getattr(torch, extras["dtype"])
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_model_auto(model_path, torch_dtype=dtype).to(device).eval()
    out_list = []
    with torch.no_grad():
        for i in range(0, len(questions), bs):
            batch = questions[i:i + bs]
            prompts = [build_qa_prompt(tok, q) for q in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new,
                                  do_sample=False,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = gen[:, enc.input_ids.shape[1]:]
            for g in new:
                t = tok.decode(g, skip_special_tokens=True)
                out_list.append(t.split("\n")[0].strip())
    return out_list


def _build_elical_scorer(args):
    from src.baselines import EliCalScorer
    return EliCalScorer(
        model_path=args.model,
        score_base_model=args.elical_score_base_model,
        pre_adapter_path=args.elical_pre_adapter,
        elical_repo=args.elical_repo,
        elical_preset=args.elical_preset,
        elical_lora_path=args.elical_lora_path,
        vector_head_path=args.elical_vector_head_path,
        nprocs=args.nprocs, dtype=args.dtype,
        batch_size=args.elical_batch_size,
        max_length=args.elical_max_length,
        max_new_tokens=args.max_new_tokens,
    )


def run_hfdp_phase(args, questions: List[str], samples_meta: list,
                    gold: List, output_dir: Path,
                    greedy_answers: List[str], labels: List[int],
                    hfdp_baselines: List[str]) -> Dict[str, dict]:
    print(f"\n=== HF DP phase: {hfdp_baselines} ===")

    if not greedy_answers and not (
            hfdp_baselines == ["elical"] and
            args.elical_answer_mode == "combined"):
        from src.baselines.utils import run_hf_dp
        print("[info] greedy decode (HF DP)...")
        extras = dict(model_path=args.model,
                      max_new_tokens=args.max_new_tokens,
                      batch_size=args.batch_size, dtype=args.dtype)
        greedy_answers = run_hf_dp(_greedy_worker, questions,
                                    nprocs=args.nprocs, extras=extras)
        labels, labeled = label_answers_by_gold_containment(
            args, questions, gold, greedy_answers)
        write_greedy(output_dir / "greedy.jsonl", samples_meta, questions,
                     greedy_answers, labels, labeled,
                     extra_fields=getattr(args, "result_context", {}))
        print(f"[info] containment answer Acc: "
              f"{sum(labels)/len(labels):.3f}")

    summary: Dict[str, dict] = {}
    for name in hfdp_baselines:
        print(f"\n--- {name} ---")
        if name == "elical":
            scorer = _build_elical_scorer(args)
        else:
            raise ValueError(f"unknown HF DP baseline: {name}")

        method_greedy = greedy_answers
        method_labels = labels
        if name == "elical" and args.elical_answer_mode == "combined":
            print("[info] EliCal combined answer decode...")
            method_greedy = scorer.generate_answers(questions)
            method_labels, labeled = label_answers_by_gold_containment(
                args, questions, gold, method_greedy)
            write_greedy(output_dir / "elical_greedy.jsonl", samples_meta,
                         questions, method_greedy, method_labels, labeled,
                         extra_fields=getattr(args, "result_context", {}))
            print(f"[info] EliCal combined containment answer Acc: "
                  f"{sum(method_labels)/len(method_labels):.3f}")

        scores = scorer.score(questions, greedy_answers=method_greedy)
        extra = {}
        if name == "elical":
            extra["elical_answer_mode"] = args.elical_answer_mode
        context = dict(getattr(args, "result_context", {}))
        context.update(extra)
        context["score_version"] = METHOD_SCORE_VERSIONS.get(name)
        summary[name] = save_result(output_dir, name, samples_meta,
                                     method_greedy, scores, method_labels,
                                     extra_fields=context)
    return summary
