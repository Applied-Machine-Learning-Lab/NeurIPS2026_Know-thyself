from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import load_dataset_by_name


OP_SSFT_DATA_VERSION = "op_ssft_repair_label_policy_v15"
OP_SSFT_LABEL_POLICIES = (
    "corrected_suffix_eos",
    "corrected_suffix",
    "answer_eos",
    "answer_only",
    "repaired_response_eos",
)


def build_raw_question_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


REPAIR_JUDGE_TEMPLATE = """You are constructing supervised fine-tuning data.

Given a question, gold answers, and a model response, decide whether the
response answers the question correctly. Then choose the exact response
substring where supervised correction should begin, and write the shortest
corrected suffix that can replace the response from that substring onward.

Rules:
- First identify the entity or value in the response that answers the question.
- Judge correctness by entity equivalence, not exact string match. Aliases,
  real names, stage names, pen names, abbreviations, or alternative spellings
  should be treated as correct if they unambiguously refer to the same entity.
  If a gold answer is a stage name or pen name and the response gives the real
  person behind that name, mark the response correct.
  Example: if the gold answer is "David Seville" and the response answers
  "Ross Bagdasarian Sr.", mark it correct because David Seville is Ross
  Bagdasarian Sr.'s stage name.
- If the response is correct, set "is_correct" to true and set "span" to the
  earliest exact response substring that answers the question, even if that
  substring is not literally in the gold answer list. Set "replacement" to the
  same text as "span". Set "corrected_suffix" to a short complete suffix that
  starts with that answer substring and can be placed after the original prefix.
  If the answer phrase cannot be isolated, output span as "NONE", replacement
  as an empty string, corrected_suffix as an empty string, and
  can_stop_after_suffix as false.
- If the response is incorrect, set "is_correct" to false and set "span" to
  the earliest exact substring where the response becomes wrong. Usually this
  is the incorrect answer phrase. Do not include later explanation text.
- The field "span" must be an exact substring of the response unless it is
  "NONE".
- For incorrect responses, the field "replacement" is the answer text that
  should replace that span. Use one of the gold answers, preferably the
  shortest correct answer. Do not invent an answer that is not in the gold
  answer list.
- For incorrect responses, the field "corrected_suffix" is the minimal
  complete continuation that should replace the response from "span" to the
  end. It must begin with "replacement" or include "replacement" as the answer
  phrase, and prefix + corrected_suffix must be grammatical enough to stop.
- Include only text needed to answer the question correctly and end naturally.
  Do not preserve the original response suffix after the span unless that text
  is required for a correct, non-contradictory answer.
- For incorrect responses, if no answer-like span exists in the response,
  output span as "NONE", corrected_suffix as an empty string, and
  can_stop_after_suffix as false.

Output strictly one JSON object on one line:
{{"is_correct": true/false, "span": "<exact substring or NONE>", "replacement": "<text>", "corrected_suffix": "<short complete suffix>", "can_stop_after_suffix": true/false}}

Question: {question}
Gold answers: {answers}
Model response: {response}
"""


def build_repair_judge_prompt(tokenizer, row: Dict) -> str:
    answers = " / ".join(str(x) for x in row.get("answers", []))
    body = REPAIR_JUDGE_TEMPLATE.format(
        question=row["question"],
        answers=answers,
        response=row["rollout"],
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
    )


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_repair(raw: str) -> Optional[Dict]:
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    is_correct = obj.get("is_correct")
    if not isinstance(is_correct, bool):
        return None
    span = obj.get("span")
    replacement = obj.get("replacement")
    corrected_suffix = obj.get("corrected_suffix")
    can_stop_after_suffix = obj.get("can_stop_after_suffix")
    if not isinstance(span, str) or not isinstance(replacement, str):
        return None
    if not isinstance(corrected_suffix, str):
        corrected_suffix = replacement
    if not isinstance(can_stop_after_suffix, bool):
        can_stop_after_suffix = bool(corrected_suffix.strip())
    span = span.strip()
    replacement = replacement.strip()
    corrected_suffix = corrected_suffix.strip()
    if not replacement and not is_correct:
        return None
    if not corrected_suffix and not is_correct:
        return None
    if not span or span.upper() == "NONE":
        return {
            "is_correct": is_correct,
            "span": None,
            "replacement": replacement,
            "corrected_suffix": corrected_suffix,
            "can_stop_after_suffix": False,
        }
    return {
        "is_correct": is_correct,
        "span": span,
        "replacement": replacement,
        "corrected_suffix": corrected_suffix,
        "can_stop_after_suffix": can_stop_after_suffix,
    }


def _normalized_with_offsets(text: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    offsets: List[int] = []
    for idx, char in enumerate(text):
        if char.isalnum():
            chars.append(char.casefold())
            offsets.append(idx)
    return "".join(chars), offsets


_EDGE_PUNCT = set("\"'`“”‘’«»「」『』()[]{}.,;:!?，。！？、")


def _expand_edge_punctuation(text: str, start: int, end: int) -> Tuple[int, int]:
    while start > 0 and text[start - 1] in _EDGE_PUNCT:
        start -= 1
    while end < len(text) and text[end] in _EDGE_PUNCT:
        end += 1
    return start, end


def locate_span(
    response: str,
    span: str,
    *,
    expand_edges: bool = True,
) -> Optional[Tuple[int, int]]:
    idx = response.find(span)
    if idx < 0:
        idx = response.lower().find(span.lower())
    if idx >= 0:
        return idx, idx + len(span)

    stripped = span.strip(" \t\r\n\"'`“”‘’«»「」『』")
    if stripped and stripped != span:
        idx = response.find(stripped)
        if idx < 0:
            idx = response.lower().find(stripped.lower())
        if idx >= 0:
            end = idx + len(stripped)
            if expand_edges:
                return _expand_edge_punctuation(response, idx, end)
            return idx, end

    norm_response, response_offsets = _normalized_with_offsets(response)
    norm_span, _ = _normalized_with_offsets(span)
    if not norm_response or not norm_span:
        return None
    norm_idx = norm_response.find(norm_span)
    if norm_idx < 0:
        return None
    start = response_offsets[norm_idx]
    end = response_offsets[norm_idx + len(norm_span) - 1] + 1
    if expand_edges:
        return _expand_edge_punctuation(response, start, end)
    return start, end


def locate_answer_in_response(
    response: str,
    answers: List[str],
    primary_answer: str,
) -> Optional[Tuple[str, str, str]]:
    response_lower = response.lower()
    best_pos = len(response)
    best_answer: Optional[str] = None
    candidates = [primary_answer] + list(answers)
    for answer in candidates:
        if answer is None:
            continue
        answer_text = str(answer).strip()
        if not answer_text:
            continue
        idx = response_lower.find(answer_text.lower())
        if idx != -1 and idx < best_pos:
            best_pos = idx
            best_answer = response[idx:idx + len(answer_text)]
    if best_answer is None:
        return None
    prefix = response[:best_pos]
    suffix = response[best_pos + len(best_answer):] if prefix else ""
    return prefix, best_answer, suffix


def is_none_answer(primary_answer: str, answers: List[str]) -> bool:
    none_tokens = {"none", "unknown", "n/a", "nil", "null", "0", "zero"}
    primary = str(primary_answer or "").strip().lower()
    if primary in none_tokens:
        return True
    valid = [str(answer).strip().lower() for answer in answers
             if str(answer).strip()]
    return bool(valid) and all(answer in none_tokens for answer in valid)


def _norm_alnum(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def _contains_normalized(text: str, needle: str) -> bool:
    norm_text = _norm_alnum(text)
    norm_needle = _norm_alnum(needle)
    return bool(norm_needle) and norm_needle in norm_text


def _starts_with_normalized(text: str, prefix: str) -> bool:
    norm_text = _norm_alnum(text)
    norm_prefix = _norm_alnum(prefix)
    return bool(norm_prefix) and norm_text.startswith(norm_prefix)


def _ensure_terminal(text: str) -> str:
    text = text.strip()
    text = _strip_unbalanced_edge_quotes(text)
    if not text:
        return text
    stripped = text.rstrip("\"'`“”‘’)]}」』")
    if stripped and stripped[-1] in ".!?。！？":
        return text
    return text + "."


def _strip_unbalanced_edge_quotes(text: str) -> str:
    text = text.strip()
    for quote in ("\"", "'", "`"):
        if text.startswith(quote) and text.count(quote) == 1:
            text = text[1:].lstrip()
        if text.endswith(quote) and text.count(quote) == 1:
            text = text[:-1].rstrip()
    pairs = (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))
    for left, right in pairs:
        if text.startswith(left) and right not in text:
            text = text[len(left):].lstrip()
        if text.endswith(right) and left not in text:
            text = text[:-len(right)].rstrip()
    return text


def corrected_suffix_for_correct(
    candidate: str,
    answer_text: str,
) -> str:
    candidate = (candidate or answer_text).strip()
    answer_text = answer_text.strip()
    if not candidate:
        candidate = answer_text
    if answer_text and not _starts_with_normalized(candidate, answer_text):
        if candidate[:1] in ",;:，、":
            candidate = answer_text + candidate
        elif not _contains_normalized(candidate, answer_text):
            candidate = answer_text
    return _ensure_terminal(candidate)


def corrected_suffix_for_incorrect(primary_answer: str, answers: List[str]) -> str:
    answer = str(primary_answer or "").strip()
    if not answer:
        for candidate in answers:
            answer = str(candidate or "").strip()
            if answer:
                break
    return _ensure_terminal(answer)


def assistant_stop_token_ids(tokenizer) -> set[int]:
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", tokenizer.eos_token):
        if not token:
            continue
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            stop_ids.add(int(token_ids[0]))
    return stop_ids


def build_training_sample(
    tokenizer,
    question: str,
    prefix: str,
    target_text: str,
    label_text: str,
    max_seq_len: int,
    train_stop_token: bool,
) -> Optional[Dict]:
    if not target_text.strip() or not label_text.strip():
        return None

    label_span = locate_span(target_text, label_text, expand_edges=False)
    if label_span is None:
        return None
    target_label_start, target_label_end = label_span

    assistant_content = prefix + target_text
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix_turn = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_char_start = len(prefix_turn) + len(prefix) + target_label_start
    answer_char_end = len(prefix_turn) + len(prefix) + target_label_end

    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    answer_start = answer_end = len(input_ids)
    for idx, (char_start, char_end) in enumerate(offsets):
        if answer_start == len(input_ids) and char_end > answer_char_start:
            answer_start = idx
        if char_start >= answer_char_end:
            answer_end = idx
            break
    if answer_start >= answer_end:
        return None

    labels = [-100] * len(input_ids)
    for idx in range(answer_start, answer_end):
        labels[idx] = input_ids[idx]

    stop_label_index = None
    stop_ids = assistant_stop_token_ids(tokenizer) if train_stop_token else set()
    if train_stop_token and stop_ids:
        for idx in range(len(input_ids) - 1, answer_end - 1, -1):
            if input_ids[idx] in stop_ids:
                labels[idx] = input_ids[idx]
                stop_label_index = idx
                break

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "answer_label_count": max(answer_end - answer_start, 0),
        "stop_label_index": stop_label_index,
        "stop_label_found": int(stop_label_index is not None),
        "target_char_count": len(target_text),
        "label_char_count": max(target_label_end - target_label_start, 0),
    }


def stage_sample_rollouts(
    questions: List[Dict],
    policy_model: str,
    k: int,
    max_new_tokens: int,
    rollouts_path: Path,
    nprocs: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: int,
) -> None:
    from src.baselines.utils.vllm_runner import VLLMRunner

    tokenizer = AutoTokenizer.from_pretrained(policy_model, trust_remote_code=True)
    prompts = [build_raw_question_prompt(tokenizer, q["question"]) for q in questions]
    runner = VLLMRunner(
        model_path=policy_model,
        nprocs=nprocs,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    try:
        samples = runner.generate_samples(
            prompts,
            num_samples=k,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
        )
    finally:
        runner.shutdown()

    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rollouts_path, "w") as f:
        for q, rollouts in zip(questions, samples):
            for rollout_idx, rec in enumerate(rollouts):
                f.write(json.dumps({
                    "qid": q["id"],
                    "question": q["question"],
                    "answers": q["answers"],
                    "primary_answer": q["primary_answer"],
                    "rollout_idx": rollout_idx,
                    "rollout": rec.text,
                    "finish_reason": rec.finish_reason,
                    "stop_reason": rec.stop_reason,
                    "generated_token_count": len(rec.token_ids or []),
                }, ensure_ascii=False) + "\n")
    print(f"[sample] wrote {len(questions) * k} rollouts -> {rollouts_path}")


def stage_judge_repairs(
    rollouts_path: Path,
    judge_model: str,
    repairs_path: Path,
    tp: int,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    judge_max_new_tokens: int,
    judge_batch_size: int,
) -> None:
    from src.baselines.utils.judge_runner import SingleVLLMJudge

    rows = [json.loads(line) for line in open(rollouts_path) if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    judge_batch_size = max(1, judge_batch_size)
    judge = SingleVLLMJudge(
        model_path=judge_model,
        tp=tp,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    repairs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = repairs_path.with_name(repairs_path.name + ".tmp")
    n_parse = 0
    n_located = 0
    try:
        with open(tmp_path, "w") as f:
            for start in range(0, len(rows), judge_batch_size):
                batch_rows = rows[start:start + judge_batch_size]
                prompts = [
                    build_repair_judge_prompt(tokenizer, row)
                    for row in batch_rows
                ]
                outputs = judge.generate_greedy(
                    prompts, max_new_tokens=judge_max_new_tokens)
                for row, raw in zip(batch_rows, outputs):
                    repair = parse_repair(raw)
                    out = dict(row)
                    out["repair_raw"] = raw
                    out["repair"] = repair
                    if repair is not None:
                        n_parse += 1
                        if repair["span"] is not None:
                            located = locate_span(
                                row["rollout"],
                                repair["span"],
                                expand_edges=not repair["is_correct"],
                            )
                            out["repair_span"] = located
                            n_located += int(located is not None)
                        else:
                            out["repair_span"] = None
                    else:
                        out["repair_span"] = None
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                done = min(start + judge_batch_size, len(rows))
                print(
                    f"[repair] batch {done}/{len(rows)} "
                    f"parsed={n_parse} located={n_located}",
                    flush=True,
                )
        os.replace(tmp_path, repairs_path)
    finally:
        judge.shutdown()
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"[repair] parsed={n_parse}/{len(rows)} located={n_located}/{len(rows)} "
          f"-> {repairs_path}")


def save_dataset_disk(records: List[Dict], dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list([
        {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }
        for row in records
    ])
    DatasetDict({"train": dataset}).save_to_disk(str(dataset_dir))


def stage_build_tokenized_dataset(
    repairs_path: Path,
    jsonl_path: Path,
    dataset_dir: Path,
    policy_model: str,
    max_seq_len: int,
    label_policy: str,
    include_original_sft: bool,
    original_queries_jsonl: Optional[Path],
    drop_correct_rollouts: bool,
) -> None:
    if label_policy not in OP_SSFT_LABEL_POLICIES:
        raise SystemExit(f"[error] unknown label_policy={label_policy}")
    tokenizer = AutoTokenizer.from_pretrained(policy_model, trust_remote_code=True)
    records: List[Dict] = []
    label_answer_only = label_policy in {"answer_only", "answer_eos"}
    label_full_response = label_policy == "repaired_response_eos"
    label_stop = label_policy in {
        "answer_eos",
        "corrected_suffix_eos",
        "repaired_response_eos",
    }
    stats = {
        "input": 0,
        "kept": 0,
        "label_policy": label_policy,
        "correct_kept": 0,
        "correct_judge_span": 0,
        "correct_located": 0,
        "correct_fallback": 0,
        "dropped_nonstop": 0,
        "correct_skip_none": 0,
        "incorrect_repaired": 0,
        "incorrect_gold_suffix": 0,
        "answer_present_truncated": 0,
        "dropped_parse": 0,
        "dropped_span": 0,
        "dropped_label_span": 0,
        "dropped_tokenize": 0,
        "original_sft_added": 0,
        "original_sft_dropped": 0,
        "dropped_correct_rollout": 0,
    }

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(repairs_path) as fin, open(jsonl_path, "w") as fout:
        for line in fin:
            stats["input"] += 1
            row = json.loads(line)
            repair = row.get("repair")
            if repair is None:
                stats["dropped_parse"] += 1
                continue
            answers = row.get("answers") or []
            primary_answer = row.get("primary_answer") or (
                answers[0] if answers else "")
            if drop_correct_rollouts and repair.get("is_correct"):
                stats["dropped_correct_rollout"] += 1
                continue
            answer_present = None
            if not is_none_answer(primary_answer, answers):
                answer_present = locate_answer_in_response(
                    row["rollout"], [], primary_answer)

            force_stop_label = False
            if answer_present is not None and (
                    not repair["is_correct"] or
                    repair.get("can_stop_after_suffix") is False):
                prefix, answer_phrase, _ = answer_present
                target_text = corrected_suffix_for_correct(
                    answer_phrase, answer_phrase)
                label_text = answer_phrase if label_answer_only else target_text
                repair_span_text = answer_phrase
                force_stop_label = True
                stats["answer_present_truncated"] += 1
            elif repair.get("can_stop_after_suffix") is False:
                stats["dropped_nonstop"] += 1
                continue
            elif repair["is_correct"]:
                if is_none_answer(primary_answer, answers):
                    stats["correct_skip_none"] += 1
                    continue
                repair_span_text = None
                span = row.get("repair_span")
                if span is not None:
                    start, _ = span
                    prefix = row["rollout"][:start]
                    answer_phrase = row["rollout"][start:span[1]]
                    target_text = corrected_suffix_for_correct(
                        repair.get("corrected_suffix") or answer_phrase,
                        answer_phrase,
                    )
                    label_text = answer_phrase if label_answer_only else target_text
                    repair_span_text = answer_phrase
                    stats["correct_judge_span"] += 1
                    stats["correct_located"] += 1
                else:
                    located = locate_answer_in_response(
                        row["rollout"], answers, primary_answer)
                    if located is not None:
                        prefix, answer_phrase, _ = located
                        target_text = corrected_suffix_for_correct(
                            repair.get("corrected_suffix") or answer_phrase,
                            answer_phrase,
                        )
                        label_text = answer_phrase if label_answer_only else target_text
                        repair_span_text = answer_phrase
                        stats["correct_located"] += 1
                    else:
                        if str(primary_answer).strip():
                            prefix = ""
                            answer_phrase = str(primary_answer).strip()
                            target_text = _ensure_terminal(answer_phrase)
                            label_text = answer_phrase if label_answer_only else target_text
                            repair_span_text = answer_phrase
                            stats["correct_fallback"] += 1
                        else:
                            stats["correct_skip_none"] += 1
                            continue
            else:
                span = row.get("repair_span")
                if span is None:
                    stats["dropped_span"] += 1
                    continue
                start, _ = span
                prefix = row["rollout"][:start]
                replacement = str(repair.get("replacement") or "").strip() or \
                    str(primary_answer or "").strip()
                if not replacement:
                    replacement = str(primary_answer or "").strip()
                target_text = repair.get("corrected_suffix") or \
                    corrected_suffix_for_incorrect(replacement, answers)
                target_text = corrected_suffix_for_correct(
                    target_text, replacement)
                label_text = replacement if label_answer_only else target_text
                repair_span_text = repair["span"]
                stats["incorrect_gold_suffix"] += 1
            sample_prefix = prefix
            sample_target_text = target_text
            sample_label_text = label_text
            if label_full_response:
                sample_target_text = prefix + target_text
                sample_label_text = sample_target_text
                sample_prefix = ""

            sample = build_training_sample(
                tokenizer=tokenizer,
                question=row["question"],
                prefix=sample_prefix,
                target_text=sample_target_text,
                label_text=sample_label_text,
                max_seq_len=max_seq_len,
                train_stop_token=label_stop and
                (force_stop_label or
                 bool(repair.get("can_stop_after_suffix", True))),
            )
            if sample is None:
                if locate_span(target_text, label_text, expand_edges=False) is None:
                    stats["dropped_label_span"] += 1
                else:
                    stats["dropped_tokenize"] += 1
                continue

            sample.update({
                "qid": row["qid"],
                "rollout_idx": row["rollout_idx"],
                "is_rollout_correct": int(repair["is_correct"]),
                "repair_span_text": repair_span_text,
                "replacement": sample_label_text,
                "target_text": sample_target_text,
                "label_policy": label_policy,
                "source": "answer_present_truncate"
                if force_stop_label else "judge_repair",
                "target_char_count": sample.get("target_char_count", 0),
                "label_char_count": sample.get("label_char_count", 0),
                "data_version": OP_SSFT_DATA_VERSION,
            })
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            records.append(sample)
            stats["kept"] += 1
            if repair["is_correct"]:
                stats["correct_kept"] += 1
            else:
                stats["incorrect_repaired"] += 1

        if include_original_sft:
            if original_queries_jsonl is None:
                raise SystemExit(
                    "[error] --include_original_sft requires "
                    "--original_queries_jsonl")
            for row in _load_query_jsonl(original_queries_jsonl):
                answers = row.get("answers") or []
                answer = str(row.get("primary_answer") or (
                    answers[0] if answers else "")).strip()
                if not answer:
                    stats["original_sft_dropped"] += 1
                    continue
                sample = build_training_sample(
                    tokenizer=tokenizer,
                    question=row["question"],
                    prefix="",
                    target_text=answer,
                    label_text=answer,
                    max_seq_len=max_seq_len,
                    train_stop_token=True,
                )
                if sample is None:
                    stats["original_sft_dropped"] += 1
                    continue
                sample.update({
                    "qid": row["id"],
                    "rollout_idx": -1,
                    "is_rollout_correct": 1,
                    "repair_span_text": answer,
                    "replacement": answer,
                    "target_text": answer,
                    "label_policy": f"{label_policy}+original_sft",
                    "source": "original_sft",
                    "target_char_count": sample.get("target_char_count", 0),
                    "label_char_count": sample.get("label_char_count", 0),
                    "data_version": OP_SSFT_DATA_VERSION,
                })
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                records.append(sample)
                stats["kept"] += 1
                stats["original_sft_added"] += 1

    if not records:
        raise SystemExit("[error] no OP-SSFT records survived repair parsing")
    save_dataset_disk(records, dataset_dir)
    print(f"[build] {json.dumps(stats, ensure_ascii=False)}")
    print(f"        jsonl={jsonl_path}")
    print(f"        tokenized_path={dataset_dir}")


def _load_query_jsonl(path: Path, limit: int | None = None) -> List[Dict]:
    rows: List[Dict] = []
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        answers = row.get("answers") or []
        rows.append({
            "id": str(row.get("id", row.get("qid", len(rows)))),
            "question": row["question"],
            "answers": answers,
            "primary_answer": row.get("primary_answer") or (
                answers[0] if answers else ""),
            "split": row.get("split", "train"),
        })
        if limit is not None and len(rows) >= limit:
            break
    return rows


def write_meta(out_dir: Path, args: argparse.Namespace, stats_path: Path) -> None:
    meta = {
        "data_version": OP_SSFT_DATA_VERSION,
        "policy_model": args.policy_model,
        "judge_model": args.judge_model or args.policy_model,
        "dataset": args.dataset,
        "split": args.split,
        "input_jsonl": args.input_jsonl,
        "num_questions": args.num_questions,
        "k": args.k,
        "max_seq_len": args.max_seq_len,
        "label_policy": args.label_policy,
        "include_original_sft": args.include_original_sft,
        "original_queries_jsonl": args.original_queries_jsonl,
        "drop_correct_rollouts": args.drop_correct_rollouts,
        "stats_path": os.path.relpath(stats_path, out_dir),
    }
    (out_dir / "op_ssft_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_model", required=True)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input_jsonl", default=None,
                        help="explicit query JSONL; overrides dataset/split")
    parser.add_argument("--num_questions", type=int, default=5000)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--policy_nprocs", type=int, default=8)
    parser.add_argument("--policy_max_model_len", type=int, default=2048)
    parser.add_argument("--judge_tp", type=int, default=8)
    parser.add_argument("--judge_max_model_len", type=int, default=4096)
    parser.add_argument("--judge_max_new_tokens", type=int, default=192)
    parser.add_argument("--judge_batch_size", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--judge_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_policy", choices=OP_SSFT_LABEL_POLICIES,
                        default="corrected_suffix_eos")
    parser.add_argument("--include_original_sft", action="store_true")
    parser.add_argument("--original_queries_jsonl", default=None)
    parser.add_argument("--drop_correct_rollouts", action="store_true")
    parser.add_argument("--phase", choices=["all", "sample", "repair", "build"],
                        default="all")
    parser.add_argument("--skip_sample", action="store_true")
    parser.add_argument("--skip_repair", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    repairs_path = out_dir / "repairs.jsonl"
    jsonl_path = out_dir / "op_ssft_tokenized.jsonl"
    dataset_dir = out_dir / "op_ssft_tokenized_dataset"
    judge_model = args.judge_model or args.policy_model

    if args.phase in ("all", "sample"):
        if args.skip_sample and rollouts_path.exists():
            print(f"[sample] skipped existing {rollouts_path}")
        else:
            if args.input_jsonl:
                print(f"[sample] loading up to {args.num_questions} questions "
                      f"from {args.input_jsonl}")
                questions = _load_query_jsonl(
                    Path(args.input_jsonl), args.num_questions)
            else:
                print(f"[sample] loading {args.num_questions} questions from "
                      f"{args.dataset}/{args.split}")
                questions = load_dataset_by_name(
                    args.dataset, args.split, args.num_questions)
            if len(questions) != args.num_questions:
                raise SystemExit(
                    f"[error] expected {args.num_questions} questions, "
                    f"loaded {len(questions)}")
            stage_sample_rollouts(
                questions=questions,
                policy_model=args.policy_model,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                rollouts_path=rollouts_path,
                nprocs=args.policy_nprocs,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                dtype=args.dtype,
                max_model_len=args.policy_max_model_len,
            )
        if args.phase == "sample":
            return

    if args.phase in ("all", "repair"):
        if not rollouts_path.exists():
            raise SystemExit(f"[error] missing rollouts for repair: {rollouts_path}")
        if args.skip_repair and repairs_path.exists():
            print(f"[repair] skipped existing {repairs_path}")
        else:
            stage_judge_repairs(
                rollouts_path=rollouts_path,
                judge_model=judge_model,
                repairs_path=repairs_path,
                tp=args.judge_tp,
                dtype=args.dtype,
                max_model_len=args.judge_max_model_len,
                gpu_memory_utilization=args.judge_gpu_memory_utilization,
                judge_max_new_tokens=args.judge_max_new_tokens,
                judge_batch_size=args.judge_batch_size,
            )
        if args.phase == "repair":
            return

    if args.phase in ("all", "build"):
        if not repairs_path.exists():
            raise SystemExit(f"[error] missing repairs for build: {repairs_path}")
        stage_build_tokenized_dataset(
            repairs_path=repairs_path,
            jsonl_path=jsonl_path,
            dataset_dir=dataset_dir,
            policy_model=args.policy_model,
            max_seq_len=args.max_seq_len,
            label_policy=args.label_policy,
            include_original_sft=args.include_original_sft,
            original_queries_jsonl=Path(args.original_queries_jsonl)
            if args.original_queries_jsonl else None,
            drop_correct_rollouts=args.drop_correct_rollouts,
        )
        write_meta(out_dir, args, jsonl_path)
        print(f"[done] OP-SSFT data version={OP_SSFT_DATA_VERSION}")


if __name__ == "__main__":
    main()
