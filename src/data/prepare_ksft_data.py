#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import Counter


JUDGMENT_SYSTEM = "Assess whether you can answer the question correctly. Wrap your answer in <answer> tags: <answer>yes</answer> or <answer>no</answer>."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected_file", type=str, required=True,
                        help="Step2 corrected JSONL (with prefix/replacement/original_response/is_correct)")
    parser.add_argument("--answers_file", type=str, default=None,
                        help="Step1 responses JSONL for locating answers in correct samples")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for LlamaFactory")
    parser.add_argument("--mode", type=str, default="replacement",
                        choices=["replacement", "full", "unlikelihood"],
                        help="Data mode: replacement (span masking), full (answer-only), "
                             "or unlikelihood (Welleck UL: wrong samples get -log(1-p) at "
                             "first-wrong-token only; correct samples get normal full SFT).")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name for LlamaFactory dataset_info.json")
    parser.add_argument("--drop_correct", action="store_true",
                        help="Drop is_correct=True samples entirely (replacement mode). "
                             "Only train on samples that actually had an error to correct.")
    parser.add_argument("--full_supervision", action="store_true",
                        help="Supervise the entire assistant_content (prefix + replacement) "
                             "instead of only the replacement span. Equivalent to answer-only "
                             "SFT on prefix+replacement targets. When set, span_wrong_text / "
                             "span_correct_text are omitted so supervised.py does not apply "
                             "span masking.")
    parser.add_argument("--unify_replacement", action="store_true",
                        help="Force replacement token to the normalized primary_answer for "
                             "all wrong samples of the same question. Reduces gradient "
                             "direction noise across the 18 wrong samples per question.")
    return parser.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def locate_answer_in_response(response, normalized_answers, primary_answer):
    resp_lower = response.lower()
    best_pos = len(response)
    best_answer = None
    for ans in [primary_answer] + normalized_answers:
        if not ans or not ans.strip():
            continue
        ans_norm = ans.lower().strip()
        idx = resp_lower.find(ans_norm)
        if idx != -1 and idx < best_pos:
            best_pos = idx
            best_answer = response[idx: idx + len(ans_norm)]
    if best_answer is None:
        return None
    return response[:best_pos], best_answer


def is_none_answer(primary_answer, normalized_answers):
    pa = (primary_answer or "").lower().strip()
    none_tokens = {"none", "unknown", "n/a", "nil", "null", "0", "zero"}
    if pa in none_tokens:
        return True
    valid = [a for a in normalized_answers if a and a.strip()]
    if valid and all(a.lower().strip() in none_tokens for a in valid):
        return True
    return False


def load_answers_map(answers_file):
    answers_map = {}
    for line in open(answers_file):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        answers_map[r.get("id", "")] = {
            "normalized_answers": r.get("normalized_answers", []),
            "primary_answer": r.get("primary_answer", ""),
        }
    return answers_map


TERMINAL_PUNCT = ".!?。！？"


def has_text_content(text):
    return any(ch.isalnum() for ch in text)


def include_terminal_period_if_answer_ends(prefix, replacement, assistant_content):
    if not replacement.strip():
        return assistant_content, replacement, False

    marker = prefix + replacement
    start = assistant_content.find(marker)
    if start < 0:
        assistant_content = marker
        start = 0
    end = start + len(marker)
    suffix = assistant_content[end:]
    if has_text_content(suffix):
        return assistant_content, replacement, False

    suffix_text = suffix.strip()
    if suffix_text:
        replacement = replacement + suffix_text
        assistant_content = assistant_content[:end] + suffix_text
        return assistant_content, replacement, False

    if replacement.rstrip().endswith(tuple(TERMINAL_PUNCT)):
        return assistant_content, replacement, False

    return assistant_content + ".", replacement + ".", True


def main():
    args = parse_args()

    corrected = load_jsonl(args.corrected_file)
    print(f"Loaded {len(corrected)} corrected samples")

    answers_map = {}
    if args.answers_file and args.mode == "replacement":
        answers_map = load_answers_map(args.answers_file)
        print(f"Loaded {len(answers_map)} answer entries")

    if args.mode == "replacement":
        data = []
        n_correct_located = 0
        n_correct_fallback = 0
        n_correct_skip = 0
        n_correct_dropped = 0
        n_terminal_period_added = 0
        n_wrong = 0
        n_skip = 0

        for r in corrected:
            question = r["question"]
            is_correct = r.get("is_correct", False)

            if is_correct:
                if args.drop_correct:
                    n_correct_dropped += 1
                    continue
                if r.get("replacement", "").strip():
                    prefix = r.get("prefix", "")
                    replacement = r["replacement"]
                    n_correct_located += 1
                else:
                    info = answers_map.get(r.get("id", ""), {})
                    na = info.get("normalized_answers", [])
                    pa = info.get("primary_answer", "")

                    if is_none_answer(pa, na):
                        n_correct_skip += 1
                        continue

                    result = locate_answer_in_response(
                        r.get("original_response", ""), na, pa
                    )
                    if result:
                        prefix, replacement = result
                        n_correct_located += 1
                    elif pa.strip():
                        prefix = ""
                        replacement = pa
                        n_correct_fallback += 1
                    else:
                        n_correct_skip += 1
                        continue

                assistant_content = r.get("original_response", "")
                if prefix and prefix in assistant_content:
                    pass
                else:
                    assistant_content = prefix + replacement

            else:
                replacement = r.get("replacement", "").strip()
                if not replacement:
                    n_skip += 1
                    continue
                prefix = r.get("prefix", "")
                if args.unify_replacement:
                    info = answers_map.get(r.get("id", ""), {})
                    pa = (info.get("primary_answer") or "").strip()
                    if pa:
                        replacement = pa
                n_wrong += 1
                assistant_content = prefix + replacement

            assistant_content, replacement, added_period = (
                include_terminal_period_if_answer_ends(
                    prefix, replacement, assistant_content))
            n_terminal_period_added += int(added_period)

            item = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": assistant_content},
                ],
            }
            if not args.full_supervision:
                item["span_wrong_text"] = assistant_content
                item["span_correct_text"] = replacement
            data.append(item)

        print(f"Replacement mode: {len(data)} samples")
        print(f"  Correct: located={n_correct_located}, fallback={n_correct_fallback}, skip={n_correct_skip}, dropped={n_correct_dropped}")
        print(f"  Wrong: {n_wrong}, skip={n_skip}")
        print(f"  Terminal periods added: {n_terminal_period_added}")

    elif args.mode == "unlikelihood":
        data = []
        n_wrong = 0
        n_correct = 0
        n_skip = 0

        for r in corrected:
            question = r["question"]
            is_correct = r.get("is_correct", False)
            original = r.get("original_response", "")
            if not original.strip():
                n_skip += 1
                continue

            if is_correct:
                if args.drop_correct:
                    continue
                item = {
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": original},
                    ],
                }
                n_correct += 1
            else:
                prefix = r.get("prefix", "")
                if not original.startswith(prefix):
                    n_skip += 1
                    continue
                if len(prefix) >= len(original):
                    n_skip += 1
                    continue
                item = {
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": original},
                    ],
                    "ul_wrong_prefix": prefix,
                }
                n_wrong += 1
            data.append(item)

        print(f"Unlikelihood mode: {len(data)} samples")
        print(f"  Wrong (UL): {n_wrong}, Correct (SFT): {n_correct}, skipped: {n_skip}")

    elif args.mode == "full":
        data = []
        for r in corrected:
            question = r["question"]
            response = r.get("corrected_response", "") or r.get("original_response", "")
            if not response.strip():
                continue

            item = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": response},
                ],
            }
            data.append(item)

        print(f"Full mode: {len(data)} samples")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path} ({len(data)} samples)")

    dataset_name = args.dataset_name or out_path.stem
    dataset_info_path = out_path.parent / "dataset_info.json"
    info = {}
    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            info = json.load(f)

    info[dataset_name] = {
        "file_name": out_path.name,
        "columns": {"messages": "conversations"},
        "formatting": "sharegpt"
    }
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Registered dataset: {dataset_name}")


if __name__ == "__main__":
    main()
