import json
import re
import os
from typing import Any, Optional

import torch

REWARD_NAME = "judgment_answer_think_v2"
REWARD_TYPE = "batch"

THINK_LENGTH_COEF = -0.001
FORMAT_BONUS = 0.1
FORMAT_PENALTY = -1.0
EOS_BONUS = 0.1
EOS_PENALTY = -0.2

_judge_model = None
_judge_tokenizer = None


def _load_judge():
    global _judge_model, _judge_tokenizer
    if _judge_model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    judge_path = os.environ.get(
        "REWARD_JUDGE_MODEL",
        "Qwen/Qwen2.5-3B-Instruct"
    )
    print(f"[LLM Judge] Loading judge model from {judge_path}...")

    _judge_tokenizer = AutoTokenizer.from_pretrained(
        judge_path, trust_remote_code=True
    )
    _judge_model = AutoModelForCausalLM.from_pretrained(
        judge_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _judge_model.eval()
    print(f"[LLM Judge] Judge model loaded successfully.")


def _judge_explanation(question: str, explanation: str) -> bool:
    words = explanation.split()
    if len(words) < 5:
        return False

    stripped = explanation.strip().lower()
    if stripped in ("yes", "no", "i don't know", "i can't", "n/a", "none", "unknown"):
        return False

    _load_judge()

    prompt = (
        f"A model was asked the following question but judged that it cannot answer correctly.\n"
        f"Question: {question}\n"
        f"Model's explanation: {explanation}\n\n"
        f"Is this explanation reasonable? A reasonable explanation should genuinely explain "
        f"why the model is uncertain or doesn't know the answer (e.g., lack of specific knowledge, "
        f"confusion between similar concepts, etc.), rather than being a lazy/meaningless response.\n"
        f"Answer with ONLY one word: reasonable or unreasonable"
    )

    messages = [{"role": "user", "content": prompt}]
    try:
        text = _judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = _judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = _judge_tokenizer(text, return_tensors="pt").to(_judge_model.device)

    with torch.no_grad():
        outputs = _judge_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = _judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    return "reasonable" in result


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s


def check_answer_correct(response_text: str, ground_truth: dict) -> bool:
    if not response_text:
        return False
    normalized_response = normalize_answer(response_text)
    if not normalized_response:
        return False

    normalized_answers = ground_truth.get("normalized_answers", [])
    for na in normalized_answers:
        if na == normalized_response:
            return True
        if len(na) > 2 and na in normalized_response:
            return True
        if len(normalized_response) > 2 and normalized_response in na:
            return True
    return False


def extract_parts(response: str) -> dict:
    result = {
        "think_content": "",
        "judgment": None,
        "response_text": "",
        "has_correct_format": False,
    }

    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        result["think_content"] = think_match.group(1).strip()

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if answer_match:
        word = answer_match.group(1).strip().lower()
        if word in ("yes", "no"):
            result["judgment"] = word

    resp_match = re.search(r"<response>\s*(.*?)\s*</response>", response, re.DOTALL)
    if resp_match:
        result["response_text"] = resp_match.group(1).strip()

    if think_match and result["judgment"] is not None and resp_match:
        result["has_correct_format"] = True

    if result["judgment"] is None:
        text = response.strip().lower()
        for word in ("yes", "no"):
            if re.search(r"\b" + word + r"\b", text):
                result["judgment"] = word
                break

    return result


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    judge_needed = []

    scores = [None] * len(reward_inputs)

    for i, inp in enumerate(reward_inputs):
        response = inp["response"]
        gt_str = inp["ground_truth"]

        try:
            gt = json.loads(gt_str)
        except (json.JSONDecodeError, TypeError):
            gt = {"ability": gt_str, "answers": [], "normalized_answers": [], "question": ""}

        ability = gt.get("ability", "cannot")
        question = gt.get("question", "")
        parts = extract_parts(response)

        judgment = parts["judgment"]
        think_content = parts["think_content"]
        response_text = parts["response_text"]
        has_correct_format = parts["has_correct_format"]

        last_token_id = inp.get("last_token_id", -1)
        eos_token_id = inp.get("eos_token_id", -1)
        has_eos = (last_token_id == eos_token_id) if eos_token_id >= 0 else False
        eos_reward = EOS_BONUS if has_eos else EOS_PENALTY

        if judgment is None:
            scores[i] = {
                "overall": FORMAT_PENALTY + eos_reward,
                "format": 0.0,
                "accuracy": 0.0,
                "answer_correct": 0.0,
                "explanation_quality": 0.0,
                "think_len": 0.0,
                "has_eos": 1.0 if has_eos else 0.0,
            }
            continue

        answer_correct = check_answer_correct(response_text, gt)
        judgment_correct = (ability == "can" and judgment == "yes") or \
                          (ability == "cannot" and judgment == "no")
        format_bonus = FORMAT_BONUS if has_correct_format else 0.0
        think_penalty = THINK_LENGTH_COEF * len(think_content)

        if ability == "can" and judgment == "yes":
            reward = 2.0 if answer_correct else -1.5
            explanation_quality = 0.0
        elif ability == "can" and judgment == "no":
            reward = -1.0
            explanation_quality = 0.0
        elif ability == "cannot" and judgment == "yes":
            reward = -0.5 if answer_correct else -2.0
            explanation_quality = 0.0
        elif ability == "cannot" and judgment == "no":
            reward = None
            explanation_quality = None
            judge_needed.append((i, question, response_text))
        else:
            reward = FORMAT_PENALTY
            explanation_quality = 0.0

        if reward is not None:
            total_reward = reward + format_bonus + think_penalty + eos_reward
            scores[i] = {
                "overall": total_reward,
                "format": 1.0 if has_correct_format else 0.0,
                "accuracy": 1.0 if judgment_correct else 0.0,
                "answer_correct": 1.0 if answer_correct else 0.0,
                "explanation_quality": explanation_quality,
                "think_len": float(len(think_content)),
                "has_eos": 1.0 if has_eos else 0.0,
            }

    if judge_needed:
        for idx, question, explanation in judge_needed:
            inp = reward_inputs[idx]
            response = inp["response"]
            gt = json.loads(inp["ground_truth"])
            parts = extract_parts(response)
            think_content = parts["think_content"]
            has_correct_format = parts["has_correct_format"]

            is_reasonable = _judge_explanation(question, explanation)
            reward = 0.5 if is_reasonable else -0.5
            explanation_quality = 1.0 if is_reasonable else 0.0

            format_bonus = FORMAT_BONUS if has_correct_format else 0.0
            think_penalty = THINK_LENGTH_COEF * len(think_content)
            total_reward = reward + format_bonus + think_penalty

            last_token_id = inp.get("last_token_id", -1)
            eos_token_id = inp.get("eos_token_id", -1)
            has_eos = (last_token_id == eos_token_id) if eos_token_id >= 0 else False
            eos_reward = EOS_BONUS if has_eos else EOS_PENALTY
            total_reward = total_reward + eos_reward

            scores[idx] = {
                "overall": total_reward,
                "format": 1.0 if has_correct_format else 0.0,
                "accuracy": 1.0,
                "answer_correct": 0.0,
                "explanation_quality": explanation_quality,
                "think_len": float(len(think_content)),
                "has_eos": 1.0 if has_eos else 0.0,
            }

    return scores
