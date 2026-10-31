import re
import string
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join([ch if ch in text and ch not in exclude else ' ' for ch in text])

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


normalize_answer = _normalize_answer


def get_tokens(text: str) -> List[str]:
    return _normalize_answer(text).split()


def exact_match(response: str, gold_answers: List[str]) -> bool:
    norm_response = _normalize_answer(response)
    for answer in gold_answers:
        if _normalize_answer(answer) == norm_response:
            return True
    return False


def compute_f1(response: str, gold_answers: List[str]) -> float:
    response_tokens = get_tokens(response)
    if not response_tokens:
        return 0.0

    best_f1 = 0.0
    for answer in gold_answers:
        answer_tokens = get_tokens(answer)
        if not answer_tokens:
            continue

        common = Counter(response_tokens) & Counter(answer_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(response_tokens)
        recall = num_common / len(answer_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def is_correct(response: str, gold_answers: List[str], f1_threshold: float = 0.5) -> bool:
    return compute_f1(response, gold_answers) >= f1_threshold


def compute_metrics(response: str, gold_answers: List[str]) -> Dict:
    f1 = compute_f1(response, gold_answers)
    return {
        "em": exact_match(response, gold_answers),
        "f1": f1,
        "is_correct": f1 >= 0.5,
    }


def evaluate_responses(responses: List[str], gold_answers: List[str]) -> Dict:
    results = [is_correct(r, gold_answers) for r in responses]
    correct_count = sum(results)
    total = len(responses)
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        "correct_count": correct_count,
        "total": total,
        "accuracy": accuracy,
        "per_response": results,
    }


def classify_ability(correct_count: int, total: int = 5) -> str:
    if correct_count == total:
        return "can"
    elif correct_count == 0:
        return "cannot"
    else:
        return "uncertain"


def classify_ability_binary(correct: bool, inverse_labels: bool = False) -> str:
    if inverse_labels:
        return "cannot" if correct else "can"
    else:
        return "can" if correct else "cannot"


def evaluate_samples(samples: List[Dict]) -> List[Dict]:
    results = []
    for sample in tqdm(samples, desc="Evaluating responses"):
        eval_result = evaluate_responses(
            sample["responses"],
            sample.get("normalized_answers", sample["answers"])
        )

        ability = classify_ability(eval_result["correct_count"], eval_result["total"])

        result = sample.copy()
        result["evaluation"] = eval_result
        result["ability"] = ability
        results.append(result)

    return results
