import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DownloadConfig, load_dataset
from tqdm import tqdm


SUPPORTED_DATASETS = ["triviaqa", "hotpotqa"]
_KT_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _triviaqa_local_candidates(split: str) -> List[Path]:
    root_env = os.environ.get("KT_DATA_ROOT")
    roots = []
    if root_env:
        roots.append(Path(root_env).expanduser())
    roots.extend([
        Path.cwd() / "data" / "triviaqa",
        _REPO_ROOT / "data" / "triviaqa",
        _KT_ROOT / "data" / "triviaqa",
    ])

    if split in {"validation", "val", "dev"}:
        names = ["triviaqa_validation_all.jsonl", "triviaqa_validation_all.json"]
    elif split == "test":
        names = ["triviaqa_test_all.jsonl", "triviaqa_test_all.json"]
    else:
        names = ["triviaqa_knowledge_all.jsonl", "triviaqa_knowledge_all.json"]

    return [root / name for root in roots for name in names]


def _triviaqa_local_roots() -> List[Path]:
    root_env = os.environ.get("KT_DATA_ROOT")
    roots = []
    if root_env:
        roots.append(Path(root_env).expanduser())
    roots.extend([
        Path.cwd() / "data" / "triviaqa",
        _REPO_ROOT / "data" / "triviaqa",
        _KT_ROOT / "data" / "triviaqa",
    ])
    out = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _triviaqa_chunk_pattern(split: str) -> str:
    if split in {"validation", "val", "dev"}:
        return "triviaqa_validation_*.json"
    if split == "test":
        return "triviaqa_test_*.json"
    return "triviaqa_knowledge_*.json"


def _hf_split_name(split: str) -> str:
    if split in {"validation", "val", "dev"}:
        return "validation"
    if split == "test":
        return "test"
    return "train"


def _chunk_sort_key(path: Path):
    m = re.search(r"_(\d+)_(\d+)\.jsonl?$", path.name)
    if m:
        return int(m.group(1)), int(m.group(2)), path.name
    return (10**12, 10**12, path.name)


def _hf_cache_roots() -> List[Path]:
    roots = []
    hf_home = os.environ.get("HF_HOME")
    datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if datasets_cache:
        roots.append(Path(datasets_cache).expanduser())
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "datasets")
    if user:
        roots.append(Path("/home") / user / ".cache" / "huggingface" / "datasets")
    for base in [Path.cwd(), _REPO_ROOT, _KT_ROOT]:
        for parent in [base, *base.parents]:
            roots.append(parent / ".cache" / "huggingface" / "datasets")
    roots.extend([
        Path.home() / ".cache" / "huggingface" / "datasets",
    ])
    out = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        try:
            if root.is_dir():
                out.append(root)
        except OSError:
            continue
    return out


def _triviaqa_arrow_candidates(split: str) -> List[Path]:
    hf_split = _hf_split_name(split)
    out = []
    for root in _hf_cache_roots():
        base = root / "mandarjoshi___trivia_qa" / "rc.nocontext"
        try:
            out.extend(base.glob(f"**/trivia_qa-{hf_split}.arrow"))
            out.extend(base.glob(f"**/trivia_qa-{hf_split}-*.arrow"))
        except OSError:
            continue
    return sorted(set(out), key=lambda path: len(path.parts), reverse=True)


def _iter_local_rows(path: Path):
    if path.suffix == ".jsonl":
        yield from (json.loads(line) for line in path.open())
    else:
        yield from json.load(path.open())


def _sample_from_hf_item(item: Dict, idx: int) -> Dict:
    answer = item["answer"]
    return {
        "id": item.get("question_id", str(idx)),
        "question": item["question"],
        "answers": answer["aliases"],
        "normalized_answers": answer["normalized_aliases"],
        "primary_answer": answer["value"],
        "normalized_primary": answer["normalized_value"],
    }


def _sample_from_local(row: Dict, idx: int) -> Dict:
    if "question" in row:
        question = row["question"]
        answers = row.get("answers") or []
        primary = row.get("primary_answer") or (answers[0] if answers else "")
    elif "conversations" in row:
        conversations = row["conversations"]
        question = conversations[0]["value"].split("Question:", 1)[-1].strip()
        primary = conversations[1]["value"].strip()
        answers = [primary]
    else:
        raise ValueError(f"Unrecognized TriviaQA local row keys: {sorted(row)}")
    normalized = row.get("normalized_answers") or [
        _normalize_answer(answer) for answer in answers
    ]
    normalized_primary = row.get("normalized_primary") or \
        _normalize_answer(primary)
    return {
        "id": row.get("id", row.get("qid", row.get("idx", str(idx)))),
        "question": question,
        "answers": answers,
        "normalized_answers": normalized,
        "primary_answer": primary,
        "normalized_primary": normalized_primary,
    }


def _load_triviaqa_local(split: str, num_samples: Optional[int]) -> Optional[List[Dict]]:
    for path in _triviaqa_local_candidates(split):
        if not path.exists():
            continue
        samples = []
        total = num_samples
        for idx, row in tqdm(enumerate(_iter_local_rows(path)), total=total,
                             desc=f"Loading local TriviaQA {split}"):
            if num_samples and idx >= num_samples:
                break
            samples.append(_sample_from_local(row, idx))
        print(f"[data] loaded TriviaQA {split} from {path}")
        return samples

    pattern = _triviaqa_chunk_pattern(split)
    for root in _triviaqa_local_roots():
        if not root.exists():
            continue
        paths = [
            path for path in root.glob(pattern)
            if "_all." not in path.name and path.suffix in {".json", ".jsonl"}
        ]
        if not paths:
            continue
        samples = []
        paths = sorted(paths, key=_chunk_sort_key)
        total = num_samples
        pbar = tqdm(total=total, desc=f"Loading local TriviaQA {split} chunks")
        try:
            for path in paths:
                for row in _iter_local_rows(path):
                    if num_samples and len(samples) >= num_samples:
                        break
                    samples.append(_sample_from_local(row, len(samples)))
                    pbar.update(1)
                if num_samples and len(samples) >= num_samples:
                    break
        finally:
            pbar.close()
        if samples:
            print(f"[data] loaded TriviaQA {split} from {len(paths)} chunk(s) in {root}")
            return samples
    return None


def _load_triviaqa_arrow_cache(split: str, num_samples: Optional[int]) -> Optional[List[Dict]]:
    paths = _triviaqa_arrow_candidates(split)
    if not paths:
        return None
    samples = []
    total = num_samples
    pbar = tqdm(total=total, desc=f"Loading cached TriviaQA {split} arrow")
    try:
        for path in paths:
            dataset = Dataset.from_file(str(path))
            for item in dataset:
                if num_samples and len(samples) >= num_samples:
                    break
                samples.append(_sample_from_hf_item(item, len(samples)))
                pbar.update(1)
            if num_samples and len(samples) >= num_samples:
                break
    finally:
        pbar.close()
    if samples:
        print(f"[data] loaded TriviaQA {split} from HF arrow cache: {paths[0]}")
        return samples
    return None


def _hf_dataset_download_allowed() -> bool:
    return os.environ.get("KT_ALLOW_HF_DATASET_DOWNLOAD", "0").lower() in {
        "1", "true", "yes"
    }


def _missing_triviaqa_message(split: str) -> str:
    searched_files = "\n".join(f"  - {path}" for path in _triviaqa_local_candidates(split))
    searched_chunks = "\n".join(
        f"  - {root / _triviaqa_chunk_pattern(split)}"
        for root in _triviaqa_local_roots()
    )
    searched_arrows = "\n".join(
        f"  - {root / 'mandarjoshi___trivia_qa/rc.nocontext/**' / ('trivia_qa-' + _hf_split_name(split) + '.arrow')}"
        for root in _hf_cache_roots()
    )
    return (
        "Local TriviaQA files were not found.\n"
        "Set KT_DATA_ROOT to the directory containing triviaqa_* files, or set "
        "KT_ALLOW_HF_DATASET_DOWNLOAD=1 to permit HuggingFace downloads.\n"
        "Searched full files:\n"
        f"{searched_files}\n"
        "Searched chunk patterns:\n"
        f"{searched_chunks}\n"
        "Searched HF arrow cache patterns:\n"
        f"{searched_arrows}"
    )


def _raise_missing_triviaqa(split: str, cache_error: Optional[Exception] = None) -> None:
    msg = _missing_triviaqa_message(split)
    if cache_error is not None:
        msg += f"\nHF local cache lookup also failed: {cache_error}"
    raise FileNotFoundError(msg)


def _load_hf_dataset(path: str, name: str, split: str,
                     local_only: Optional[bool] = None):
    offline = os.environ.get("HF_DATASETS_OFFLINE", "1").lower() in \
        {"1", "true", "yes"}
    if local_only is None:
        local_only = offline
    if local_only:
        return load_dataset(
            path,
            name,
            split=split,
            download_config=DownloadConfig(local_files_only=True),
        )
    return load_dataset(path, name, split=split)


def load_dataset_by_name(
    dataset_name: str,
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict]:
    dataset_name = dataset_name.lower()
    if dataset_name == "triviaqa":
        return load_triviaqa(split=split, num_samples=num_samples)
    elif dataset_name == "hotpotqa":
        return load_hotpotqa(split=split, num_samples=num_samples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {SUPPORTED_DATASETS}")


def load_triviaqa(split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
    local = _load_triviaqa_local(split, num_samples)
    if local is not None:
        return local

    arrow = _load_triviaqa_arrow_cache(split, num_samples)
    if arrow is not None:
        return arrow

    try:
        dataset = _load_hf_dataset(
            "mandarjoshi/trivia_qa",
            "rc.nocontext",
            split,
            local_only=True,
        )
    except Exception as exc:
        if not _hf_dataset_download_allowed():
            _raise_missing_triviaqa(split, exc)
        dataset = _load_hf_dataset(
            "mandarjoshi/trivia_qa",
            "rc.nocontext",
            split,
            local_only=False,
        )


    total = num_samples if num_samples else len(dataset)
    samples = []
    for i, item in tqdm(enumerate(dataset), total=total, desc=f"Loading {split} data"):
        if num_samples and i >= num_samples:
            break

        sample = {
            "id": item.get("question_id", str(i)),
            "question": item["question"],
            "answers": item["answer"]["aliases"],
            "normalized_answers": item["answer"]["normalized_aliases"],
            "primary_answer": item["answer"]["value"],
            "normalized_primary": item["answer"]["normalized_value"],
        }
        samples.append(sample)

    return samples


def load_hotpotqa(split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
    dataset = _load_hf_dataset("hotpotqa/hotpot_qa", "distractor", split)

    total = num_samples if num_samples else len(dataset)
    samples = []
    for i, item in tqdm(enumerate(dataset), total=total, desc=f"Loading HotpotQA {split} data"):
        if num_samples and i >= num_samples:
            break

        answer = item["answer"]
        normalized_answer = answer.lower().strip()

        context_data = item.get("context", {})
        titles = context_data.get("title", [])
        sentences_list = context_data.get("sentences", [])

        context_paragraphs = []
        for title, sentences in zip(titles, sentences_list):
            paragraph_text = "".join(sentences)
            context_paragraphs.append(f"[{title}]\n{paragraph_text}")

        context = "\n\n".join(context_paragraphs)

        sample = {
            "id": item["id"],
            "question": item["question"],
            "answers": [answer],
            "normalized_answers": [normalized_answer],
            "primary_answer": answer,
            "normalized_primary": normalized_answer,
            "context": context,
            "type": item.get("type", ""),
            "level": item.get("level", ""),
        }
        samples.append(sample)

    return samples


def format_question_prompt(question: str, context: Optional[str] = None) -> str:
    instruction = "Answer the following question with only the answer, nothing else."
    if context:
        return f"{instruction}\n\n{context}\n\n{question}"
    else:
        return f"{instruction}\n\n{question}"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TriviaQA")
    print("=" * 60)
    samples = load_triviaqa("validation", num_samples=2)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answers'][:3]}")
        print(f"Prompt:\n{format_question_prompt(s['question'], s.get('context'))[:200]}...")
        print("-" * 50)

    print("\n" + "=" * 60)
    print("Testing HotpotQA (with context)")
    print("=" * 60)
    samples = load_hotpotqa("validation", num_samples=2)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answers']}")
        print(f"Type: {s['type']}, Level: {s['level']}")
        print(f"Context length: {len(s['context'])} chars")
        print(f"Prompt preview:\n{format_question_prompt(s['question'], s['context'])[:500]}...")
        print("-" * 50)

    print("\n" + "=" * 60)
    print("Testing unified loader")
    print("=" * 60)
    for dataset_name in SUPPORTED_DATASETS:
        samples = load_dataset_by_name(dataset_name, "validation", num_samples=1)
        print(f"\n{dataset_name.upper()}:")
        for s in samples:
            print(f"  Q: {s['question'][:60]}...")
            print(f"  A: {s['primary_answer']}")
            has_context = "Yes" if s.get("context") else "No"
            print(f"  Has context: {has_context}")
