from typing import List


QA_PROMPT_VERSION = "opsft_raw_question_v1"


def build_qa_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def build_qa_prompts(tokenizer, questions: List[str]) -> List[str]:
    return [build_qa_prompt(tokenizer, q) for q in questions]
