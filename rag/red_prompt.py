from __future__ import annotations

from typing import List, Dict


def format_examples(examples: List[Dict], max_items: int = 5) -> str:
    lines = []
    for i, ex in enumerate(examples[:max_items], 1):
        payload = ex.get("payload", "")
        reason = ex.get("reason", ex.get("reasoning", ""))
        src = ex.get("source", "")
        lines.append(f"[Ex{i} src={src}]\npayload: {payload}\nwhy: {reason}\n")
    return "\n".join(lines)


def build_prompt(instruction: str, context: str, constraints: str, examples: List[Dict]) -> str:
    exemplar_block = format_examples(examples)
    prompt = (
        "You are an AI Red Team assistant operating strictly in a lab. "
        "Generate a single SQL injection test payload and a concise reasoning for WAF evaluation. "
        "Do not include any destructive actions.\n\n"
        f"Instruction: {instruction}\n"
        f"Context: {context}\n"
        f"Constraints: {constraints}\n\n"
        f"Relevant examples (for style only):\n{exemplar_block}\n\n"
        "Respond in the following format exactly:\n"
        "Payload: <payload>\n"
        "Reasoning: <one or two sentences>\n"
    )
    return prompt

