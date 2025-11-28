from __future__ import annotations

from typing import List, Dict


def _fmt_examples(examples: List[Dict], max_items: int = 5) -> str:
    out = []
    for i, ex in enumerate(examples[:max_items], 1):
        payload = ex.get("payload", "").strip()
        why = ex.get("reason", ex.get("reasoning", "")).strip()
        src = ex.get("source", "").strip()
        out.append(f"[Ex{i} src={src}]\nPayload: {payload}\nReasoning: {why}\n")
    return "\n".join(out)


def build_prompt_v2(dbms: str,
                    instruction: str,
                    constraints: str,
                    waf_probe: Dict,
                    examples: List[Dict]) -> str:
    """Structured prompt with WAF probe signals and strict output schema.

    Produces two-part output (Payload + concise Reasoning) without chain-of-thought.
    """
    blocked = waf_probe.get("blocked_sqli_keywords", []) if waf_probe else []
    exemplar_block = _fmt_examples(examples)

    prompt = (
        "You are an AI Red Team assistant operating only within a lab target. "
        "Generate one test SQL injection payload and a concise reason.\n\n"
        f"Attack-Profile:\n- DBMS: {dbms}\n- Blocked-Keywords: {blocked}\n\n"
        f"Instruction: {instruction}\n"
        f"Constraints: {constraints}\n\n"
        f"Relevant Examples (for style only):\n{exemplar_block}\n"
        "Policy: Do not include tools, commands, or multi-step reasoning. Keep reasoning to a brief summary.\n\n"
        "Respond exactly in this schema:\n"
        "Payload: <payload>\n"
        "Reasoning: <one or two sentences>\n"
    )
    return prompt

