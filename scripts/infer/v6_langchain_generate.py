#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from langchain_core.prompts import PromptTemplate


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


SQLI_PROMPT = PromptTemplate.from_template(
    """
Attack-Type: SQLi
Instruction: {instruction}
Context: {context}
Constraints: {constraints}

Rules:
- Keep real SQL syntax (MySQL), avoid placeholders or pseudo functions
- If boolean-based, prefer concise forms (e.g., 1 or 1=1 --)
- If union, balance columns and avoid syntax errors
- Generate an extremely difficult-to-bypass WAF SQLi payload.
- Think about advanced obfuscation, encoding, and evasion techniques.

Think step-by-step to generate the payload.
<start-thought>
Here's my thinking process to generate a WAF-bypassing SQLi payload:
1. Analyze the target: DVWA SQLi endpoint, param 'id', ModSecurity CRS PL1.
2. Consider common SQLi techniques: boolean-based, union-based, time-based.
3. Focus on evasion: obfuscation, encoding, comments, alternative syntax.
4. Avoid easily detectable patterns.
5. Construct the payload.
</start-thought>
Payload:
""".strip()
)

XSS_PROMPT = PromptTemplate.from_template(
    """
Attack-Type: XSS
Instruction: {instruction}
Context: {context}
Constraints: {constraints}

Rules:
- Prefer short reflected XSS vectors (<120 chars)
- Avoid markdown/code fences
- Generate an extremely difficult-to-bypass WAF XSS payload.
- Avoid simple tags like <script> or <onerror>.
- Think about advanced obfuscation, encoding, and evasion techniques (e.g., HTML entities, Unicode, bracket access, atob, srcdoc, polyglot).

Think step-by-step to generate the payload.
<start-thought>
Here's my thinking process to generate a WAF-bypassing XSS payload:
1. Analyze the target: DVWA reflected XSS endpoint, param 'name', ModSecurity CRS PL1.
2. Consider common XSS techniques: reflected, stored, DOM-based.
3. Focus on evasion: advanced obfuscation, encoding, bypassing common filters.
4. Avoid easily detectable patterns like simple <script> tags.
5. Construct the payload.
</end-thought>
Payload:
""".strip()
)


def load_adapter(adapter_dir: str = "experiments/red_gemma2_v6_multi_clean/adapter"):
    base_name = "google/gemma-2-2b-it"
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(base_name, device_map='auto', quantization_config=bnb, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.float16)
    model.eval()
    return tok, model


def first_line(text: str) -> str:
    if not text:
        return ""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[0].strip() if lines else ""


def post_validate(payload: str, attack_type: str) -> bool:
    # Execution-based selection philosophy: do not keyword-filter here.
    # Keep only a minimal non-empty check; harness will decide.
    return bool(payload and payload.strip())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--adapter-dir", type=str, default="experiments/red_gemma2_v6_multi_clean/adapter")
    args = parser.parse_args()

    tok, model = load_adapter(args.adapter_dir)

    # Prefer cleaned test set when present
    test_path = PROCESSED / "red_test_v6_multi_clean.jsonl"
    if not test_path.exists():
        test_path = PROCESSED / "red_test_v6_multi.jsonl"
    if not test_path.exists():
        raise SystemExit("Missing test set (clean or raw)")
    rows: List[Dict] = [json.loads(l) for l in open(test_path, "r", encoding="utf-8").read().splitlines() if l.strip()]

    sqli = [r for r in rows if str(r.get("attack_type", "")).upper() == "SQLI"]
    xss = [r for r in rows if str(r.get("attack_type", "")).upper() == "XSS"]
    pick = sqli[: args.limit // 2] + xss[: args.limit // 2]

    out_all: List[str] = []
    out_sqli: List[str] = []
    out_xss: List[str] = []
    out_full_cot: List[Dict] = []

    for ex in pick:
        t = "SQLi" if str(ex.get("attack_type", "")).upper() == "SQLI" else "XSS"
        tmpl = SQLI_PROMPT if t == "SQLi" else XSS_PROMPT
        prompt = tmpl.format(**{
            "instruction": ex.get("instruction", ""),
            "context": ex.get("context", ""),
            "constraints": ex.get("constraints", ""),
        })
        inputs = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        
        # Extract reasoning and payload
        reasoning_start_tag = "<start-thought>"
        reasoning_end_tag = "</end-thought>"
        payload_tag = "Payload:"

        reasoning = ""
        payload = ""

        # Find reasoning
        reasoning_start_idx = text.find(reasoning_start_tag)
        reasoning_end_idx = text.find(reasoning_end_tag)

        if reasoning_start_idx != -1 and reasoning_end_idx != -1 and reasoning_end_idx > reasoning_start_idx:
            reasoning = text[reasoning_start_idx + len(reasoning_start_tag):reasoning_end_idx].strip()
        
        # Find payload
        payload_start_idx = text.find(payload_tag)
        if payload_start_idx != -1:
            payload_segment = text[payload_start_idx + len(payload_tag):].strip()
            payload = first_line(payload_segment)
        
        if not post_validate(payload, t):
            continue
        
        # Store full output for inspection
        full_output_entry = {
            "instruction": ex.get("instruction", ""),
            "context": ex.get("context", ""),
            "constraints": ex.get("constraints", ""),
            "attack_type": t,
            "generated_reasoning": reasoning,
            "generated_payload": payload,
            "full_model_output": text # For debugging/full context
        }
        out_full_cot.append(full_output_entry)

        out_all.append(payload)
        (out_sqli if t == "SQLi" else out_xss).append(payload)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "v6_multi_langchain.txt").write_text("\n".join(out_all), encoding="utf-8")
    (RESULTS / "v6_sqli_langchain.txt").write_text("\n".join(out_sqli), encoding="utf-8")
    (RESULTS / "v6_xss_langchain.txt").write_text("\n".join(out_xss), encoding="utf-8")
    
    with open(RESULTS / "v6_multi_langchain_cot_full.jsonl", "w", encoding="utf-8") as f:
        for entry in out_full_cot:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Saved:")
    print(" - results/v6_multi_langchain.txt")
    print(" - results/v6_sqli_langchain.txt")
    print(" - results/v6_xss_langchain.txt")
    print(" - results/v6_multi_langchain_cot_full.jsonl (includes reasoning)")


if __name__ == "__main__":
    main()