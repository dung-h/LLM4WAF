"""
Dataset Construction - Phase 1 (LLM + WAF filter)

- Sinh payload SQLI/XSS bằng LLM (DeepSeek OpenAI-compatible hoặc Gemini).
- Kiểm tra qua WAF DVWA (ModSecurity) và chỉ giữ mẫu PASSED.
- Xuất JSONL giống `data/processed/phase1_passed_only_39k.jsonl`.

Yêu cầu:
- WAF/DVWA chạy tại `http://localhost:8000/dvwa` (có thể đổi bằng --waf-base).
- API key:
    * Ưu tiên: DEEPSEEK_API_KEY (OpenAI style)
    * Fallback: GOOGLE_API_KEY (Gemini REST)
"""

import argparse
import json
import os
import random
import time
import logging
from typing import Dict, List

import httpx

logger = logging.getLogger("dataset_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PHASE1_PROMPT_TEMPLATE = """Generate a {attack_type} payload that bypasses WAF.
Technique: {technique}

IMPORTANT: Generate ONLY the payload. Do not explain."""

TECHNIQUES_SLI = [
    "Double URL Encode",
    "Triple URL Encode",
    "Comment Obfuscation",
    "UNION SELECT",
    "Boolean OR tautology",
    "Hex encoded string injection",
    "Inline comment split keyword",
    "Error-based UPDATEXML",
]

TECHNIQUES_XSS = [
    "Script tag",
    "IMG onerror",
    "SVG onload",
    "Double URL encode script",
    "Event handler (onmouseover)",
    "JS protocol",
]


def choose_provider() -> str:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    raise RuntimeError("Missing API key: set DEEPSEEK_API_KEY or GOOGLE_API_KEY")


def call_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    resp = httpx.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    api_key = os.environ["GOOGLE_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 128},
    }
    resp = httpx.post(url, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_payload(prompt: str, provider: str) -> str:
    if provider == "deepseek":
        return call_deepseek(prompt)
    if provider == "gemini":
        return call_gemini(prompt)
    raise ValueError(f"Unknown provider {provider}")


def waf_check(payload: str, attack_type: str, base_url: str) -> Dict[str, str]:
    """
    Gửi payload tới DVWA qua WAF.
    Returns: {"result": "passed"/"blocked"/"error", "status_code": int}
    """
    client = httpx.Client(timeout=10.0, follow_redirects=True)
    try:
        if attack_type == "SQLI":
            url = f"{base_url}/vulnerabilities/sqli/"
            r = client.get(url, params={"id": payload, "Submit": "Submit"})
        else:  # XSS demo
            url = f"{base_url}/vulnerabilities/xss_r/"
            r = client.get(url, params={"name": payload, "Submit": "Submit"})
        status = "passed" if r.status_code != 403 else "blocked"
        return {"result": status, "status_code": r.status_code}
    except Exception as e:
        return {"result": "error", "status_code": -1, "error": str(e)}
    finally:
        client.close()


def build_phase1(args):
    provider = choose_provider()
    logger.info(f"Using provider: {provider}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    random.seed(42)

    techniques = TECHNIQUES_SLI if args.attack_type == "SQLI" else TECHNIQUES_XSS
    kept = 0
    attempted = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for tech in techniques:
            for _ in range(args.num_per_tech):
                prompt = PHASE1_PROMPT_TEMPLATE.format(attack_type=args.attack_type, technique=tech)
                attempted += 1
                try:
                    payload = generate_payload(prompt, provider)
                except Exception as e:
                    logger.error(f"LLM error ({tech}): {e}")
                    continue

                waf_res = waf_check(payload, args.attack_type, args.waf_base.rstrip("/"))
                if waf_res["result"] != "passed":
                    continue

                record = {
                    "attack_type": args.attack_type,
                    "technique": tech,
                    "result": waf_res["result"],
                    "status_code": waf_res["status_code"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": payload},
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                logger.info(f"[KEPT] {tech} status={waf_res['status_code']} kept={kept}")
                time.sleep(args.sleep)

    logger.info(f"Done. Kept {kept}/{attempted} samples -> {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="Build Phase 1 dataset via LLM + WAF filter")
    p.add_argument("--output", default="data/processed/phase1_passed_only_generated.jsonl", help="Output JSONL path")
    p.add_argument("--attack-type", default="SQLI", choices=["SQLI", "XSS"])
    p.add_argument("--num-per-tech", type=int, default=10, help="Samples per technique")
    p.add_argument("--waf-base", default="http://localhost:8000/dvwa", help="Base DVWA URL (behind WAF)")
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_phase1(args)
