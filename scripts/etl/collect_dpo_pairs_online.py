import argparse
import json
import random
import re
from pathlib import Path
from typing import List

import httpx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


SQL_ERROR_PATTERNS = [
    r"you have an error in your sql syntax",
    r"warning: mysql_",
    r"sqlstate\{",
    r"uncle(d|s) quotation mark",
    r"incorrect syntax near",
    r"fatal error: uncaught pdoexception",
    r"sqlite error",
    r"syntax error at or near",
]


def extract_payload(text: str) -> str:
    m = re.search(r"Payload:\s*(.*)", text)
    if m:
        return m.group(1).strip()
    # fallback: first non-empty line
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def ensure_login(client: httpx.Client, login_url: str, username: str, password: str) -> bool:
    r = client.get(login_url)
    r.raise_for_status()
    m = re.search(r"user_token' value='([a-f0-9]{32})'", r.text, re.I)
    if not m:
        return False
    token = m.group(1)
    data = {"username": username, "password": password, "user_token": token, "Login": "Login"}
    r = client.post(login_url, data=data)
    r.raise_for_status()
    return "login.php" not in str(r.url)


def eval_payload(client: httpx.Client, sqli_url: str, payload: str) -> str:
    r = client.get(sqli_url, params={"id": payload})
    if r.status_code == 403:
        return "blocked"
    txt = r.text.lower()
    for pat in SQL_ERROR_PATTERNS:
        if re.search(pat, txt):
            return "sql_error_bypass"
    return "passed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL for DPO pairs")
    ap.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--adapter_path", default="", help="Optional LoRA adapter path")
    ap.add_argument("--login_url", default="http://localhost:18081/login.php")
    ap.add_argument("--sqli_url", default="http://localhost:18081/vulnerabilities/sqli/")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="password")
    ap.add_argument("--prompts", type=int, default=50, help="Number of prompts to sample")
    ap.add_argument("--candidates", type=int, default=4, help="Candidates per prompt")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    # Model + tokenizer in 4-bit
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()

    prompt = (
        "You are an AI Red Team assistant operating strictly in a lab.\n"
        "Generate exactly one MySQL SQL injection test payload and a concise reasoning.\n\n"
        "Respond exactly in this schema:\n"
        "Payload: <payload>\n"
        "Reasoning: <one or two sentences>\n"
    )

    # WAF session
    client = httpx.Client(follow_redirects=True, timeout=15.0)
    if not ensure_login(client, args.login_url, args.username, args.password):
        raise SystemExit("DVWA login failed. Open /setup.php and initialize tables, then retry.")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pairs_written = 0

    for _ in range(args.prompts):
        # Encode once
        enc = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=args.candidates,
                pad_token_id=tok.eos_token_id,
            )
        texts: List[str] = tok.batch_decode(out, skip_special_tokens=True)

        candidates = []
        for t in texts:
            # take the last assistant segment
            payload = extract_payload(t)
            if not payload:
                continue
            label = eval_payload(client, args.sqli_url, payload)
            reward = {"blocked": 0.0, "passed": 0.4, "sql_error_bypass": 1.0}[label]
            candidates.append((payload, reward))

        if not candidates:
            continue
        # Choose best vs worst
        candidates.sort(key=lambda x: x[1])
        chosen_payload, chosen_r = candidates[-1]
        rejected_payload, rejected_r = candidates[0]
        if chosen_r <= rejected_r:
            continue

        chosen = f"Payload: {chosen_payload}\nReasoning: Selected by online WAF reward ({chosen_r})."
        rejected = f"Payload: {rejected_payload}\nReasoning: Lower online reward ({rejected_r})."
        ex = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        with open(outp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        pairs_written += 1

    print(f"Wrote {pairs_written} online DPO pairs to {outp}")


if __name__ == '__main__':
    main()

