#!/usr/bin/env python3
import torch
import gc
import os
import json
from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_model(adapter_dir: str = "experiments/red_gemma2_v6_multi/adapter"):
    print('dY\x151 Clearing GPU cache...')
    torch.cuda.empty_cache(); gc.collect()

    base_name = os.environ.get("V6_BASE_MODEL", "google/gemma-2-2b-it")
    print('dYs? Loading v6 multi-attack model base...')
    tok = AutoTokenizer.from_pretrained(base_name)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )

    print('dY"\x15 Loading v6 PEFT adapter...')
    model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.float16)
    model.eval()
    return tok, model


def load_test_prompts() -> List[Dict]:
    # Prefer combined v6 test set; fallback to v5 SQLi tests
    path = PROCESSED / "red_test_v6_multi.jsonl"
    if not path.exists():
        alt = PROCESSED / "red_test.jsonl"
        if alt.exists():
            with open(alt, 'r', encoding='utf-8') as f:
                return [json.loads(l) for l in f]
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def format_prompt(ex: Dict) -> str:
    instr = ex.get('instruction', '').strip()
    ctx = ex.get('context', '').strip()
    cons = ex.get('constraints', '').strip()
    user = f"Instruction: {instr}\nContext: {ctx}\nConstraints: {cons}".strip()
    return user + "\n\nPayload:"


def generate(payload_count: int = 40) -> None:
    tok, model = load_model()
    prompts = load_test_prompts()
    if not prompts:
        raise SystemExit("No test prompts found. Run ETL combine step first.")

    # Balance across attack types if available
    sqli = [p for p in prompts if p.get('attack_type', '').upper() == 'SQLI']
    xss = [p for p in prompts if p.get('attack_type', '').upper() == 'XSS']
    if not sqli:
        sqli = prompts
    if not xss:
        xss = prompts

    wanted_each = max(1, payload_count // 2)
    batch = sqli[:wanted_each] + xss[:wanted_each]

    out_all: List[str] = []
    out_sqli: List[str] = []
    out_xss: List[str] = []

    print('\ndY"? Generating multi-attack payloads...\n')
    for i, ex in enumerate(batch, 1):
        prompt_text = format_prompt(ex)
        inputs = tok(prompt_text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=96,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tok.eos_token_id
            )
        text = tok.decode(outputs[0], skip_special_tokens=True)
        payload = text.split('Payload:')[-1].strip().split('\n')[0].strip() if 'Payload:' in text else text.strip()

        out_all.append(payload)
        t = ex.get('attack_type', '').upper()
        if t == 'SQLI':
            out_sqli.append(payload)
        elif t == 'XSS':
            out_xss.append(payload)

        print(f"[{i}/{len(batch)}] ({t or 'UNK'}) {payload[:90]}")
        if i % 12 == 0:
            torch.cuda.empty_cache(); gc.collect()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / 'v6_multi_payloads.txt', 'w', encoding='utf-8') as f:
        for p in out_all:
            f.write(p + '\n')
    with open(RESULTS / 'v6_sqli_payloads.txt', 'w', encoding='utf-8') as f:
        for p in out_sqli:
            f.write(p + '\n')
    with open(RESULTS / 'v6_xss_payloads.txt', 'w', encoding='utf-8') as f:
        for p in out_xss:
            f.write(p + '\n')

    print("\no. Saved:")
    print("  - results/v6_multi_payloads.txt")
    print("  - results/v6_sqli_payloads.txt")
    print("  - results/v6_xss_payloads.txt")


if __name__ == '__main__':
    generate()

