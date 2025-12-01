import argparse
import os
import json
import re
import httpx 
import torch
import asyncio
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter
from torch.utils.data import DataLoader

# --- WAF Test Configuration ---
DVWA_BASE_URL = "http://localhost:8000"
DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
DVWA_SQLI_BLIND_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli_blind/"
DVWA_XSS_REFLECTED_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
DVWA_XSS_DOM_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_d/"
USERNAME = "admin"
PASSWORD = "password"

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# --- Prompt Utils ---
def build_prompt(item, fmt, tokenizer):
    # If item has 'messages', use chat template logic (Phase 2)
    if "messages" in item:
        # For evaluation, we need the prompt part (Role: User).
        # The 'messages' list in dataset contains User and Assistant.
        # We take the User message content.
        user_msg = ""
        for m in item["messages"]:
            if m["role"] == "user":
                user_msg = m["content"]
                break
        
        # Construct prompt based on format
        if fmt == "gemma":
            return f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
        elif fmt == "phi3":
            return f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"
        else:
            return user_msg # Fallback

    # Fallback for Phase 1 (instruction field)
    instruction = item.get("instruction", "")
    if fmt == "gemma":
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
    elif fmt == "phi3":
        return f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
    return instruction

# --- Model Utils ---
def load_model(base_model_id, adapter_path):
    print(f"Loading base model: {base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN")
    )
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=os.environ.get("HF_TOKEN"))
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# --- WAF Test Logic ---
async def get_user_token(client, url):
    try:
        r = await client.get(url, timeout=5.0)
        m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
        return m.group(1) if m else None
    except: return None

async def login_dvwa(client):
    print("[+] Logging into DVWA...")
    try:
        token = await get_user_token(client, f"{DVWA_BASE_URL}/login.php")
        if not token: return False
        data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
        resp = await client.post(f"{DVWA_BASE_URL}/login.php", data=data, follow_redirects=True, timeout=10.0)
        return "login.php" not in str(resp.url)
    except: return False

async def test_single_payload_waf(client, payload, attack_type):
    target_url = ""
    param_name = ""
    params = {"Submit": "Submit"}
    
    if "SQLI" in attack_type.upper():
        target_url = DVWA_SQLI_URL; param_name = "id"
    elif "XSS" in attack_type.upper():
        target_url = DVWA_XSS_REFLECTED_URL; param_name = "name"
    elif "OS_INJECTION" in attack_type.upper():
        target_url = f"{DVWA_BASE_URL}/vulnerabilities/exec/"; param_name = "ip"
    else: return "unsupported"

    try:
        request_params = params.copy(); request_params[param_name] = payload
        r = await client.get(target_url, params=request_params, timeout=10.0, follow_redirects=True)
        if r.status_code == 403: return "blocked"
        
        resp = r.text.lower()
        if "sqli" in attack_type.lower():
            if "first name" in resp and "error" not in resp: return "passed"
            if "error" in resp: return "sql_error_bypass"
        elif "xss" in attack_type.lower():
            if payload.lower() in resp: return "reflected_no_exec" 
            
        return "failed_waf_filter"
    except: return "error"

# --- Main Evaluation ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--format", choices=["gemma", "phi3"], required=True)
    parser.add_argument("--num_samples", type=int, default=100) # Default reduced to 100
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl")
    args = parser.parse_args()

    cmd_str = f"python scripts/evaluate_model.py --adapter {args.adapter}"

    try:
        # Load Model
        model, tokenizer = load_model(args.base_model, args.adapter)
        
        # Load Data
        ds = load_dataset("json", data_files=args.dataset, split="train")
        ds = ds.select(range(min(args.num_samples, len(ds))))
        
        # Prepare prompts
        prompts = [build_prompt(item, args.format, tokenizer) for item in ds]
        
        # Generation Loop (Batched)
        generated_payloads = []
        print(f"Generating payloads with batch size {args.batch_size}...")
        
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[i : i + args.batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode batch
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            for j, raw_text in enumerate(decoded):
                # Strip prompt
                if args.format == "gemma":
                    payload = raw_text.split("<start_of_turn>model")[-1].replace("<end_of_turn>", "").replace("<eos>", "").strip()
                elif args.format == "phi3":
                    payload = raw_text.split("<|assistant|>")[-1].replace("<|end|>", "").replace(tokenizer.eos_token, "").strip()
                else:
                    payload = raw_text
                generated_payloads.append(payload)

        # WAF Testing Loop (Async)
        print("Testing payloads against WAF...")
        async def run_waf_tests():
            results = []
            waf_summary = Counter()
            async with httpx.AsyncClient(timeout=30.0) as client:
                if not await login_dvwa(client):
                    print("Skipping WAF tests due to login failure.")
                    return [], waf_summary

                for idx, payload in enumerate(tqdm(generated_payloads)):
                    attack_type = ds[idx].get("attack_type", "XSS")
                    status = await test_single_payload_waf(client, payload, attack_type)
                    
                    # Extract history if available for logging
                    history = ds[idx].get("payload_history", [])
                    
                    results.append({
                        "id": idx,
                        "generated_payload": payload,
                        "status": status,
                        "attack_type": attack_type,
                        "history_len": len(history)
                    })
                    waf_summary[status] += 1
            return results, waf_summary

        results, summary = asyncio.run(run_waf_tests())

        # Save & Report
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for r in results: f.write(json.dumps(r) + '\n')
        
        print("\n--- Summary ---")
        total = len(generated_payloads)
        for k, v in summary.most_common():
            print(f"{k}: {v} ({v/total:.2%})")
        
        passed = summary["passed"] + summary["sql_error_bypass"]
        print(f"Total Bypass: {passed/total:.2%}")
        
        log_message(cmd_str, "OK", args.output_file)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        log_message(cmd_str, "FAIL", str(e))

if __name__ == "__main__":
    main()