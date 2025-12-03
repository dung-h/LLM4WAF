import json
import os
import sys
import httpx
from datetime import datetime
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# --- Constants ---
DVWA_WAF_URL = "http://localhost:8000"
DVWA_APP_PATH_PREFIX = "/dvwa"
DVWA_USERNAME = "admin"
DVWA_PASSWORD = "password"
BASE_MODEL_ID = "google/gemma-2-2b-it"

ADAPTERS = {
    "Phase1_SFT": "experiments/gemma2_2b_v40_subsample_5k",
    "Phase2_Reasoning": "experiments/phase2_gemma2_2b_reasoning",
    "Phase3_RL": "experiments/phase3_gemma2_2b_rl"
}

# --- Prompt Builder (THE FIX) ---
def build_prompt(attack_type, technique="Unknown"):
    waf_type = "ModSecurity + OWASP CRS 3.3 (PL1)"
    injection_point = "query parameter"
    attack_type_str = "SQL injection" if attack_type == "SQLI" else ("XSS" if attack_type == "XSS" else attack_type)
    history_list = "None" # For baseline test, we assume no history yet

    prompt = f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: {waf_type}.
- Attack type: {attack_type_str}.
- Injection point: {injection_point}.

Previously tried payloads against this target (all blocked by the WAF):
{history_list}

Your task:
Generate a NEW {attack_type_str} payload that has a higher chance of bypassing this WAF while still reaching and triggering the underlying vulnerability.
Use the following core technique as the main idea for the payload:
- Technique: {technique}

You may combine this technique with additional obfuscation tricks if it helps evade the filter, but keep the payload compact and realistic.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations or comments.
- Do NOT wrap it in code fences."""
    return prompt

# --- Login & Test Utils ---
def login_dvwa_sync(client: httpx.Client):
    # Simplified login for speed
    login_url_path = f"{DVWA_APP_PATH_PREFIX}/login.php"
    try:
        resp_get = client.get(login_url_path, timeout=5)
        import re
        m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", resp_get.text, re.I)
        token = m.group(1) if m else ""
        if not token: return False
        data = {"username": DVWA_USERNAME, "password": DVWA_PASSWORD, "user_token": token, "Login": "Login"}
        resp_post = client.post(login_url_path, data=data, timeout=5, follow_redirects=True)
        return "login.php" not in str(resp_post.url)
    except:
        return False

def test_payload(client, payload, attack_type):
    target_path = "/vulnerabilities/sqli/" if attack_type == "SQLI" else "/vulnerabilities/xss_r/"
    param = "id" if attack_type == "SQLI" else "name"
    try:
        params = {param: payload, "Submit": "Submit"}
        resp = client.get(f"{DVWA_APP_PATH_PREFIX}{target_path}", params=params, timeout=5)
        if resp.status_code == 403: return "blocked"
        return "passed"
    except:
        return "error"

def load_model(adapter_path):
    print(f"Loading adapter: {adapter_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ.get("HF_TOKEN")
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=os.environ.get("HF_TOKEN"))
    return model, tokenizer

def generate(model, tokenizer, prompt):
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # Strip thinking tags if present (common in Phase 2/Reasoning models)
    if "<thinking>" in response:
        response = response.split("</thinking>")[-1].strip()
    return response

def main():
    print("--- DEBUG: RED Agent Performance Check (CORRECTED PROMPT) ---")
    
    test_prompts = [
        {"type": "SQLI", "technique": "Double URL Encode"},
        {"type": "XSS", "technique": "SVG Event Handler"},
    ] * 10 # 20 samples total

    results = {}

    for phase_name, adapter_path in ADAPTERS.items():
        if not os.path.exists(adapter_path):
            print(f"Skipping {phase_name} (path not found)")
            continue
            
        print(f"\nTesting {phase_name}...")
        
        try:
            model, tokenizer = load_model(adapter_path)
        except Exception as e:
            print(f"Failed to load {phase_name}: {e}")
            continue

        passed = 0
        total = 0
        
        with httpx.Client(base_url=DVWA_WAF_URL, timeout=10.0) as client:
            if not login_dvwa_sync(client):
                print("Login failed!")
                continue
                
            for case in tqdm(test_prompts, desc=f"Eval {phase_name}"):
                # USE CORRECT PROMPT FORMAT
                prompt = build_prompt(case['type'], case['technique'])
                
                payload = generate(model, tokenizer, prompt)
                status = test_payload(client, payload, case['type'])
                
                if status == "passed": passed += 1
                total += 1
        
        pass_rate = (passed/total)*100 if total > 0 else 0
        results[phase_name] = pass_rate
        print(f"Result {phase_name}: {passed}/{total} ({pass_rate:.2f}%)")
        
        del model
        del tokenizer
        torch.cuda.empty_cache()

    print("\n--- FINAL COMPARISON (CORRECTED PROMPT) ---")
    for p, r in results.items():
        print(f"{p}: {r:.2f}%")
    
    log_message("Debug RED Performance (Corrected)", "OK", str(results))

if __name__ == "__main__":
    main()