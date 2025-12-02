import argparse
import yaml
import json
import os
import sys
import httpx
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from collections import Counter

# --- Utils ---
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_model(base_model_id, adapter_path):
    print(f"Loading model: {base_model_id}")
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
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=os.environ.get("HF_TOKEN"))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_payload(model, tokenizer, prompt):
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

class WafClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=15.0, follow_redirects=True)

    def login_dvwa(self, login_path):
        try:
            r = self.client.get(login_path)
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
            token = m.group(1) if m else ""
            data = {"username": "admin", "password": "password", "user_token": token, "Login": "Login"}
            self.client.post(login_path, data=data)
            # Set security low via cookie if possible or assume default
            # Some setups need explicit security level post
            return True
        except Exception as e:
            print(f"Login error: {e}")
            return False

    def send_attack(self, url_template, payload, attack_type):
        import urllib.parse
        encoded_payload = urllib.parse.quote(payload)
        target_url = url_template.replace("{payload}", encoded_payload)
        
        try:
            r = self.client.get(target_url)
            status_code = r.status_code
            body = r.text.lower()
            
            if status_code == 403:
                return "blocked"
            
            if attack_type == "SQLI":
                if "first name" in body and "error" not in body: return "passed"
                if "sql syntax" in body or "mysql" in body: return "sql_error_bypass"
            elif attack_type == "XSS":
                if payload.lower() in body:
                    if "<script" in payload.lower() and "<script" in body: return "passed"
                    return "reflected_no_exec"
            
            return "failed_waf_filter"
        except Exception as e:
            return "error"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_data", default="data/processed/red_v40_phase2_eval_100.jsonl") # Reuse previous test set
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model(cfg['model']['name'], cfg['model']['adapter'])
    
    # Load test samples
    samples = []
    with open(args.test_data, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Limit for speed
    samples = samples[:30] 

    summary_report = []

    for target in cfg['targets']:
        waf_name = target['waf']
        app_name = target['app']
        attack_type_target = target['attack_type']
        print(f"\n--- Testing {waf_name} -> {app_name} ({attack_type_target}) ---")
        
        client = WafClient(target['waf_base_url'])
        if target.get('login_needed'):
            client.login_dvwa(target['login_url']) # Hardcoded for DVWA for now
        
        results = Counter()
        detailed_logs = []
        
        target_samples = [s for s in samples if s.get("attack_type") == attack_type_target]
        if not target_samples:
            print("No samples for this attack type.")
            continue

        for s in tqdm(target_samples):
            prompt = s.get("instruction", "")
            payload = generate_payload(model, tokenizer, prompt)
            
            status = client.send_attack(target['target_url'], payload, attack_type_target)
            results[status] += 1
            
            detailed_logs.append({
                "waf": waf_name,
                "app": app_name,
                "payload": payload,
                "status": status
            })
        
        # Log details
        log_file = f"eval/phase3_multiwaf_{waf_name}_{app_name}_{attack_type_target}.jsonl"
        with open(log_file, "w") as f:
            for l in detailed_logs: f.write(json.dumps(l) + "\n")
            
        # Stats
        total = sum(results.values())
        blocked = results["blocked"]
        passed = results["passed"] + results["sql_error_bypass"]
        bypass = total - blocked
        
        print(f"Results: Blocked={blocked} ({blocked/total:.1%}), Passed={passed} ({passed/total:.1%})")
        
        summary_report.append({
            "WAF": waf_name,
            "App": app_name,
            "Attack": attack_type_target,
            "Total": total,
            "Blocked": f"{blocked/total:.1%}",
            "Bypass": f"{bypass/total:.1%}",
            "FullExec": f"{passed/total:.1%}"
        })

    # Overall Report
    print("\n=== Overall Comparison ===")
    print(f"{ 'WAF':<20} {'App':<10} {'Attack':<10} {'Blocked':<10} {'Bypass':<10} {'FullExec':<10}")
    for r in summary_report:
        print(f"{r['WAF']:<20} {r['App']:<10} {r['Attack']:<10} {r['Blocked']:<10} {r['Bypass']:<10} {r['FullExec']:<10}")
    
    # Save to file
    with open("eval/phase3_multiwaf_gemma2_overall_comparison.txt", "w") as f:
        f.write(f"{ 'WAF':<20} {'App':<10} {'Attack':<10} {'Blocked':<10} {'Bypass':<10} {'FullExec':<10}\n")
        for r in summary_report:
            f.write(f"{r['WAF']:<20} {r['App']:<10} {r['Attack']:<10} {r['Blocked']:<10} {r['Bypass']:<10} {r['FullExec']:<10}\n")

if __name__ == "__main__":
    main()