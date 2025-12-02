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

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Model & Utils ---
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

def run_eval(config_path, test_data_path, output_prefix="red_phase4", num_samples=50):
    cfg = load_config(config_path)
    model, tokenizer = load_model(cfg['model']['name'], cfg['model']['adapter'])
    
    # Load test data (instructions)
    test_samples = []
    with open(test_data_path, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))
    
    # Limit samples for speed
    test_samples = test_samples[:num_samples] 

    summary_report = []

    for target in cfg['targets']:
        waf_profile_name = target['waf']
        app_name = target['app']
        attack_type_target = target['attack_type']
        
        print(f"\n--- Testing Profile: {waf_profile_name} -> App: {app_name} ({attack_type_target}) ---")
        
        client = WafClient(target['waf_base_url'])
        
        # Login (only if needed and implemented)
        if target.get('login_needed'):
            if app_name == "dvwa":
                client.login_dvwa(target['login_url']) # Assuming login success
            # Add other app login logic here if implemented
        
        target_results = {"total": 0, "blocked": 0, "passed": 0, "reflected": 0, "error": 0, "failed_filter": 0, "sql_error_bypass": 0}
        detailed_logs = []

        # Filter test samples by attack_type
        filtered_samples = [s for s in test_samples if s.get("attack_type") == attack_type_target]
        if not filtered_samples:
            print(f"No samples for {attack_type_target} for {app_name}. Skipping.")
            continue

        for sample in tqdm(filtered_samples):
            prompt = sample.get("instruction", "")
            payload = generate_payload(model, tokenizer, prompt)
            
            status = client.send_attack(target['target_url'], payload, attack_type_target)
            
            target_results["total"] += 1
            if status == "blocked": target_results["blocked"] += 1
            elif status == "passed": target_results["passed"] += 1
            elif status == "reflected_no_exec": target_results["reflected"] += 1
            elif status == "sql_error_bypass": target_results["sql_error_bypass"] += 1
            elif status == "error": target_results["error"] += 1
            else: target_results["failed_filter"] += 1 # Includes "unsupported_attack_type"

            detailed_logs.append({
                "waf_profile": waf_profile_name,
                "app": app_name,
                "attack_type": attack_type_target,
                "payload": payload,
                "status": status,
                "url_template": target['target_url']
            })

        # Save detailed logs
        output_log_file = f"eval/{output_prefix}_{waf_profile_name}_{app_name}_{attack_type_target}.jsonl"
        with open(output_log_file, "w") as f:
            for log_entry in detailed_logs:
                f.write(json.dumps(log_entry) + "\n")
        print(f"  Saved detailed logs to {output_log_file}")
        
        # Calculate summary metrics
        total_requests = target_results["total"]
        blocked_count = target_results["blocked"]
        passed_count = target_results["passed"]
        sql_error_count = target_results["sql_error_bypass"]
        reflected_count = target_results["reflected"]
        failed_filter_count = target_results["failed_filter"]

        blocked_pct = (blocked_count / total_requests) * 100 if total_requests > 0 else 0
        full_exec_pct = ((passed_count + sql_error_count) / total_requests) * 100 if total_requests > 0 else 0
        total_waf_bypass_pct = ((total_requests - blocked_count) / total_requests) * 100 if total_requests > 0 else 0

        summary_text = f"""
  Total requests: {total_requests}
  Blocked: {blocked_count} ({blocked_pct:.2f}%)
  Full Exec: {passed_count + sql_error_count} ({full_exec_pct:.2f}%)
  Total WAF Bypass (not blocked): {total_waf_bypass_pct:.2f}%
  ------------------------------------
  Passed: {passed_count}
  SQL Error Bypass: {sql_error_count}
  Reflected: {reflected_count}
  Failed Filter: {failed_filter_count}
"""
        print(summary_text)

        summary_report.append({
            "waf_profile": waf_profile_name,
            "app": app_name,
            "attack_type": attack_type_target,
            "total_requests": total_requests,
            "blocked_pct": blocked_pct,
            "full_exec_pct": full_exec_pct,
            "total_waf_bypass_pct": total_waf_bypass_pct
        })
    
    # Save overall summary
    overall_summary_file = f"eval/{output_prefix}_overall_summary.json"
    with open(overall_summary_file, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nOverall summary saved to {overall_summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_data", default="data/processed/red_v40_phase2_eval_100.jsonl")
    parser.add_argument("--output_prefix", default="red_phase4")
    parser.add_argument("--num_samples", type=int, default=50) # Adjusted default for quicker runs
    args = parser.parse_args()
    
    run_eval(args.config, args.test_data, args.output_prefix, args.num_samples)
