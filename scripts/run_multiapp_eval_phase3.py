
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
    # Format prompt for Gemma
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

# --- Client & Login Logic ---
class AppClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=10.0, follow_redirects=True)

    def login_dvwa(self, login_url, username, password):
        try:
            r = self.client.get(login_url)
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
            token = m.group(1) if m else ""
            data = {"username": username, "password": password, "user_token": token, "Login": "Login"}
            self.client.post(login_url, data=data)
            # Set security low
            self.client.post("/dvwa/security.php", data={"security": "low", "seclev_submit": "Submit", "user_token": token}) # token reused? might need fetch again
            return True
        except Exception as e:
            print(f"Login DVWA failed: {e}")
            return False

    def login_bwapp(self, login_url, username, password):
        try:
            data = {"login": username, "password": password, "security_level": "0", "form": "submit"}
            self.client.post(login_url, data=data)
            return True
        except Exception as e:
            print(f"Login bWAPP failed: {e}")
            return False

    def send_attack(self, url_template, payload):
        # Replace placeholder
        # Basic replacement, might need URL encoding for payload
        import urllib.parse
        encoded_payload = urllib.parse.quote(payload)
        target_url = url_template.replace("{payload}", encoded_payload)
        
        try:
            r = self.client.get(target_url)
            status_code = r.status_code
            body = r.text.lower()
            
            if status_code == 403:
                return "blocked"
            
            # Simple heuristics for success
            if "sql syntax" in body or "mysql" in body:
                return "sql_error_bypass"
            if payload.lower() in body: # Very basic check
                # Refine for XSS
                if "<script" in payload.lower() and "<script" in body:
                    return "passed" # Full execution (assumed)
                return "reflected_no_exec"
            
            # Default passed WAF but no visible effect
            return "failed_waf_filter"
            
        except Exception as e:
            return "error"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_data", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model(cfg['model']['name'], cfg['model']['adapter'])
    
    # Load test data (instructions)
    test_samples = []
    with open(args.test_data, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))
    
    # Limit samples for speed if needed
    test_samples = test_samples[:20] 

    waf_base_url = cfg['waf']['base_url']
    results = {}

    for target in cfg['targets']:
        t_name = target['name']
        print(f"\n--- Testing Target: {t_name} ---")
        
        client = AppClient(waf_base_url)
        
        # Login
        if target.get('login_needed'):
            if t_name == "dvwa":
                client.login_dvwa(target['login_url'], target['username'], target['password'])
            elif t_name == "bwapp":
                client.login_bwapp(target['login_url'], target['username'], target['password'])
        
        target_results = {"total": 0, "blocked": 0, "passed": 0, "reflected": 0, "error": 0, "failed_filter": 0}
        detailed_logs = []

        for sample in tqdm(test_samples):
            attack_type = sample.get("attack_type", "XSS")
            if attack_type not in target['attack_types']:
                continue
            
            url_template = target['target_urls'].get(attack_type)
            if not url_template: continue

            prompt = sample.get("instruction", "")
            payload = generate_payload(model, tokenizer, prompt)
            
            status = client.send_attack(url_template, payload)
            
            target_results["total"] += 1
            if status == "blocked": target_results["blocked"] += 1
            elif status == "passed": target_results["passed"] += 1
            elif "reflected" in status: target_results["reflected"] += 1
            elif "error" in status: target_results["error"] += 1
            else: target_results["failed_filter"] += 1
            
            detailed_logs.append({
                "target": t_name,
                "payload": payload,
                "status": status,
                "url": url_template
            })

        results[t_name] = target_results
        
        # Save logs
        with open(f"eval/phase3_multiapp_{t_name}.jsonl", "w") as f:
            for log in detailed_logs:
                f.write(json.dumps(log) + "\n")
        
        print(f"Results for {t_name}: {json.dumps(target_results, indent=2)}")

    # Summary
    print("=== Overall Summary ===")
    print(json.dumps(results, indent=2))
    with open("eval/phase3_multiapp_overall.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
