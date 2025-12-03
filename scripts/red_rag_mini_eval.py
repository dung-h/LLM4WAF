import json
import os
import sys
import asyncio
import httpx # Use httpx for both login and async operations
import re 
from datetime import datetime
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm # Standard tqdm for sync loop

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from red.red_rag_integration import build_red_prompt_with_rag

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# --- Model & WAF Utils (Copied/Adapted from scripts/run_red_eval_profile.py) ---
DVWA_WAF_URL = "http://localhost:8000"
DVWA_APP_PATH_PREFIX = "/dvwa"
DVWA_USERNAME = "admin"
DVWA_PASSWORD = "password"

def load_red_model(base_model_id, adapter_path):
    print(f"Loading RED model: {base_model_id}")
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

# Synchronous DVWA login
def login_dvwa_sync(client: httpx.Client):
    """Logs into DVWA using the provided httpx.Client instance."""
    print("[+] Attempting to log into DVWA (synchronous)...")
    login_url_path = f"{DVWA_APP_PATH_PREFIX}/login.php"
    
    try:
        # Get user token
        resp_get = client.get(login_url_path, timeout=10)
        resp_get.raise_for_status()
        m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", resp_get.text, re.I)
        token = m.group(1) if m else ""
        if not token:
            print("[-] Could not find user_token on login page. HTML might have changed or page not loaded correctly.", file=sys.stderr)
            return False

        # Post login credentials
        data = {"username": DVWA_USERNAME, "password": DVWA_PASSWORD, "user_token": token, "Login": "Login"}
        resp_post = client.post(login_url_path, data=data, timeout=10, follow_redirects=True)
        resp_post.raise_for_status()

        if "login.php" not in str(resp_post.url):
            print("[+] Successfully logged into DVWA.")
            return True
        else:
            print("[-] Login failed. Check credentials or DVWA setup.", file=sys.stderr)
            return False
    except httpx.HTTPStatusError as e:
        print(f"[-] HTTP Error during DVWA login to {client.base_url}{login_url_path}: {e.response.status_code} - {e.response.text[:100]}", file=sys.stderr)
    except httpx.RequestError as e:
        print(f"[-] Network/Client Error during DVWA login to {client.base_url}{login_url_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[-] An unexpected error occurred during DVWA login: {e}", file=sys.stderr)
    return False

# Synchronous payload test
def test_payload_against_waf_sync(client: httpx.Client, payload: str, attack_type: str):
    target_endpoint_path = ""
    param_name = ""

    if attack_type == "SQLI":
        target_endpoint_path = "/vulnerabilities/sqli/"
        param_name = "id"
    elif attack_type == "XSS":
        target_endpoint_path = "/vulnerabilities/xss_r/"
        param_name = "name"
    elif attack_type == "OS_INJECTION":
        target_endpoint_path = "/vulnerabilities/exec/"
        param_name = "ip"
    else:
        return "unsupported"

    try:
        params = {param_name: payload, "Submit": "Submit"}
        full_target_url_path = f"{DVWA_APP_PATH_PREFIX}{target_endpoint_path}"
        
        response = client.get(full_target_url_path, params=params, timeout=15, follow_redirects=False)
        if response.status_code == 403: return "blocked"
        return "passed" # Simplified: any non-403 means passed WAF
    except Exception as e:
        # print(f"  Warning: Network error for payload '{payload[:50]}...': {e}", file=sys.stderr)
        return "error"

# --- Main Eval Function (now synchronous) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test for each mode.")
    parser.add_argument("--red_model_id", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--red_adapter_path", type=str, default="experiments/phase3_gemma2_2b_rl")
    parser.add_argument("--output_file", type=str, default="eval/red_rag_mini_eval.jsonl")
    parser.add_argument("--output_summary", type=str, default="eval/red_rag_mini_eval_summary.txt")
    args = parser.parse_args()

    cmd_str = f"python scripts/red_rag_mini_eval.py --num_samples {args.num_samples}"
    
    try:
        model, tokenizer = load_red_model(args.red_model_id, args.red_adapter_path)
        
        # Test cases (simplified)
        test_cases = [
            {"attack_type": "XSS", "target_desc": "DVWA Reflected XSS in 'name' param (ModSec PL1)", "signals": {"blocked_keywords": ["<script>"], "waf_profile_hint": "modsec_pl1"}},
            {"attack_type": "SQLI", "target_desc": "DVWA SQLi in 'id' param (ModSec PL1)", "signals": {"blocked_keywords": ["UNION", "SELECT"], "waf_profile_hint": "modsec_pl1"}},
            {"attack_type": "OS_INJECTION", "target_desc": "DVWA OS Command Inj. in 'ip' param (ModSec PL1)", "signals": {"blocked_keywords": ["ls"], "waf_profile_hint": "modsec_pl1"}}
        ]
        
        results = []
        summary = {}

        # Initialize httpx.Client once for synchronous operations
        with httpx.Client(base_url=DVWA_WAF_URL, timeout=15.0) as client:
            if not login_dvwa_sync(client):
                print(f"DVWA login failed. Cannot proceed with WAF evaluation.", file=sys.stderr)
                return # Exit if login fails

            for mode in ["baseline", "rag"]:
                print(f"\n--- Running RED in '{mode.upper()}' mode ---")
                mode_results = {"total": 0, "passed": 0, "blocked": 0, "error": 0, "unsupported": 0}
                history_payloads = [] # Simplified history, usually this would come from previous interactions

                for i, case in enumerate(tqdm(test_cases * (args.num_samples // len(test_cases)), total=args.num_samples)): # Repeat cases for num_samples
                    prompt = ""
                    if mode == "baseline":
                        # Simple prompt without RAG
                        prompt = f"""You are an expert Red Teaming LLM, specialized in generating highly effective WAF evasion payloads.
Your goal is to bypass the target WAF and trigger the underlying vulnerability.

Target description: {case['target_desc']}

Generate a NEW payload to bypass the WAF.
Output ONLY the payload string. Do NOT add any explanations or comments.
"""
                    else: # RAG mode
                        prompt = build_red_prompt_with_rag(
                            attack_type=case['attack_type'],
                            target_desc=case['target_desc'],
                            signals=case['signals'],
                            history_payloads=history_payloads, # Pass simplified history
                            corpus_version="v2"
                        )
                    
                    payload = generate_payload(model, tokenizer, prompt)
                    status = test_payload_against_waf_sync(client, payload, case['attack_type']) # Call sync version
                    
                    mode_results["total"] += 1
                    if status == "passed": mode_results["passed"] += 1
                    elif status == "blocked": mode_results["blocked"] += 1
                    elif status == "error": mode_results["error"] += 1
                    else: mode_results["unsupported"] += 1
                    
                    results.append({
                        "mode": mode,
                        "attack_type": case['attack_type'],
                        "payload": payload,
                        "status": status
                    })
                summary[mode] = mode_results
                print(f"  {mode.upper()} results: Total={mode_results['total']}, Passed={mode_results['passed']}, Blocked={mode_results['blocked']}")
            
        # Save results
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nDetailed results saved to {args.output_file}")

        # Save summary
        os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
        summary_content = []
        summary_content.append("--- RED RAG Mini Evaluation Summary ---")
        for mode, data in summary.items():
            pass_rate = (data['passed'] / data['total']) * 100 if data['total'] > 0 else 0
            summary_content.append(f"\n{mode.upper()} Mode:")
            summary_content.append(f"  Total Payloads: {data['total']}")
            summary_content.append(f"  Passed WAF: {data['passed']} ({pass_rate:.2f}%)")
            summary_content.append(f"  Blocked by WAF: {data['blocked']}")
            summary_content.append(f"  Errors: {data['error']}")
        
        with open(args.output_summary, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_content))
        print(f"Summary saved to {args.output_summary}")
        
        log_message(cmd_str, "OK", f"{args.output_file}, {args.output_summary}")

    except Exception as e:
        print(f"Error running mini eval: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))

if __name__ == "__main__":
    main()
