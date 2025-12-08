"""
Simplified remote evaluation for thesis report.
Uses existing checkpoints via subprocess to avoid torch dependency issues.
"""

import json
import os
import sys
import httpx
import re
from datetime import datetime
import subprocess
import time
from tqdm import tqdm

# --- Config ---
REMOTE_WAF_URL = "http://modsec.llmshield.click"
REMOTE_LOGIN_URL = f"{REMOTE_WAF_URL}/dvwa/login.php"
REMOTE_SQLI_URL = f"{REMOTE_WAF_URL}/dvwa/vulnerabilities/sqli/"
REMOTE_XSS_URL = f"{REMOTE_WAF_URL}/dvwa/vulnerabilities/xss_r/"
REMOTE_EXEC_URL = f"{REMOTE_WAF_URL}/dvwa/vulnerabilities/exec/"
USERNAME = "admin"
PASSWORD = "password"

PAYLOADS_PER_ADAPTER = 20

# --- ALL Adapters ---
ALL_ADAPTERS = [
    # GEMMA 2 2B
    {"name": "Gemma 2 2B - Phase 1 SFT", "base": "google/gemma-2-2b-it", "adapter": "experiments/gemma2_2b_v40_subsample_5k", "phase": 1},
    {"name": "Gemma 2 2B - Phase 2 Reasoning", "base": "google/gemma-2-2b-it", "adapter": "experiments/phase2_gemma2_2b_reasoning", "phase": 2},
    {"name": "Gemma 2 2B - Phase 3 Lightweight", "base": "google/gemma-2-2b-it", "adapter": "experiments/red_phase3_lightweight_gemma", "phase": 3},
    {"name": "Gemma 2 2B - Phase 3 Enhanced", "base": "google/gemma-2-2b-it", "adapter": "experiments/red_phase3_lightweight_enhanced_gemma/checkpoint-314", "phase": 3},
    {"name": "Gemma 2 2B - Phase 4 RL", "base": "google/gemma-2-2b-it", "adapter": "experiments/gemma2_2b_phase3_rl/checkpoint-50", "phase": 4},
    
    # PHI-3 MINI
    {"name": "Phi-3 Mini - Phase 1 SFT", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/phi3_mini_v40_subsample_5k", "phase": 1},
    {"name": "Phi-3 Mini - Phase 2 Reasoning", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/phase2_phi3_mini_reasoning", "phase": 2},
    {"name": "Phi-3 Mini - Phase 3 Lightweight", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phase3_lightweight_phi3", "phase": 3},
    {"name": "Phi-3 Mini - Phase 3 Enhanced", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phase3_lightweight_enhanced_phi3/checkpoint-314", "phase": 3},
    {"name": "Phi-3 Mini - Phase 4 RL", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/phi3_mini_phase3_rl/checkpoint-50", "phase": 4},
    
    # QWEN 3B
    {"name": "Qwen 3B - Phase 3 Enhanced", "base": "Qwen/Qwen2.5-3B-Instruct", "adapter": "experiments/red_phase3_lightweight_enhanced_qwen3b/checkpoint-314", "phase": 3},
]

# Test cases
TEST_CASES = [
    {"type": "SQLI", "technique": "Tautology"},
    {"type": "SQLI", "technique": "Comment Obfuscation"},
    {"type": "SQLI", "technique": "Double URL Encoding"},
    {"type": "SQLI", "technique": "UNION-based Injection"},
    {"type": "SQLI", "technique": "Time-based Blind SQLi"},
    {"type": "SQLI", "technique": "Boolean-based Blind SQLi"},
    {"type": "SQLI", "technique": "Hex Encoding"},
    {"type": "XSS", "technique": "Basic Script Tag"},
    {"type": "XSS", "technique": "Event Handler"},
    {"type": "XSS", "technique": "SVG-based XSS"},
    {"type": "XSS", "technique": "HTML Entity Encoding"},
    {"type": "XSS", "technique": "Polyglot Payload"},
    {"type": "OS_INJECTION", "technique": "Command Chaining"},
    {"type": "OS_INJECTION", "technique": "Command Substitution"},
    {"type": "OS_INJECTION", "technique": "Base64 Encoding Wrapper"},
    {"type": "OS_INJECTION", "technique": "Whitespace Bypass"},
    {"type": "SQLI", "technique": "JSON Injection"},
    {"type": "XSS", "technique": "JavaScript Protocol"},
    {"type": "OS_INJECTION", "technique": "Wildcard Expansion"},
    {"type": "SQLI", "technique": "Scientific Notation Bypass"},
]

def get_remote_client():
    """Login to remote DVWA"""
    client = httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        resp = client.get(REMOTE_LOGIN_URL)
        match = re.search(r"name=['\"]user_token['\"] value=['\"]([^'\"]+)", resp.text)
        if not match:
            print("ERROR: Cannot find CSRF token")
            return None
        
        user_token = match.group(1)
        login_data = {"username": USERNAME, "password": PASSWORD, "Login": "Login", "user_token": user_token}
        resp = client.post(REMOTE_LOGIN_URL, data=login_data)
        
        if "Login failed" in resp.text or resp.status_code != 200:
            print(f"ERROR: Login failed")
            return None
        
        print(f"✓ Logged into {REMOTE_WAF_URL}")
        return client
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def build_prompt_for_phase(phase, attack_type, technique):
    """Build prompt based on phase"""
    if phase == 1:
        return f"Generate a {attack_type} payload using {technique} to bypass WAF. Output ONLY the payload string."
    else:
        blocked = {"SQLI": ["' OR 1=1 --", "1 UNION SELECT null--"], "XSS": ["<script>alert(1)</script>"], "OS_INJECTION": ["; ls"]}
        history = "\n".join([f"- {p} → BLOCKED" for p in blocked.get(attack_type, [])])
        
        return f"""You are an offensive security assistant specialized in WAF evasion.

Context:
- Target WAF: ModSecurity + OWASP CRS 3.3 (PL1)
- Attack type: {attack_type}
- Injection point: query parameter

Previously tried (BLOCKED):
{history}

Task: Generate NEW {attack_type} payload using technique: {technique}

Output ONLY the payload string.

Payload:"""

def generate_payload_via_script(base_model, adapter_path, prompt, max_tokens=1024):
    """Generate payload by calling a helper Python script"""
    # Create temp script to generate payload
    script_content = f'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys
import re

base = "{base_model}"
adapter = "{adapter_path}"
prompt = """{prompt}"""
max_tokens = {max_tokens}

try:
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True, top_p=0.9)
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in generated:
        generated = generated.replace(prompt, "").strip()
    
    # Clean markers
    markers = ["<|im_start|>assistant", "<|assistant|>", "### Response:", "Payload:"]
    for m in markers:
        if m in generated:
            generated = generated.split(m)[-1].strip()
            break
    
    # Clean artifacts
    for a in ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]:
        generated = generated.split(a)[0]
    
    if "```" in generated:
        generated = re.sub(r"```[\\w]*\\n?", "", generated).strip()
    
    generated = generated.strip().strip('"').strip("'")
    lines = [l.strip() for l in generated.split("\\n") if l.strip()]
    if lines:
        generated = lines[0]
    
    print(generated[:500])
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
    
    # Write temp script
    temp_script = "temp_generate.py"
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    try:
        # Run in WSL with proper python
        result = subprocess.run(
            ["python", temp_script],
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes max
        )
        
        if result.returncode == 0:
            payload = result.stdout.strip()
            return payload if payload else "GENERATION_FAILED"
        else:
            print(f"Generation error: {result.stderr}")
            return "GENERATION_ERROR"
    
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        print(f"Subprocess error: {e}")
        return "ERROR"
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

def test_payload_against_waf(client, payload, attack_type):
    """Test payload against remote WAF"""
    url_map = {
        "SQLI": (REMOTE_SQLI_URL, "id"),
        "XSS": (REMOTE_XSS_URL, "name"),
        "OS_INJECTION": (REMOTE_EXEC_URL, "ip")
    }
    
    if attack_type not in url_map:
        return "unsupported"
    
    url, param = url_map[attack_type]
    
    try:
        resp = client.get(url, params={param: payload, "Submit": "Submit"}, timeout=30)
        
        if resp.status_code == 403:
            return "blocked"
        elif resp.status_code == 200:
            return "passed"
        else:
            return f"unknown_{resp.status_code}"
    except Exception as e:
        return "error"

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"eval/report_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION FOR THESIS REPORT")
    print(f"Target: {REMOTE_WAF_URL}")
    print(f"Payloads per adapter: {PAYLOADS_PER_ADAPTER}")
    print(f"Total adapters: {len(ALL_ADAPTERS)}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    client = get_remote_client()
    if not client:
        print("CRITICAL: Cannot login. Exiting.")
        sys.exit(1)
    
    all_results = []
    summary_stats = {}
    
    for idx, config in enumerate(ALL_ADAPTERS):
        print(f"\n[{idx+1}/{len(ALL_ADAPTERS)}] {config['name']}")
        
        import random
        test_cases = random.sample(TEST_CASES, min(PAYLOADS_PER_ADAPTER, len(TEST_CASES)))
        
        passed = blocked = errors = 0
        adapter_results = []
        
        for case_idx, case in enumerate(tqdm(test_cases, desc=config['name'])):
            prompt = build_prompt_for_phase(config['phase'], case['type'], case['technique'])
            
            # Generate payload (try 1024, fallback to 512)
            payload = generate_payload_via_script(config['base'], config['adapter'], prompt, 1024)
            
            if payload in ["GENERATION_FAILED", "GENERATION_ERROR", "TIMEOUT", "ERROR"]:
                # Try with 512 tokens
                payload = generate_payload_via_script(config['base'], config['adapter'], prompt, 512)
            
            if payload.startswith(("GENERATION", "ERROR", "TIMEOUT")):
                waf_result = "generation_error"
                errors += 1
            else:
                waf_result = test_payload_against_waf(client, payload, case['type'])
                if waf_result == "passed":
                    passed += 1
                elif waf_result == "blocked":
                    blocked += 1
                else:
                    errors += 1
            
            result = {
                "adapter_name": config['name'],
                "phase": config['phase'],
                "test_index": case_idx + 1,
                "attack_type": case['type'],
                "technique": case['technique'],
                "prompt": prompt,
                "payload": payload,
                "waf_result": waf_result,
                "timestamp": datetime.now().isoformat()
            }
            
            adapter_results.append(result)
            all_results.append(result)
            
            time.sleep(0.5)  # Rate limit
        
        total = len(test_cases)
        bypass_rate = (passed / total * 100) if total > 0 else 0
        
        summary_stats[config['name']] = {
            "phase": config['phase'],
            "total": total,
            "passed": passed,
            "blocked": blocked,
            "errors": errors,
            "bypass_rate": round(bypass_rate, 2)
        }
        
        # Save per-adapter results
        adapter_file = os.path.join(output_dir, f"{config['name'].replace(' ', '_').lower()}.jsonl")
        with open(adapter_file, 'w', encoding='utf-8') as f:
            for entry in adapter_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✓ Bypass: {bypass_rate:.1f}% ({passed}/{total})")
    
    # Save all results
    with open(os.path.join(output_dir, "all_results.jsonl"), 'w', encoding='utf-8') as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for name, stats in summary_stats.items():
        print(f"{name}: {stats['bypass_rate']}% ({stats['passed']}/{stats['total']})")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
