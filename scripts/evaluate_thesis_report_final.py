"""
Final comprehensive evaluation script for thesis report.
Uses existing demo infrastructure to generate payloads, then tests on remote WAF.
Saves ALL prompts, payloads, and results for reporting.
"""

import json
import os
import sys
import httpx
import re
from datetime import datetime
from tqdm import tqdm
import time
import random

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import demo infrastructure
from demo.model_loader import model_loader
from demo.prompts import build_structured_prompt

# --- Config ---
REMOTE_WAF_URL = "http://modsec.llmshield.click"
REMOTE_LOGIN_URL = f"{REMOTE_WAF_URL}/dvwa/login.php"
REMOTE_SQLI_URL = f"{REMOTE_WAF_URL}/dvwa/vulnerabilities/sqli/"
REMOTE_XSS_URL = f"{REMOTE_WAF_URL}/dvwa/vulnerabilities/xss_r/"
REMOTE_EXEC_URL = f"{REMOTE_WAF_URL}/dvwa/vulnerabilities/exec/"
USERNAME = "admin"
PASSWORD = "password"

PAYLOADS_PER_ADAPTER = 20

# --- ALL Adapters to Test ---
ALL_ADAPTERS = [
    # GEMMA 2 2B
    {"name": "Gemma_2_2B_Phase1_SFT", "base": "google/gemma-2-2b-it", "adapter": "experiments/gemma2_2b_v40_subsample_5k", "phase": 1},
    {"name": "Gemma_2_2B_Phase2_Reasoning", "base": "google/gemma-2-2b-it", "adapter": "experiments/phase2_gemma2_2b_reasoning", "phase": 2},
    {"name": "Gemma_2_2B_Phase3_Lightweight", "base": "google/gemma-2-2b-it", "adapter": "experiments/red_phase3_lightweight_gemma", "phase": 3},
    {"name": "Gemma_2_2B_Phase3_Enhanced", "base": "google/gemma-2-2b-it", "adapter": "experiments/red_phase3_lightweight_enhanced_gemma/checkpoint-314", "phase": 3},
    {"name": "Gemma_2_2B_Phase4_RL", "base": "google/gemma-2-2b-it", "adapter": "experiments/gemma2_2b_phase3_rl/checkpoint-50", "phase": 4},
    
    # PHI-3 MINI
    {"name": "Phi3_Mini_Phase1_SFT", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/phi3_mini_v40_subsample_5k", "phase": 1},
    {"name": "Phi3_Mini_Phase2_Reasoning", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/phase2_phi3_mini_reasoning", "phase": 2},
    {"name": "Phi3_Mini_Phase3_Lightweight", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phase3_lightweight_phi3", "phase": 3},
    {"name": "Phi3_Mini_Phase3_Enhanced", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phase3_lightweight_enhanced_phi3/checkpoint-314", "phase": 3},
    {"name": "Phi3_Mini_Phase4_RL", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/phi3_mini_phase3_rl/checkpoint-50", "phase": 4},
    
    # QWEN 3B
    {"name": "Qwen_3B_Phase3_Enhanced", "base": "Qwen/Qwen2.5-3B-Instruct", "adapter": "experiments/red_phase3_lightweight_enhanced_qwen3b/checkpoint-314", "phase": 3},
]

# Test cases - 20 diverse techniques
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
        print(f"Connecting to {REMOTE_WAF_URL}...")
        resp = client.get(REMOTE_LOGIN_URL)
        match = re.search(r"name=['\"]user_token['\"] value=['\"]([^'\"]+)", resp.text)
        if not match:
            print("ERROR: Cannot find CSRF token on login page")
            return None
        
        user_token = match.group(1)
        login_data = {
            "username": USERNAME,
            "password": PASSWORD,
            "Login": "Login",
            "user_token": user_token
        }
        resp = client.post(REMOTE_LOGIN_URL, data=login_data)
        
        if "Login failed" in resp.text or resp.status_code != 200:
            print(f"ERROR: Login failed (status={resp.status_code})")
            return None
        
        print(f"✓ Successfully logged into {REMOTE_WAF_URL}")
        return client
    except Exception as e:
        print(f"ERROR during login: {e}")
        return None

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
            return f"unknown_status_{resp.status_code}"
    
    except httpx.TimeoutException:
        return "timeout"
    except Exception as e:
        print(f"WAF test error: {e}")
        return "error"

def generate_payload(model, tokenizer, prompt, max_tokens=1024):
    """Generate payload using loaded model"""
    import torch
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        
        # Clean artifacts
        payload_raw = generated_text.strip()
        
        markers = ["<|im_start|>assistant", "<|start_header_id|>assistant<|end_header_id|>", 
                   "<|assistant|>", "### Response:", "Assistant:", "Payload:"]
        for marker in markers:
            if marker in payload_raw:
                payload_raw = payload_raw.split(marker)[-1].strip()
                break
        
        artifacts = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "<|end|>", "</s>"]
        for artifact in artifacts:
            if artifact in payload_raw:
                payload_raw = payload_raw.split(artifact)[0].strip()
        
        if "```" in payload_raw:
            payload_raw = re.sub(r"```[\w]*\n?", "", payload_raw).strip()
        
        payload_raw = payload_raw.strip('"').strip("'")
        
        lines = [l.strip() for l in payload_raw.split('\n') if l.strip()]
        if lines:
            payload_raw = lines[0]
        
        return payload_raw[:500]
    
    except Exception as e:
        print(f"Generation error: {e}")
        return ""

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"eval/thesis_report_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION FOR THESIS REPORT")
    print(f"Target WAF: {REMOTE_WAF_URL}")
    print(f"Payloads per adapter: {PAYLOADS_PER_ADAPTER}")
    print(f"Total adapters: {len(ALL_ADAPTERS)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Login to remote WAF
    client = get_remote_client()
    if not client:
        print("CRITICAL: Cannot login to remote WAF. Exiting.")
        sys.exit(1)
    
    all_results = []
    summary_stats = {}
    
    for adapter_idx, config in enumerate(ALL_ADAPTERS):
        print(f"\n{'='*80}")
        print(f"[{adapter_idx + 1}/{len(ALL_ADAPTERS)}] {config['name']}")
        print(f"Base: {config['base']}")
        print(f"Adapter: {config['adapter']}")
        print(f"Phase: {config['phase']}")
        print(f"{'='*80}\n")
        
        try:
            # Load model
            print("Loading model...")
            model, tokenizer = model_loader.load_model(config['base'], config['adapter'])
            print("✓ Model loaded\n")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            continue
        
        # Select test cases
        test_cases_selected = random.sample(TEST_CASES, min(PAYLOADS_PER_ADAPTER, len(TEST_CASES)))
        
        passed_count = 0
        blocked_count = 0
        error_count = 0
        adapter_results = []
        
        for case_idx, case in enumerate(tqdm(test_cases_selected, desc=f"Testing {config['name']}")):
            try:
                # Build prompt using demo infrastructure
                prompt = build_structured_prompt(case['type'], case['technique'])
                
                # Generate payload (start with 1024, fallback to 512 on OOM)
                try:
                    payload = generate_payload(model, tokenizer, prompt, max_tokens=1024)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠️ OOM with 1024 tokens, fallback to 512")
                        import torch
                        torch.cuda.empty_cache()
                        payload = generate_payload(model, tokenizer, prompt, max_tokens=512)
                    else:
                        raise
                
                # Test against WAF
                waf_result = test_payload_against_waf(client, payload, case['type'])
                
                # Track stats
                if waf_result == "passed":
                    passed_count += 1
                elif waf_result == "blocked":
                    blocked_count += 1
                else:
                    error_count += 1
                
                # Save detailed result
                result_entry = {
                    "adapter_name": config['name'],
                    "adapter_phase": config['phase'],
                    "adapter_base": config['base'],
                    "adapter_path": config['adapter'],
                    "test_index": case_idx + 1,
                    "attack_type": case['type'],
                    "technique": case['technique'],
                    "prompt_used": prompt,
                    "generated_payload": payload,
                    "waf_result": waf_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                adapter_results.append(result_entry)
                all_results.append(result_entry)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"ERROR on test case {case}: {e}")
                error_count += 1
        
        # Calculate stats
        total_tests = len(test_cases_selected)
        bypass_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        summary_stats[config['name']] = {
            "phase": config['phase'],
            "total_tests": total_tests,
            "passed": passed_count,
            "blocked": blocked_count,
            "errors": error_count,
            "bypass_rate": round(bypass_rate, 2)
        }
        
        # Save per-adapter results
        adapter_file = os.path.join(output_dir, f"{config['name']}_details.jsonl")
        with open(adapter_file, 'w', encoding='utf-8') as f:
            for entry in adapter_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n✓ {config['name']} completed:")
        print(f"  Bypass Rate: {bypass_rate:.1f}% ({passed_count}/{total_tests})")
        print(f"  Saved to: {adapter_file}\n")
        
        # Unload model to free memory
        model_loader.unload_model()
    
    # Save all results
    all_results_file = os.path.join(output_dir, "all_results.jsonl")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED")
    print(f"{'='*80}\n")
    
    for adapter_name, stats in summary_stats.items():
        print(f"{adapter_name}: {stats['bypass_rate']}% ({stats['passed']}/{stats['total_tests']})")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - All results: {all_results_file}")
    print(f"  - Summary: {summary_file}")
    
    # Generate markdown report
    report_md = os.path.join(output_dir, "THESIS_REPORT.md")
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write(f"# Thesis Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Target:** {REMOTE_WAF_URL}\n\n")
        f.write(f"**Payloads per Adapter:** {PAYLOADS_PER_ADAPTER}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Results Summary\n\n")
        f.write(f"| Adapter | Phase | Bypass Rate | Passed/Total | Blocked | Errors |\n")
        f.write(f"|---------|-------|-------------|--------------|---------|--------|\n")
        
        for name, stats in summary_stats.items():
            f.write(f"| {name} | {stats['phase']} | {stats['bypass_rate']}% | {stats['passed']}/{stats['total_tests']} | {stats['blocked']} | {stats['errors']} |\n")
    
    print(f"\n✓ Markdown report: {report_md}\n")

if __name__ == "__main__":
    main()
