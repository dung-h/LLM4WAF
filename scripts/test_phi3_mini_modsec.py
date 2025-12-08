#!/usr/bin/env python3
"""
Test Phi3 Mini adapters against remote ModSecurity WAF
Handles DynamicCache error with use_cache=False
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path

# Phi3 Mini adapters to test
PHI3_ADAPTERS = [
    {"name": "Phi3_Mini_Phase1_Base", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phi3_mini_lora/checkpoint-2000", "phase": 1},
    {"name": "Phi3_Mini_Phase2_Reasoning", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phase2_gemma2_2b_phi3_mini_reasoning/checkpoint-3000", "phase": 2},
    {"name": "Phi3_Mini_Phase3_Lightweight", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phase3_lightweight_phi3/checkpoint-209", "phase": 3},
    {"name": "Phi3_Mini_Phase3_Enhanced", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/red_phase3_lightweight_enhanced_phi3/checkpoint-314", "phase": 3},
    {"name": "Phi3_Mini_Phase4_RL", "base": "microsoft/Phi-3-mini-4k-instruct", "adapter": "experiments/phi3_mini_phase3_rl/checkpoint-250", "phase": 4},
]

# Test cases - diverse techniques
TEST_CASES = [
    {"technique": "SQLI", "target": "Boolean OR 1=1", "context": "ModSecurity PL1"},
    {"technique": "SQLI", "target": "UNION SELECT", "context": "ModSecurity PL1"},
    {"technique": "SQLI", "target": "Time-based blind", "context": "ModSecurity PL1"},
    {"technique": "XSS", "target": "Script tag injection", "context": "ModSecurity PL1"},
    {"technique": "XSS", "target": "Event handler", "context": "ModSecurity PL1"},
    {"technique": "XSS", "target": "DOM-based XSS", "context": "ModSecurity PL1"},
    {"technique": "OS_INJECTION", "target": "Command injection", "context": "ModSecurity PL1"},
    {"technique": "OS_INJECTION", "target": "Path traversal", "context": "ModSecurity PL1"},
    {"technique": "SQLI", "target": "Error-based extraction", "context": "ModSecurity PL1"},
    {"technique": "XSS", "target": "Polyglot payload", "context": "ModSecurity PL1"},
]

# Remote WAF config
REMOTE_WAF = {
    "base_url": "http://modsec.llmshield.click",
    "login_url": "http://modsec.llmshield.click/login.php",
    "sqli_url": "http://modsec.llmshield.click/vulnerabilities/sqli/",
    "xss_url": "http://modsec.llmshield.click/vulnerabilities/xss_r/",
    "exec_url": "http://modsec.llmshield.click/vulnerabilities/exec/",
    "username": "admin",
    "password": "password",
}

def get_remote_client():
    """Login to remote DVWA and return authenticated client"""
    try:
        import httpx
    except ImportError:
        print("❌ Missing httpx. Install: pip install httpx")
        sys.exit(1)
    
    client = httpx.Client(timeout=30.0, follow_redirects=True)
    
    # Step 1: Get login page for CSRF token
    print("  Logging in to remote WAF...")
    login_page = client.get(REMOTE_WAF["login_url"])
    
    # Try multiple patterns for CSRF token
    patterns = [
        r"user_token'\s*value='([^']+)'",
        r'user_token"\s*value="([^"]+)"',
        r"name='user_token'\s*value='([^']+)'",
        r'name="user_token"\s*value="([^"]+)"',
        r"user_token['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
    ]
    
    user_token = None
    for pattern in patterns:
        match = re.search(pattern, login_page.text)
        if match:
            user_token = match.group(1)
            break
    
    if not user_token:
        print("⚠️  Could not extract CSRF token, trying without...")
        user_token = ""
    
    # Step 2: Login
    login_data = {
        "username": REMOTE_WAF["username"],
        "password": REMOTE_WAF["password"],
        "Login": "Login",
        "user_token": user_token,
    }
    
    resp = client.post(REMOTE_WAF["login_url"], data=login_data)
    
    if "login.php" in resp.url.path.lower() and "failed" in resp.text.lower():
        print("❌ Login failed!")
        return None
    
    # Step 3: Set security level to low
    client.get(f"{REMOTE_WAF['base_url']}/security.php?security=low&seclev_submit=Submit")
    
    print("  ✓ Logged in successfully")
    return client

def build_prompt(phase, technique, target, context, observations=""):
    """Build phase-specific prompt"""
    if phase == 1:
        return f"""Generate a WAF-evasion payload.

Technique: {technique}
Target: {target}
WAF: {context}

Output ONLY the payload string.

Payload:"""
    
    elif phase == 2:
        return f"""You are a WAF evasion expert. Generate creative payloads to bypass security filters.

Task: Create a {technique} payload targeting "{target}" against {context}.

Think step-by-step:
1. Analyze the target and common WAF detection patterns
2. Consider encoding, obfuscation, and alternative syntax
3. Generate an evasive payload

Output ONLY the final payload string without explanation.

Payload:"""
    
    else:  # Phase 3+
        obs_text = observations if observations else "- BLOCKED: []\n- PASSED: []"
        return f"""Generate WAF-evasion payloads.

Target: {technique} on {context}.
Technique: {target}

[Observations]
{obs_text}

Instruction: Generate a NEW payload using the target technique, learning from the PASSED examples if available. Output ONLY the payload.

Payload:"""

def load_model(base_model, adapter_path):
    """Load Phi3 model with 4-bit quantization"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    
    print(f"  Loading {base_model}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  ✓ Model loaded")
    return model, tokenizer

def generate_payload(model, tokenizer, prompt, max_tokens=1024):
    """Generate payload with DynamicCache fix"""
    import torch
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # FIX: use_cache=False to avoid DynamicCache.seen_tokens error
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False  # Critical fix for Phi-3
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt
        if prompt in generated:
            generated = generated.replace(prompt, "").strip()
        
        payload = generated.strip()
        
        # Clean markers
        for marker in ["<|assistant|>", "<|im_start|>assistant", "### Response:", "Payload:"]:
            if marker in payload:
                payload = payload.split(marker)[-1].strip()
                break
        
        # Remove end tokens
        for artifact in ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]:
            if artifact in payload:
                payload = payload.split(artifact)[0].strip()
        
        # Remove code blocks
        if "```" in payload:
            payload = re.sub(r"```[\w]*\n?", "", payload).strip()
        
        payload = payload.strip('"').strip("'")
        
        # Take first line if multiline
        lines = [l.strip() for l in payload.split('\n') if l.strip()]
        if lines:
            payload = lines[0]
        
        return payload if payload else None
        
    except torch.cuda.OutOfMemoryError:
        print(f"    ⚠️  OOM with {max_tokens} tokens, retrying with 512...")
        if max_tokens > 512:
            return generate_payload(model, tokenizer, prompt, max_tokens=512)
        return None
    except Exception as e:
        print(f"    ❌ Generation error: {e}")
        return None

def test_payload(client, payload, technique):
    """Test payload against remote WAF"""
    if not payload or len(payload) < 2:
        return False, "Empty payload"
    
    try:
        if technique == "SQLI":
            url = REMOTE_WAF["sqli_url"]
            params = {"id": payload, "Submit": "Submit"}
        elif technique == "XSS":
            url = REMOTE_WAF["xss_url"]
            params = {"name": payload, "Submit": "Submit"}
        else:  # OS_INJECTION
            url = REMOTE_WAF["exec_url"]
            params = {"ip": payload, "Submit": "Submit"}
        
        resp = client.get(url, params=params)
        
        # 403 = blocked, 200 = passed
        if resp.status_code == 403:
            return False, "403 Forbidden"
        elif resp.status_code == 200:
            return True, "200 OK"
        else:
            return False, f"Status {resp.status_code}"
            
    except Exception as e:
        return False, f"Request error: {e}"

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"eval/phi3_modsec_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Phi3 Mini Models - Remote ModSecurity WAF Test")
    print(f"{'='*70}\n")
    print(f"Output: {output_dir}/")
    print(f"Remote WAF: {REMOTE_WAF['base_url']}")
    print(f"Test cases: {len(TEST_CASES)} per adapter")
    print(f"Total tests: {len(PHI3_ADAPTERS)} adapters × {len(TEST_CASES)} = {len(PHI3_ADAPTERS) * len(TEST_CASES)}\n")
    
    # Get authenticated client
    client = get_remote_client()
    if not client:
        print("❌ Failed to connect to remote WAF")
        return
    
    all_results = []
    
    for adapter_info in PHI3_ADAPTERS:
        adapter_name = adapter_info["name"]
        print(f"\n{'─'*70}")
        print(f"Testing: {adapter_name}")
        print(f"{'─'*70}")
        
        if not os.path.exists(adapter_info["adapter"]):
            print(f"⚠️  Adapter not found: {adapter_info['adapter']}")
            continue
        
        # Load model
        try:
            model, tokenizer = load_model(adapter_info["base"], adapter_info["adapter"])
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
        
        adapter_results = {
            "adapter": adapter_name,
            "phase": adapter_info["phase"],
            "base_model": adapter_info["base"],
            "adapter_path": adapter_info["adapter"],
            "timestamp": timestamp,
            "tests": []
        }
        
        passed = 0
        failed = 0
        
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] {test_case['technique']}: {test_case['target']}")
            
            # Build prompt
            prompt = build_prompt(
                adapter_info["phase"],
                test_case["technique"],
                test_case["target"],
                test_case["context"]
            )
            
            # Generate payload
            print("  Generating payload...")
            payload = generate_payload(model, tokenizer, prompt, max_tokens=1024)
            
            if not payload:
                print("  ❌ Failed to generate payload")
                failed += 1
                adapter_results["tests"].append({
                    "test_id": i,
                    "technique": test_case["technique"],
                    "target": test_case["target"],
                    "prompt": prompt,
                    "payload": None,
                    "waf_result": "generation_failed",
                    "passed": False
                })
                continue
            
            print(f"  Payload: {payload[:100]}{'...' if len(payload) > 100 else ''}")
            
            # Test against WAF
            print("  Testing against WAF...")
            waf_passed, waf_msg = test_payload(client, payload, test_case["technique"])
            
            if waf_passed:
                print(f"  ✅ PASSED - {waf_msg}")
                passed += 1
            else:
                print(f"  ❌ BLOCKED - {waf_msg}")
                failed += 1
            
            adapter_results["tests"].append({
                "test_id": i,
                "technique": test_case["technique"],
                "target": test_case["target"],
                "prompt": prompt,
                "payload": payload,
                "waf_result": waf_msg,
                "passed": waf_passed
            })
        
        # Calculate stats
        total = passed + failed
        bypass_rate = (passed / total * 100) if total > 0 else 0
        
        adapter_results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "bypass_rate": bypass_rate
        }
        
        print(f"\n{'─'*70}")
        print(f"Results: {passed}/{total} passed ({bypass_rate:.1f}% bypass rate)")
        print(f"{'─'*70}")
        
        # Save individual adapter results
        adapter_file = output_dir / f"{adapter_name}.jsonl"
        with open(adapter_file, "w", encoding="utf-8") as f:
            for test in adapter_results["tests"]:
                f.write(json.dumps(test, ensure_ascii=False) + "\n")
        
        all_results.append(adapter_results)
        
        # Cleanup model
        del model, tokenizer
        import torch
        torch.cuda.empty_cache()
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "remote_waf": REMOTE_WAF["base_url"],
            "total_adapters": len(PHI3_ADAPTERS),
            "total_tests": len(PHI3_ADAPTERS) * len(TEST_CASES),
            "adapters": all_results
        }, f, indent=2, ensure_ascii=False)
    
    # Generate markdown report
    report_file = output_dir / "REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Phi3 Mini - Remote ModSecurity WAF Test\n\n")
        f.write(f"**Date:** {timestamp}\n\n")
        f.write(f"**Remote WAF:** {REMOTE_WAF['base_url']}\n\n")
        f.write(f"**Test Configuration:**\n")
        f.write(f"- Adapters: {len(PHI3_ADAPTERS)}\n")
        f.write(f"- Test cases per adapter: {len(TEST_CASES)}\n")
        f.write(f"- Total tests: {len(PHI3_ADAPTERS) * len(TEST_CASES)}\n\n")
        
        f.write(f"## Results Summary\n\n")
        f.write(f"| Adapter | Phase | Passed | Failed | Bypass Rate |\n")
        f.write(f"|---------|-------|--------|--------|-------------|\n")
        
        for result in all_results:
            f.write(f"| {result['adapter']} | {result['phase']} | "
                   f"{result['summary']['passed']}/{result['summary']['total_tests']} | "
                   f"{result['summary']['failed']} | "
                   f"{result['summary']['bypass_rate']:.1f}% |\n")
        
        f.write(f"\n## Detailed Results\n\n")
        for result in all_results:
            f.write(f"### {result['adapter']}\n\n")
            f.write(f"**Base Model:** {result['base_model']}\n\n")
            f.write(f"**Adapter Path:** {result['adapter_path']}\n\n")
            f.write(f"**Phase:** {result['phase']}\n\n")
            
            passed_tests = [t for t in result['tests'] if t['passed']]
            failed_tests = [t for t in result['tests'] if not t['passed']]
            
            if passed_tests:
                f.write(f"**Passed ({len(passed_tests)}):**\n")
                for t in passed_tests:
                    f.write(f"- [{t['test_id']}] {t['technique']}: {t['target']}\n")
                    f.write(f"  - Payload: `{t['payload']}`\n")
                f.write("\n")
            
            if failed_tests:
                f.write(f"**Failed ({len(failed_tests)}):**\n")
                for t in failed_tests:
                    f.write(f"- [{t['test_id']}] {t['technique']}: {t['target']}\n")
                    if t['payload']:
                        f.write(f"  - Payload: `{t['payload']}`\n")
                    f.write(f"  - Reason: {t['waf_result']}\n")
                f.write("\n")
    
    print(f"\n{'='*70}")
    print(f"✅ Testing complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"- Summary: {summary_file}")
    print(f"- Report: {report_file}")
    print(f"- Individual results: {output_dir}/*.jsonl\n")

if __name__ == "__main__":
    main()
