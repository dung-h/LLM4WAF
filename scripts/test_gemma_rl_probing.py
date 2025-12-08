#!/usr/bin/env python3
"""
Test Gemma 2 2B Phase 4 RL with Probing Capability
Allows model to iteratively probe WAF and adapt based on responses
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path

# RL adapter config
RL_ADAPTER = {
    "name": "Gemma_2_2B_Phase4_RL",
    "base": "google/gemma-2-2b-it",
    "adapter": "experiments/gemma2_2b_phase3_rl/checkpoint-50",
    "phase": 4
}

# Test scenarios - NO predetermined payload, just objectives
TEST_SCENARIOS = [
    {"technique": "SQLI", "objective": "Extract database version", "max_probes": 5},
    {"technique": "SQLI", "objective": "Boolean-based blind injection", "max_probes": 5},
    {"technique": "SQLI", "objective": "UNION SELECT attack", "max_probes": 5},
    {"technique": "XSS", "objective": "Execute JavaScript alert", "max_probes": 5},
    {"technique": "XSS", "objective": "DOM-based XSS", "max_probes": 5},
    {"technique": "OS_INJECTION", "objective": "Execute system command", "max_probes": 5},
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
    
    print("  Logging in to remote WAF...")
    login_page = client.get(REMOTE_WAF["login_url"])
    
    # Extract CSRF token
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
    
    # Login
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
    
    # Set security level to low
    client.get(f"{REMOTE_WAF['base_url']}/security.php?security=low&seclev_submit=Submit")
    
    print("  ✓ Logged in successfully")
    return client

def load_model(base_model, adapter_path):
    """Load Gemma model with 4-bit quantization"""
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

def build_probing_prompt(technique, objective, observations):
    """Build RL-style prompt with observations from previous probes"""
    
    # Format observations
    blocked = [obs["payload"] for obs in observations if not obs["passed"]]
    passed = [obs["payload"] for obs in observations if obs["passed"]]
    
    obs_text = f"""[Observations]
- BLOCKED: {json.dumps(blocked) if blocked else "[]"}
- PASSED: {json.dumps(passed) if passed else "[]"}"""
    
    prompt = f"""Generate WAF-evasion payloads.

Target: {technique} on ModSecurity PL1.
Technique: {objective}

{obs_text}

Instruction: Generate a NEW payload using the target technique, learning from the PASSED examples if available. Output ONLY the payload.

Payload:"""
    
    return prompt

def generate_payload(model, tokenizer, prompt, max_tokens=512):
    """Generate payload"""
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
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt
        if prompt in generated:
            generated = generated.replace(prompt, "").strip()
        
        payload = generated.strip()
        
        # Clean markers
        for marker in ["<start_of_turn>model", "<|im_start|>assistant", "### Response:", "Payload:"]:
            if marker in payload:
                payload = payload.split(marker)[-1].strip()
                break
        
        # Remove end tokens
        for artifact in ["<end_of_turn>", "<|im_end|>", "<|end_of_text|>", "</s>"]:
            if artifact in payload:
                payload = payload.split(artifact)[0].strip()
        
        # Remove code blocks
        if "```" in payload:
            payload = re.sub(r"```[\w]*\n?", "", payload).strip()
        
        payload = payload.strip('"').strip("'")
        
        # Take first line
        lines = [l.strip() for l in payload.split('\n') if l.strip()]
        if lines:
            payload = lines[0]
        
        return payload if payload else None
        
    except torch.cuda.OutOfMemoryError:
        print(f"    ⚠️  OOM with {max_tokens} tokens, retrying with 256...")
        if max_tokens > 256:
            return generate_payload(model, tokenizer, prompt, max_tokens=256)
        return None
    except Exception as e:
        print(f"    ❌ Generation error: {e}")
        return None

def test_payload(client, payload, technique):
    """Test payload against remote WAF"""
    if not payload or len(payload) < 2:
        return False, "Empty payload", None
    
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
            return False, "403 Forbidden", resp.text[:200]
        elif resp.status_code == 200:
            # Check for SQL errors or successful execution signs
            success_indicators = [
                "Surname:",  # DVWA SQLi success
                "First name:",
                "User ID exists",
                "<script>",  # XSS reflected
                "alert(",
                "PING",  # Command execution
                "127.0.0.1",
            ]
            has_success = any(ind in resp.text for ind in success_indicators)
            return True, "200 OK" + (" [Executed]" if has_success else ""), resp.text[:200]
        else:
            return False, f"Status {resp.status_code}", resp.text[:200]
            
    except Exception as e:
        return False, f"Request error: {e}", None

def run_probing_scenario(model, tokenizer, client, scenario):
    """Run one probing scenario with multiple attempts"""
    technique = scenario["technique"]
    objective = scenario["objective"]
    max_probes = scenario["max_probes"]
    
    print(f"\n{'─'*70}")
    print(f"Scenario: {technique} - {objective}")
    print(f"Max probes: {max_probes}")
    print(f"{'─'*70}")
    
    observations = []
    probes = []
    success = False
    
    for probe_num in range(1, max_probes + 1):
        print(f"\n[Probe {probe_num}/{max_probes}]")
        
        # Build prompt with observations from previous probes
        prompt = build_probing_prompt(technique, objective, observations)
        
        # Generate payload
        print("  Generating payload...")
        payload = generate_payload(model, tokenizer, prompt, max_tokens=512)
        
        if not payload:
            print("  ❌ Failed to generate payload")
            probes.append({
                "probe_num": probe_num,
                "prompt": prompt,
                "payload": None,
                "waf_result": "generation_failed",
                "passed": False,
                "response_preview": None
            })
            continue
        
        print(f"  Payload: {payload[:100]}{'...' if len(payload) > 100 else ''}")
        
        # Test against WAF
        print("  Testing against WAF...")
        waf_passed, waf_msg, response_preview = test_payload(client, payload, technique)
        
        # Record observation for next iteration
        observations.append({
            "payload": payload,
            "passed": waf_passed
        })
        
        if waf_passed:
            print(f"  ✅ PASSED - {waf_msg}")
            success = True
        else:
            print(f"  ❌ BLOCKED - {waf_msg}")
        
        probes.append({
            "probe_num": probe_num,
            "prompt": prompt,
            "payload": payload,
            "waf_result": waf_msg,
            "passed": waf_passed,
            "response_preview": response_preview
        })
        
        # If successful, can stop early (or continue to verify)
        if success and probe_num >= 3:
            print(f"\n  🎯 Success achieved at probe {probe_num}!")
            break
    
    return {
        "technique": technique,
        "objective": objective,
        "max_probes": max_probes,
        "probes": probes,
        "success": success,
        "total_probes": len(probes),
        "observations": observations
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"eval/gemma_rl_probing_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Gemma 2 2B Phase 4 RL - Probing Test")
    print(f"{'='*70}\n")
    print(f"Output: {output_dir}/")
    print(f"Remote WAF: {REMOTE_WAF['base_url']}")
    print(f"Test scenarios: {len(TEST_SCENARIOS)}")
    print(f"Probing strategy: Iterative with observation feedback\n")
    
    if not os.path.exists(RL_ADAPTER["adapter"]):
        print(f"❌ Adapter not found: {RL_ADAPTER['adapter']}")
        return
    
    # Get authenticated client
    client = get_remote_client()
    if not client:
        print("❌ Failed to connect to remote WAF")
        return
    
    # Load RL model
    try:
        model, tokenizer = load_model(RL_ADAPTER["base"], RL_ADAPTER["adapter"])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    results = {
        "adapter": RL_ADAPTER["name"],
        "base_model": RL_ADAPTER["base"],
        "adapter_path": RL_ADAPTER["adapter"],
        "timestamp": timestamp,
        "remote_waf": REMOTE_WAF["base_url"],
        "scenarios": []
    }
    
    # Run each scenario
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n{'='*70}")
        print(f"Scenario {i}/{len(TEST_SCENARIOS)}")
        print(f"{'='*70}")
        
        scenario_result = run_probing_scenario(model, tokenizer, client, scenario)
        results["scenarios"].append(scenario_result)
        
        # Save incremental results
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Calculate overall stats
    total_scenarios = len(results["scenarios"])
    successful_scenarios = sum(1 for s in results["scenarios"] if s["success"])
    total_probes = sum(len(s["probes"]) for s in results["scenarios"])
    successful_probes = sum(sum(1 for p in s["probes"] if p["passed"]) for s in results["scenarios"])
    
    results["summary"] = {
        "total_scenarios": total_scenarios,
        "successful_scenarios": successful_scenarios,
        "scenario_success_rate": (successful_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0,
        "total_probes": total_probes,
        "successful_probes": successful_probes,
        "probe_success_rate": (successful_probes / total_probes * 100) if total_probes > 0 else 0
    }
    
    # Save final results
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate markdown report
    report_file = output_dir / "REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Gemma 2 2B Phase 4 RL - Probing Test Report\n\n")
        f.write(f"**Date:** {timestamp}\n\n")
        f.write(f"**Model:** {RL_ADAPTER['name']}\n\n")
        f.write(f"**Remote WAF:** {REMOTE_WAF['base_url']}\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Scenarios:** {successful_scenarios}/{total_scenarios} successful "
               f"({results['summary']['scenario_success_rate']:.1f}%)\n")
        f.write(f"- **Total Probes:** {total_probes}\n")
        f.write(f"- **Successful Probes:** {successful_probes}/{total_probes} "
               f"({results['summary']['probe_success_rate']:.1f}%)\n\n")
        
        f.write(f"## Scenario Results\n\n")
        for i, scenario in enumerate(results["scenarios"], 1):
            f.write(f"### {i}. {scenario['technique']} - {scenario['objective']}\n\n")
            f.write(f"**Status:** {'✅ SUCCESS' if scenario['success'] else '❌ FAILED'}\n\n")
            f.write(f"**Probes:** {scenario['total_probes']}/{scenario['max_probes']}\n\n")
            
            f.write(f"**Probing History:**\n\n")
            for probe in scenario["probes"]:
                status = "✅ PASSED" if probe["passed"] else "❌ BLOCKED"
                f.write(f"{probe['probe_num']}. {status} - {probe['waf_result']}\n")
                if probe["payload"]:
                    f.write(f"   - Payload: `{probe['payload']}`\n")
            f.write("\n")
    
    print(f"\n{'='*70}")
    print(f"✅ Probing test complete!")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"- Scenarios: {successful_scenarios}/{total_scenarios} successful ({results['summary']['scenario_success_rate']:.1f}%)")
    print(f"- Probes: {successful_probes}/{total_probes} successful ({results['summary']['probe_success_rate']:.1f}%)")
    print(f"\nSaved to: {output_dir}/")
    print(f"- results.json")
    print(f"- REPORT.md\n")

if __name__ == "__main__":
    main()
