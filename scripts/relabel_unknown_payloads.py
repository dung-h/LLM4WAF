#!/usr/bin/env python3
"""
Re-label "unknown" payloads by testing them against real WAF
Creates new file with updated labels
"""
import json
import httpx
import re
from tqdm import tqdm
import time

# DVWA Config
DVWA_BASE_URL = "http://localhost:8000/modsec_dvwa"
LOGIN_URL = f"{DVWA_BASE_URL}/login.php"
SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"

INPUT_FILE = "data/processed/red_v40_balanced_final_v13.jsonl"
OUTPUT_FILE = "data/processed/red_v40_balanced_final_v13_relabeled.jsonl"

def login():
    """Login to DVWA"""
    client = httpx.Client(timeout=15.0, follow_redirects=True)
    
    try:
        r = client.get(LOGIN_URL)
        token = None
        patterns = [
            r"user_token'\s*value='([a-f0-9]{32})'",
            r'user_token"\s*value="([a-f0-9]{32})"',
        ]
        for pattern in patterns:
            m = re.search(pattern, r.text, re.I)
            if m:
                token = m.group(1)
                break
        
        if token:
            r = client.post(LOGIN_URL, data={
                "username": USERNAME, 
                "password": PASSWORD, 
                "user_token": token, 
                "Login": "Login"
            })
        else:
            r = client.post(LOGIN_URL, data={
                "username": USERNAME, 
                "password": PASSWORD, 
                "Login": "Login"
            })
        
        success = "login.php" not in str(r.url).lower()
        if success:
            # Set security to low
            r = client.get(f"{DVWA_BASE_URL}/security.php")
            m = re.search(r'user_token.*?value=["\']([a-f0-9]+)', r.text)
            if m:
                token = m.group(1)
                client.post(
                    f"{DVWA_BASE_URL}/security.php",
                    data={"security": "low", "seclev_submit": "Submit", "user_token": token}
                )
            print(">>> Logged in and set security to LOW")
        return client if success else None
    except Exception as e:
        print(f"[ERROR] Login error: {e}")
        return None

def test_payload(client, payload, attack_type):
    """Test payload against WAF - returns (waf_bypassed, sqli_success)"""
    try:
        if attack_type == "SQLI":
            r = client.get(SQLI_URL, params={"id": payload, "Submit": "Submit"})
        else:  # XSS
            r = client.get(XSS_URL, params={"name": payload, "Submit": "Submit"})
        
        # Check if WAF blocked
        if r.status_code == 403 or "403 Forbidden" in r.text or "Not Acceptable" in r.text:
            return ("blocked", False)
        
        # Check if redirected to login (session lost)
        if "login.php" in str(r.url):
            return ("error", False)
        
        # WAF bypassed, check if attack succeeded
        if attack_type == "SQLI":
            if "Surname:" in r.text or "First name:" in r.text:
                return ("passed", True)  # WAF bypassed AND SQL injection worked
            else:
                return ("passed", False)  # WAF bypassed but SQL injection failed
        else:  # XSS
            # For XSS, just check if payload reflected (basic check)
            if payload.replace("%20", " ").lower() in r.text.lower():
                return ("passed", True)
            else:
                return ("passed", False)
        
    except Exception as e:
        print(f"  ⚠️  Test error: {e}")
        return ("error", False)

def main():
    print("="*80)
    print("Re-labeling 'unknown' payloads by testing against real WAF")
    print("="*80)
    
    # Login first
    client = login()
    if not client:
        print("[ERROR] Failed to login, exiting...")
        return
    
    # Load all data
    print(f"\n>>> Loading from: {INPUT_FILE}")
    data = []
    unknown_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(item)
            # Count payloads without result or with unknown result
            result = item.get('result')
            if result is None or result == 'unknown':
                unknown_count += 1
    
    print(f">>> Loaded {len(data)} total payloads")
    print(f"[FOUND] {unknown_count} payloads with 'unknown' result")
    
    if unknown_count == 0:
        print(">>> No unknown payloads to relabel!")
        return
    
    # Test and relabel unknown payloads
    print(f"\n[TESTING] {unknown_count} unknown payloads against WAF...")
    
    relabeled = 0
    errors = 0
    passed_count = 0
    blocked_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for item in tqdm(data, desc="Processing"):
            # If has valid result (not None, not 'unknown'), keep as is
            result = item.get('result')
            if result is not None and result != 'unknown':
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                continue
            
            # Extract payload
            payload = None
            if 'messages' in item and len(item['messages']) > 1:
                payload = item['messages'][1]['content']
            elif 'payload' in item:
                payload = item['payload']
            
            if not payload:
                # Can't test, keep as unknown
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                continue
            
            # Test payload
            attack_type = item.get('attack_type', 'SQLI')
            new_result, sqli_success = test_payload(client, payload, attack_type)
            
            if new_result == "error":
                errors += 1
                # Keep as unknown if error
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                # Update result
                item['result'] = new_result
                item['relabeled'] = True
                if sqli_success:
                    item['sqli_success'] = True
                
                relabeled += 1
                if new_result == "passed":
                    passed_count += 1
                else:
                    blocked_count += 1
                
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Small delay to avoid overwhelming WAF
            time.sleep(0.1)
    
    print("\n" + "="*80)
    print("[RESULTS] Relabeling Results:")
    print("="*80)
    print(f"  Total payloads:     {len(data)}")
    print(f"  Unknown (before):   {unknown_count}")
    print(f"  Relabeled:          {relabeled}")
    print(f"    -> PASSED:        {passed_count}")
    print(f"    -> BLOCKED:       {blocked_count}")
    print(f"  Errors:             {errors}")
    print("="*80)
    print(f"\n>>> Saved to: {OUTPUT_FILE}")
    print(f"   Use this file for training with accurate labels!")

if __name__ == "__main__":
    main()
