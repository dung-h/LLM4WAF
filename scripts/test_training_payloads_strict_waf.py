#!/usr/bin/env python3
"""
Test training data payloads (labeled PASSED) against strict localhost WAF
"""
import json
import httpx
import re

# Localhost DVWA with STRICT ModSecurity
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
            print("✅ Logged in\n")
            return client
        else:
            print("❌ Login failed")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_payload(client, payload):
    """Test SQLI payload - returns tuple (waf_status, sqli_status)"""
    try:
        r = client.get(SQLI_URL, params={"id": payload, "Submit": "Submit"})
        
        # Check if WAF blocked
        if r.status_code == 403 or "403 Forbidden" in r.text or "Not Acceptable" in r.text:
            return ("WAF_BLOCKED", "N/A")
        
        # Check if redirected to login (session lost)
        if "login.php" in str(r.url):
            return ("ERROR", "SESSION_LOST")
        
        # WAF bypassed, now check if SQL injection worked
        if "Surname:" in r.text or "First name:" in r.text:
            return ("WAF_BYPASSED", "SQLI_SUCCESS")
        
        # WAF bypassed but SQL injection failed
        return ("WAF_BYPASSED", "SQLI_FAILED")
    except Exception as e:
        return ("ERROR", str(e))

def main():
    print("="*80)
    print("🧪 Testing Training Data Payloads (PASSED labels) vs Strict WAF")
    print("="*80)
    
    client = login()
    if not client:
        return
    
    # Load PASSED payloads from training data
    passed_payloads = []
    with open("data/processed/red_v40_passed_waf_only.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # File format: {"payload": "...", "attack_type": "SQLI", "status": "..."}
            if data.get("attack_type") == "SQLI":  # Only test SQLI
                passed_payloads.append(data["payload"])
    
    print(f"Loaded {len(passed_payloads)} SQLI payloads from training data\n")
    
    # Test first 20 payloads
    results = {"PASSED": 0, "BLOCKED": 0, "ERROR": 0}
    test_count = min(20, len(passed_payloads))
    
    print(f"Testing {test_count} payloads against strict localhost WAF...\n")
    
    waf_bypassed = 0
    waf_blocked = 0
    sqli_success = 0
    sqli_failed = 0
    errors = 0
    
    for i, payload in enumerate(passed_payloads[:test_count]):
        waf_status, sqli_status = test_payload(client, payload)
        
        if waf_status == "WAF_BYPASSED":
            waf_bypassed += 1
            if sqli_status == "SQLI_SUCCESS":
                icon = "✅✅"
                sqli_success += 1
            else:
                icon = "✅❌"
                sqli_failed += 1
        elif waf_status == "WAF_BLOCKED":
            icon = "❌❌"
            waf_blocked += 1
        else:
            icon = "⚠️⚠️"
            errors += 1
        
        payload_short = payload[:50].ljust(50)
        print(f"{i+1:2d}. {icon} {payload_short} -> {waf_status:12s} | {sqli_status}")
    
    # Summary
    total = test_count
    print(f"\n{'='*80}")
    print(f"📊 Results Summary:")
    print(f"{'='*80}")
    if total > 0:
        print(f"\nWAF Bypass Rate:")
        print(f"   WAF BYPASSED: {waf_bypassed:2d}/{total} ({waf_bypassed/total*100:.1f}%)")
        print(f"   WAF BLOCKED:  {waf_blocked:2d}/{total} ({waf_blocked/total*100:.1f}%)")
        print(f"   ERRORS:       {errors:2d}/{total}")
        
        if waf_bypassed > 0:
            print(f"\nSQL Injection Success (among bypassed payloads):")
            print(f"   SQLI SUCCESS: {sqli_success:2d}/{waf_bypassed} ({sqli_success/waf_bypassed*100:.1f}%)")
            print(f"   SQLI FAILED:  {sqli_failed:2d}/{waf_bypassed} ({sqli_failed/waf_bypassed*100:.1f}%)")
    print(f"{'='*80}")
    
    print("\nLegend:")
    print("  ✅✅ = WAF bypassed AND SQL injection successful")
    print("  ✅❌ = WAF bypassed BUT SQL injection failed")
    print("  ❌❌ = WAF blocked the request (403)")
    
    if waf_bypassed > 0:
        print(f"\n✅ {waf_bypassed}/{total} payloads BYPASSED the WAF ({waf_bypassed/total*100:.1f}%)")
        if sqli_success > 0:
            print(f"✅ {sqli_success}/{waf_bypassed} also executed SQL injection successfully")
        else:
            print("⚠️  None of the bypassed payloads executed SQL injection successfully")
            print("   → Payloads bypass WAF but don't exploit DVWA (maybe wrong syntax)")
    else:
        print("\n❌ ZERO payloads bypassed the WAF - it's blocking everything!")
        print("   → This localhost WAF is MUCH stricter than training environment")
    
    client.close()

if __name__ == "__main__":
    main()
