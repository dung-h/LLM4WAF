#!/usr/bin/env python3
"""
Test SQL oracle with proper DVWA login
"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dvwa_session import DVWASession

def oracle_test_authenticated():
    """Test oracles with authenticated DVWA session"""
    
    print("=== Step 3B: SQL Oracle with Authentication ===\n")
    
    BASE = "http://localhost:8080"
    BACKEND = "http://localhost:18081"
    
    sess = DVWASession(BASE, timeout=20.0, backend_base=BACKEND)
    
    try:
        # Login
        print("Logging in to DVWA...")
        if not sess.login():
            print("⚠️  Login failed, trying with cookie fallback...")
            sess.client.cookies.set("security", "low", domain="localhost")
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
        
        if baseline_code == 403:
            print("❌ Even baseline is blocked! WAF is too strict for this test.")
            return
        
        # Test 2: Boolean oracle
        print("\nTesting boolean oracle...")
        t1 = time.time()
        r1 = sess.get("/vulnerabilities/sqli/", params={"id": "1 AND 1=1", "Submit": "Submit"})
        bool_true_time = int((time.time() - t1) * 1000)
        bool_true_len = len(r1.content)
        
        t2 = time.time()
        r2 = sess.get("/vulnerabilities/sqli/", params={"id": "1 AND 1=2", "Submit": "Submit"})
        bool_false_time = int((time.time() - t2) * 1000)
        bool_false_len = len(r2.content)
        
        print(f"  True:  status={r1.status_code} time={bool_true_time}ms len={bool_true_len}")
        print(f"  False: status={r2.status_code} time={bool_false_time}ms len={bool_false_len}")
        print(f"  Length diff: {abs(bool_true_len - bool_false_len)} bytes")
        
        if r1.status_code == 403 or r2.status_code == 403:
            print("  ⚠️  Boolean queries blocked by WAF")
        elif abs(bool_true_len - bool_false_len) > 50:
            print("  ✅ Boolean oracle WORKS - SQL reaches sink")
        else:
            print("  ❌ Boolean oracle FAILS - no significant difference")
        
        # Test 3: Simple time test (without sleep function that WAF might block)
        print("\nTesting simple query delay...")
        t3 = time.time()
        r3 = sess.get("/vulnerabilities/sqli/", params={"id": "1'", "Submit": "Submit"})
        simple_time = int((time.time() - t3) * 1000)
        print(f"  Simple quote: status={r3.status_code} time={simple_time}ms")
        
        if r3.status_code == 403:
            print("  ⚠️  Even simple quote blocked by WAF")
        
        print("\n" + "="*60)
        print("DIAGNOSIS:")
        if baseline_code == 403:
            print("  ❌ WAF blocks even baseline queries")
            print("  → Need to test directly against backend (port 18081)")
            print("  → Or relax WAF rules temporarily")
        elif r1.status_code == 403:
            print("  ⚠️  WAF blocks AND operator in queries")
            print("  → This explains why replay payloads appear as 'bypass'")
            print("  → They're blocked before reaching SQL, so no oracle")
        else:
            print("  ✅ Queries reach application layer")
            
    finally:
        sess.close()

if __name__ == "__main__":
    try:
        oracle_test_authenticated()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
