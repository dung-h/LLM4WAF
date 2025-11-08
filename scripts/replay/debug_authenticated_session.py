#!/usr/bin/env python3
"""
Debug script to test authenticated SQLi endpoint access
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dvwa_session import DVWASession

def debug_session():
    BASE = "http://localhost:8080"
    BACKEND = "http://localhost:18081"
    
    print("=== Testing DVWA Session ===\n")
    
    sess = DVWASession(BASE, timeout=20.0, backend_base=BACKEND)
    
    try:
        # Login
        print("1. Attempting login...")
        if sess.login():
            print("   ✅ Login successful")
        else:
            print("   ❌ Login failed")
            return
        
        # Set security
        print("\n2. Setting security level...")
        if sess.set_security("low"):
            print("   ✅ Security set to low")
        else:
            print("   ⚠️  Security setting uncertain")
        
        # Check cookies
        print("\n3. Current cookies:")
        for cookie in sess.client.cookies.jar:
            print(f"   {cookie.name}={cookie.value[:20]}... (domain={cookie.domain})")
        
        # Test endpoint on backend
        print("\n4. Testing SQLi endpoint on BACKEND...")
        r1 = sess.client.get(f"{BACKEND}/vulnerabilities/sqli/", params={"id": "1", "Submit": "Submit"})
        print(f"   Status: {r1.status_code}")
        print(f"   Length: {len(r1.content)} bytes")
        print(f"   URL: {r1.url}")
        has_data = "First name" in r1.text or "Surname" in r1.text
        has_login = "login to your account" in r1.text.lower()
        print(f"   Has SQL data: {has_data}")
        print(f"   Is login page: {has_login}")
        
        # Test endpoint through WAF
        print("\n5. Testing SQLi endpoint through WAF...")
        r2 = sess.get("/vulnerabilities/sqli/", params={"id": "1", "Submit": "Submit"})
        print(f"   Status: {r2.status_code}")
        print(f"   Length: {len(r2.content)} bytes")
        print(f"   URL: {r2.url}")
        has_data2 = "First name" in r2.text or "Surname" in r2.text
        has_login2 = "login to your account" in r2.text.lower()
        print(f"   Has SQL data: {has_data2}")
        print(f"   Is login page: {has_login2}")
        
        # Test with potential SQLi
        print("\n6. Testing with simple payload...")
        r3 = sess.get("/vulnerabilities/sqli/", params={"id": "1' OR '1'='1", "Submit": "Submit"})
        print(f"   Status: {r3.status_code}")
        print(f"   Length: {len(r3.content)} bytes")
        if r3.status_code == 403:
            print("   ✅ WAF blocked the payload (expected)")
        elif "First name" in r3.text:
            print("   ⚠️  Payload reached database (bypass!)")
        else:
            print("   ❓ Unexpected response")
        
    finally:
        sess.close()

if __name__ == "__main__":
    try:
        debug_session()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
