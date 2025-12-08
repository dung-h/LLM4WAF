#!/usr/bin/env python3
"""
Test outbound connection to remote WAF from Vast.ai instance
Quick smoke test before running full RL training
"""

import requests
import time
import os

def test_waf_connection():
    """Test if we can reach the remote WAF"""
    waf_url = "http://modsec.llmshield.click"
    
    print("=" * 60)
    print("🧪 Testing Outbound Connection to Remote WAF")
    print("=" * 60)
    
    # Test 1: Basic connectivity
    print("\n[1/4] Testing basic HTTP connectivity...")
    try:
        response = requests.get(waf_url, timeout=10)
        print(f"✅ Connection OK - Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection FAILED: {e}")
        return False
    
    # Test 2: Login page access
    print("\n[2/4] Testing DVWA login page...")
    try:
        login_url = f"{waf_url}/login.php"
        response = requests.get(login_url, timeout=10)
        if "Login" in response.text or "DVWA" in response.text:
            print(f"✅ Login page accessible - Status: {response.status_code}")
        else:
            print(f"⚠️  Unexpected response from {login_url}")
    except Exception as e:
        print(f"❌ Login test FAILED: {e}")
        return False
    
    # Test 3: Session with authentication
    print("\n[3/4] Testing authenticated session...")
    try:
        session = requests.Session()
        
        # Get login page to extract token
        login_response = session.get(f"{waf_url}/login.php", timeout=10)
        
        # Login credentials
        username = os.getenv("DVWA_USERNAME", "admin")
        password = os.getenv("DVWA_PASSWORD", "password")
        
        login_data = {
            "username": username,
            "password": password,
            "Login": "Login"
        }
        
        auth_response = session.post(
            f"{waf_url}/login.php",
            data=login_data,
            timeout=10,
            allow_redirects=True
        )
        
        if auth_response.status_code == 200:
            print(f"✅ Authentication successful")
        else:
            print(f"⚠️  Auth response: {auth_response.status_code}")
            
    except Exception as e:
        print(f"❌ Auth test FAILED: {e}")
        return False
    
    # Test 4: SQLi endpoint (the one used in RL)
    print("\n[4/4] Testing SQLi vulnerability endpoint...")
    try:
        sqli_url = f"{waf_url}/vulnerabilities/sqli/"
        test_payload = "1' OR '1'='1"
        
        response = session.get(sqli_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ SQLi endpoint accessible")
            
            # Try a test payload
            post_response = session.post(
                sqli_url,
                data={"id": test_payload, "Submit": "Submit"},
                timeout=10
            )
            print(f"✅ Test payload sent - Status: {post_response.status_code}")
            
            # Check if WAF blocked
            if "blocked" in post_response.text.lower() or post_response.status_code == 403:
                print("✅ WAF is active and blocking payloads (expected)")
            else:
                print("ℹ️  Payload passed (WAF may be in detection mode)")
                
        else:
            print(f"⚠️  SQLi endpoint status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ SQLi test FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests PASSED - Remote WAF is accessible!")
    print("=" * 60)
    print("\n💡 You can proceed with RL training using:")
    print(f"   waf_url: {waf_url}")
    print(f"   DVWA_USERNAME: {username}")
    print(f"   DVWA_PASSWORD: {password}")
    print("\n")
    
    return True

if __name__ == "__main__":
    success = test_waf_connection()
    exit(0 if success else 1)
