"""
Test connection to remote WAF and debug login process
"""

import httpx
import re

REMOTE_WAF_URL = "http://modsec.llmshield.click"
USERNAME = "admin"
PASSWORD = "password"

print(f"Testing connection to: {REMOTE_WAF_URL}\n")

client = httpx.Client(timeout=30.0, follow_redirects=True)

# Step 1: Test basic connectivity and detect DVWA path
print("1. Testing basic connectivity and detecting DVWA path...")
dvwa_paths = [
    "",  # Root
    "/dvwa",  # Standard path
    "/DVWA",  # Uppercase variant
]

base_path = None
try:
    # First check root
    resp = client.get(REMOTE_WAF_URL)
    print(f"   Root status: {resp.status_code}")
    print(f"   Content length: {len(resp.text)}")
    
    # Try to detect DVWA
    for path in dvwa_paths:
        test_url = f"{REMOTE_WAF_URL}{path}/login.php" if path else f"{REMOTE_WAF_URL}/login.php"
        try:
            resp = client.get(test_url)
            if resp.status_code == 200 and "DVWA" in resp.text:
                base_path = path
                print(f"   ✓ Found DVWA at: {test_url}")
                break
        except:
            pass
    
    if base_path is None:
        print("   ✗ Cannot find DVWA login page")
        print("   Trying common paths...")
        # Check if root has login
        resp = client.get(REMOTE_WAF_URL)
        if "login" in resp.text.lower() or "dvwa" in resp.text.lower():
            print(f"   Root page content (first 500 chars):")
            print(f"   {resp.text[:500]}")
            base_path = ""  # Assume root
        else:
            print("   Cannot detect DVWA. Exiting.")
            exit(1)

except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

REMOTE_LOGIN_URL = f"{REMOTE_WAF_URL}{base_path}/login.php" if base_path else f"{REMOTE_WAF_URL}/login.php"
print(f"\n   Using login URL: {REMOTE_LOGIN_URL}")

# Step 2: Get login page
print("\n2. Getting login page...")
try:
    resp = client.get(REMOTE_LOGIN_URL)
    print(f"   Status: {resp.status_code}")
    print(f"   URL: {resp.url}")
    print(f"   Content length: {len(resp.text)}")
    
    # Save HTML for inspection
    with open("login_page.html", "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"   Saved to: login_page.html")
    
    # Try to find CSRF token with different patterns
    patterns = [
        r"name=['\"]user_token['\"] value=['\"]([^'\"]+)",
        r"name='user_token' value='([^']+)",
        r'name="user_token" value="([^"]+)',
        r"user_token['\"]?\s*value=['\"]([^'\"]+)",
        r"<input[^>]*name=['\"]user_token['\"][^>]*value=['\"]([^'\"]+)",
    ]
    
    print("\n3. Searching for CSRF token...")
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, resp.text)
        if match:
            print(f"   ✓ Found with pattern {i+1}: {match.group(1)[:20]}...")
            user_token = match.group(1)
            break
    else:
        print("   ✗ No token found with any pattern")
        print("\n   First 1000 chars of HTML:")
        print("   " + resp.text[:1000])
        exit(1)
    
    # Step 3: Try to login
    print("\n4. Attempting login...")
    login_data = {
        "username": USERNAME,
        "password": PASSWORD,
        "Login": "Login",
        "user_token": user_token
    }
    
    resp = client.post(REMOTE_LOGIN_URL, data=login_data)
    print(f"   Status: {resp.status_code}")
    print(f"   URL after login: {resp.url}")
    
    if "Login failed" in resp.text:
        print("   ✗ Login failed message found")
    elif "logout" in resp.text.lower():
        print("   ✓ Login successful (logout button found)")
    else:
        print("   ? Uncertain login status")
        with open("after_login.html", "w", encoding="utf-8") as f:
            f.write(resp.text)
        print("   Saved to: after_login.html")
    
    # Step 4: Test SQLi endpoint
    print("\n5. Testing SQLi endpoint...")
    sqli_url = f"{REMOTE_WAF_URL}{base_path}/vulnerabilities/sqli/" if base_path else f"{REMOTE_WAF_URL}/vulnerabilities/sqli/"
    resp = client.get(sqli_url, params={"id": "1", "Submit": "Submit"})
    print(f"   Status: {resp.status_code}")
    print(f"   URL: {resp.url}")
    
    if resp.status_code == 200:
        print("   ✓ SQLi page accessible")
    elif resp.status_code == 403:
        print("   ! WAF blocking (403)")
    else:
        print(f"   ? Unexpected status: {resp.status_code}")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Test completed. Check login_page.html for details.")
