
import asyncio
import httpx
import re
from urllib.parse import urlsplit

# --- Payloads for Probing ---
BASIC_SQLI_PAYLOADS = [
    "'",
    "''",
    "' OR 1=1--",
    "' OR '1'='1",
    "1' UNION SELECT 1,2--",
    "1' AND 1=1--",
]

BASIC_XSS_PAYLOADS = [
    "<script>alert(1)</script>",
    "<svg onload=alert(1)>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(1)",
    "'\"--><script>alert(1)</script>",
]

# --- DVWA Config ---
DVWA_BASE_URL = "http://localhost:8080"
DVWA_LOGIN_URL = f"{DVWA_BASE_URL}/login.php"
DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
DVWA_XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"

def login_dvwa() -> httpx.Cookies:
    """
    Logs into DVWA and returns the session cookies.
    """
    cookies = httpx.Cookies()
    with httpx.Client(follow_redirects=True, cookies=cookies, timeout=15.0) as client:
        try:
            # 1. GET the login page to get a user_token (CSRF token)
            r = client.get(DVWA_LOGIN_URL)
            r.raise_for_status()
            m = re.search(r'name=["\']user_token["\']\s+value=["\']([^"\']+)["\']', r.text, re.I)
            token = m.group(1) if m else ""
            
            # 2. POST the credentials along with the token
            data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
            r = client.post(DVWA_LOGIN_URL, data=data)
            r.raise_for_status()

            # Check if login was successful by looking for 'Welcome' message or lack of 'login.php' in response
            if "login.php" in str(r.url) or "Login failed" in r.text:
                print("[-] DVWA login failed.")
                return cookies

            print("[+] DVWA login successful.")
            # 3. The client's cookies are now updated with the session cookie
            return client.cookies
        except httpx.RequestError as e:
            print(f"[-] Error connecting to DVWA: {e}")
            return cookies

async def probe_payload(client: httpx.AsyncClient, url: str, param: str, payload: str):
    """
    Sends a single payload and prints the result.
    """
    try:
        r = await client.get(url, params={param: payload}, timeout=10.0)
        response_snippet = r.text.replace('\n', ' ').strip()[:200]
        print(f"  - Payload: {payload}")
        print(f"    - Status: {r.status_code}")
        print(f"    - Blocked: {r.status_code == 403}")
        print(f"    - Response Snippet: {response_snippet}...")
    except httpx.RequestError as e:
        print(f"  - Payload: {payload}")
        print(f"    - Error: {e}")

async def main():
    """
    Main function to run the fuzzing and probing.
    """
    print("--- Starting WAF Fuzzing & Probing ---")
    
    # 1. Login to DVWA
    session_cookies = login_dvwa()
    if not session_cookies:
        print("[!] Aborting probe due to login failure.")
        return

    async with httpx.AsyncClient(cookies=session_cookies, follow_redirects=True) as client:
        # 2. Probe SQLi endpoint
        print("\n[+] Probing SQLi endpoint...")
        for payload in BASIC_SQLI_PAYLOADS:
            await probe_payload(client, DVWA_SQLI_URL, "id", payload)
            
        # 3. Probe XSS endpoint
        print("\n[+] Probing XSS (Reflected) endpoint...")
        for payload in BASIC_XSS_PAYLOADS:
            await probe_payload(client, DVWA_XSS_URL, "name", payload)

    print("\n--- Fuzzing & Probing Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
