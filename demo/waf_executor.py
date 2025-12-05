# demo/waf_executor.py
import httpx
import re
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- WAFEnv Configuration (Matches rl/waf_env.py for consistency) ---
DVWA_DEFAULT_BASE_URL = "http://localhost:8000"
DVWA_LOGIN_URL_SUFFIX = "/dvwa/login.php"
DVWA_SQLI_URL_SUFFIX = "/dvwa/vulnerabilities/sqli/"
DVWA_XSS_URL_SUFFIX = "/dvwa/vulnerabilities/xss_r/"
DVWA_EXEC_URL_SUFFIX = "/dvwa/vulnerabilities/exec/"
DVWA_USERNAME = "admin"
DVWA_PASSWORD = "password"

class WAFExecutor:
    def __init__(self):
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)
        self.dvwa_base_url = DVWA_DEFAULT_BASE_URL
        self.dvwa_username = DVWA_USERNAME
        self.dvwa_password = DVWA_PASSWORD
        self.target_param = "id" # Default for DVWA SQLi
        self.login_required = True # Default for DVWA

    def update_config(self, config: Dict[str, Any]):
        self.dvwa_base_url = config.get("base_url", DVWA_DEFAULT_BASE_URL)
        self.dvwa_username = config.get("username", DVWA_USERNAME)
        self.dvwa_password = config.get("password", DVWA_PASSWORD)
        self.target_param = config.get("target_param", "id")
        self.login_required = config.get("login_required", True)
        
        # Reset client to ensure new session if base_url changes
        self.client.close()
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)


    def _login_dvwa(self) -> bool:
        login_url = f"{self.dvwa_base_url}{DVWA_LOGIN_URL_SUFFIX}"
        try:
            r = self.client.get(login_url)
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
            if not m:
                logger.error("Failed to find user_token on DVWA login page.")
                return False
            token = m.group(1)
            
            data = {"username": self.dvwa_username, "password": self.dvwa_password, "user_token": token, "Login": "Login"}
            r = self.client.post(login_url, data=data)
            
            if "login.php" not in str(r.url):
                logger.info(f"Successfully logged into DVWA at {self.dvwa_base_url}")
                return True
            else:
                logger.error(f"DVWA login failed. Check credentials or WAF status. Current URL: {r.url}")
                return False
        except httpx.ConnectError as e:
            logger.error(f"Connection error to DVWA at {self.dvwa_base_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error during DVWA login: {e}")
            return False

    def verify_target(self) -> str:
        """
        Verifies connection to the target and attempts login if required.
        Returns a status message.
        """
        logger.info(f"Verifying target: {self.dvwa_base_url}")
        try:
            # First, check if base URL is reachable
            r = self.client.get(self.dvwa_base_url, follow_redirects=True)
            r.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
            
            if self.login_required:
                if "DVWA" not in r.text and DVWA_LOGIN_URL_SUFFIX not in str(r.url):
                    # Not DVWA, but login required - try custom login_url if provided in config
                    login_success = self._login_dvwa()
                else: # It's DVWA
                    login_success = self._login_dvwa()
                
                if not login_success:
                    return "ERROR: Login failed or target unreachable."
            
            return f"SUCCESS: Target {self.dvwa_base_url} reachable. Login {'successful' if self.login_required else 'not required'}."
        
        except httpx.ConnectError as e:
            return f"ERROR: Could not connect to {self.dvwa_base_url}. Is the server running? ({e})"
        except httpx.HTTPStatusError as e:
            return f"ERROR: HTTP Status {e.response.status_code} for {self.dvwa_base_url}"
        except Exception as e:
            return f"ERROR: An unexpected error occurred during verification: {e}"

    def execute_attack(self, payload: str, attack_type: str) -> Dict[str, Any]:
        """
        Executes a single attack with the given payload and returns the result.
        """
        start_time = time.time()
        
        target_url = ""
        params = {"Submit": "Submit"}
        
        if attack_type == "SQL Injection":
            target_url = f"{self.dvwa_base_url}{DVWA_SQLI_URL_SUFFIX}"
            params[self.target_param] = payload
        elif attack_type == "XSS":
            target_url = f"{self.dvwa_base_url}{DVWA_XSS_URL_SUFFIX}"
            params[self.target_param] = payload
        elif attack_type == "OS Injection":
            target_url = f"{self.dvwa_base_url}{DVWA_EXEC_URL_SUFFIX}"
            params[self.target_param] = payload
        else:
            return {"status": "ERROR", "http_code": "-", "latency": 0, "message": "Unsupported attack type"}

        try:
            response = self.client.get(target_url, params=params, follow_redirects=True)
            response_time = (time.time() - start_time) * 1000 # ms
            
            # WAF Block (ModSecurity returns 403)
            if response.status_code == 403:
                return {"status": "BLOCKED", "http_code": 403, "latency": response_time, "message": "WAF blocked the request."}
            
            # Check for successful attack (simplified for demo)
            response_text_lower = response.text.lower()
            if attack_type == "SQL Injection":
                # Look for SQL error (implies bypass) or expected content (if no error, indicates normal behavior)
                if "error" in response_text_lower or "sql syntax" in response_text_lower:
                     return {"status": "PASSED (SQL Error)", "http_code": response.status_code, "latency": response_time, "message": "SQL Error detected, WAF bypassed."}
                elif "first name" in response_text_lower and "admin" in response_text_lower: # Example for DVWA SQLi
                    return {"status": "PASSED (SQLi Success)", "http_code": response.status_code, "latency": response_time, "message": "SQLi payload executed, data retrieved."}
                else:
                    return {"status": "BLOCKED (No SQLi effect)", "http_code": response.status_code, "latency": response_time, "message": "WAF passed, but no SQLi effect detected."}
            elif attack_type == "XSS":
                # Very basic XSS detection: if payload is reflected and contains script-like tags.
                # In a real scenario, this would involve a headless browser.
                if payload.lower() in response_text_lower and (("<script" in payload.lower() or "on" in payload.lower()) and "token" not in payload.lower()):
                    return {"status": "PASSED (Reflected XSS)", "http_code": response.status_code, "latency": response_time, "message": "Payload reflected, possible XSS."}
                else:
                    return {"status": "BLOCKED (No XSS effect)", "http_code": response.status_code, "latency": response_time, "message": "WAF passed, but no XSS effect detected."}
            elif attack_type == "OS Injection":
                if "exec" in response_text_lower and ("root" in response_text_lower or "www-data" in response_text_lower): # Example for `id` command
                    return {"status": "PASSED (OSi Success)", "http_code": response.status_code, "latency": response_time, "message": "OS command executed."}
                else:
                    return {"status": "BLOCKED (No OSi effect)", "http_code": response.status_code, "latency": response_time, "message": "WAF passed, but no OSi effect detected."}
            
            return {"status": "UNKNOWN", "http_code": response.status_code, "latency": response_time, "message": "WAF passed, but result unclear."}

        except httpx.RequestError as e:
            response_time = (time.time() - start_time) * 1000
            return {"status": "ERROR", "http_code": "-", "latency": response_time, "message": f"Network/Request Error: {e}"}
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {"status": "ERROR", "http_code": "-", "latency": response_time, "message": f"Unexpected error: {e}"}

# Global instance
waf_executor = WAFExecutor()
