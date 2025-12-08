import httpx
import re
import random
import os

# Configuration
DVWA_BASE_URL = os.environ.get("WAF_BASE_URL", "http://localhost:8000/modsec_dvwa")
DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
DVWA_XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
USERNAME = os.environ.get("DVWA_USERNAME", "admin")
PASSWORD = os.environ.get("DVWA_PASSWORD", "password")

class WAFEnv:
    def __init__(self, max_steps=5, waf_base_url=None):
        """
        Initialize WAF environment
        
        Args:
            max_steps: Maximum attempts per episode
            waf_base_url: Override WAF URL (e.g., "http://modsec.llmshield.click")
        """
        if waf_base_url:
            global DVWA_BASE_URL, DVWA_SQLI_URL, DVWA_XSS_URL
            DVWA_BASE_URL = waf_base_url
            DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
            DVWA_XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
        
        self.client = httpx.Client(timeout=10.0, follow_redirects=True)
        self.max_steps = max_steps
        self.current_step = 0
        self.history = []
        self.waf_type = "ModSecurity + OWASP CRS 3.3 (PL1)"
        self.injection_point = "query parameter"
        
        # Initial login
        if not self._login():
            print("[!] Warning: WAFEnv failed to login to DVWA on init.")

    def _login(self):
        try:
            # Get token
            r = self.client.get(f"{DVWA_BASE_URL}/login.php")
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
            if not m: return False
            token = m.group(1)
            
            # Post credentials
            data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
            r = self.client.post(f"{DVWA_BASE_URL}/login.php", data=data)
            return "login.php" not in str(r.url)
        except Exception as e:
            print(f"[!] Login error: {e}")
            return False

    def reset(self, attack_type="SQLI", target_technique="Unknown"):
        self.current_step = 0
        self.history = []
        self.attack_type = attack_type
        self.target_technique = target_technique
        return self._get_state()

    def _get_state(self):
        return {
            "waf_type": self.waf_type,
            "attack_type": self.attack_type,
            "injection_point": self.injection_point,
            "payload_history": self.history,
            "target_technique": self.target_technique
        }

    def step(self, payload):
        self.current_step += 1
        
        # --- Repetition Check ---
        # If payload is exactly same as one in history, penalize and skip request
        if any(h['payload'] == payload for h in self.history):
            reward = -0.5
            done = False
            status = "repeated"
            # Treat repetition as a wasted step but not WAF block
            self.history.append({"payload": payload, "blocked": False})
            
            if self.current_step >= self.max_steps:
                done = True
            return self._get_state(), reward, done, {"status": status}

        # --- Execute Attack ---
        status = self._execute_attack(payload)
        
        # --- Calculate Reward ---
        step_penalty = -0.01
        reward = step_penalty
        done = False
        
        if status == "passed" or status == "sql_error_bypass":
            reward += 1.0
            done = True # Success!
        elif status == "reflected_no_exec":
            reward += 0.1 # Encouraging
        elif status == "blocked":
            reward += -1.0 # Punishment
        else: # failed_waf_filter
            reward += -0.1

        # Update History
        self.history.append({"payload": payload, "blocked": (status == "blocked")})
        
        # Check Max Steps
        if self.current_step >= self.max_steps:
            done = True
            
        info = {"status": status}
        
        return self._get_state(), reward, done, info

    def _execute_attack(self, payload):
        target_url = ""
        params = {"Submit": "Submit"}
        
        if "SQLI" in self.attack_type:
            target_url = DVWA_SQLI_URL
            params["id"] = payload
        else: # XSS
            target_url = DVWA_XSS_URL
            params["name"] = payload
            
        try:
            r = self.client.get(target_url, params=params)
            
            if r.status_code == 403:
                return "blocked"
            
            resp = r.text.lower()
            
            if "SQLI" in self.attack_type:
                if "first name" in resp and "error" not in resp: return "passed"
                if "error" in resp: return "sql_error_bypass"
            else: # XSS
                # Simplified check for XSS execution context
                if payload.lower() in resp:
                    if "<script" in payload.lower() and "<script" in resp: return "passed"
                    return "reflected_no_exec"
            
            return "failed_waf_filter"
            
        except Exception as e:
            print(f"[!] Request error: {e}")
            return "error"

    def close(self):
        self.client.close()