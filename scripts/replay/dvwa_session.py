import re
from typing import Optional
import httpx


class DVWASession:
    def __init__(self, base_url: str, timeout: float = 15.0, backend_base: Optional[str] = None):
        self.base = base_url.rstrip('/')
        self.backend_base = (backend_base or "").rstrip('/') if backend_base else None
        self.client = httpx.Client(follow_redirects=True, timeout=timeout)

    def _extract_csrf(self, html: str) -> Optional[str]:
        # Try double quotes
        m = re.search(r'name="user_token"\s+value="([^"]+)"', html)
        if m:
            return m.group(1)
        # Try single quotes
        m = re.search(r"name='user_token'\s+value='([^']+)'", html)
        return m.group(1) if m else None

    def login(self, username: str = "admin", password: str = "password") -> bool:
        """Login to DVWA and verify success by checking for authenticated content"""
        # Ensure DVWA database is initialized (prefer backend)
        try:
            if self.backend_base:
                self.ensure_setup(backend=True)
            else:
                self.ensure_setup(backend=False)
        except Exception:
            pass

        def _do_login(base: str) -> bool:
            # Get login page and extract CSRF token
            r = self.client.get(f"{base}/login.php")
            token = self._extract_csrf(r.text)
            
            # Build login data
            data = {"username": username, "password": password, "Login": "Login"}
            if token:
                data["user_token"] = token
            
            # Submit login
            r2 = self.client.post(f"{base}/login.php", data=data, follow_redirects=True)
            
            # Check for successful login indicators
            # Success: redirects to index.php, contains "Welcome" or "Logout", no "Login" form
            url_check = "index.php" in str(r2.url)
            content_check = ("Welcome" in r2.text or "Logout" in r2.text) and "Login to your account" not in r2.text
            
            if url_check or content_check:
                return True
            
            # If token-less login failed, retry without token (some DVWA configs)
            if token:
                r3 = self.client.post(f"{base}/login.php", 
                                     data={"username": username, "password": password, "Login": "Login"},
                                     follow_redirects=True)
                url_check3 = "index.php" in str(r3.url)
                content_check3 = ("Welcome" in r3.text or "Logout" in r3.text) and "Login to your account" not in r3.text
                if url_check3 or content_check3:
                    return True
            
            return False

        # Try backend first to establish session, then check if base also works
        if self.backend_base:
            if _do_login(self.backend_base):
                # Session established on backend, should work on WAF too
                return True
        
        # Fallback: try base directly
        return _do_login(self.base)

    def set_security(self, level: str = "low") -> bool:
        # Try backend first when available to obtain token, then WAF
        def _set(base: str) -> bool:
            r = self.client.get(f"{base}/security.php")
            token = self._extract_csrf(r.text) or ""
            data = {"security": level, "seclev_submit": "Submit"}
            if token:
                data["user_token"] = token
            r2 = self.client.post(f"{base}/security.php", data=data)
            return (level in r2.text) or ("Security level set to" in r2.text)

        if self.backend_base and _set(self.backend_base):
            return True
        return _set(self.base)

    def ensure_setup(self, backend: bool = False) -> bool:
        base = self.backend_base if (backend and self.backend_base) else self.base
        r = self.client.get(f"{base}/setup.php")
        if "Create / Reset Database" not in r.text and "Database has been created" not in r.text:
            # Already set up
            return True
        token = self._extract_csrf(r.text) or ""
        data = {"create_db": "Create / Reset Database"}
        if token:
            data["user_token"] = token
        r2 = self.client.post(f"{base}/setup.php", data=data)
        if ("Database has been created" in r2.text):
            return True
        # Re-check the page to confirm setup state
        r3 = self.client.get(f"{base}/setup.php")
        return "Create / Reset Database" not in r3.text

    def get(self, path: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.base}{path}"
        return self.client.get(url, headers=headers, params=params)

    def close(self):
        self.client.close()
