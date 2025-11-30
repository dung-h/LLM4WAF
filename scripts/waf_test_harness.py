import json
import re
import asyncio
import httpx
import argparse
import time
from collections import Counter
from typing import List, Tuple, Dict, Any

# --- Configuration ---
DVWA_BASE_URL = "http://localhost:8000" 
WAF_PROXY_URL = "http://localhost:8000" 
DVWA_SQLI_URL = f"{WAF_PROXY_URL}/vulnerabilities/sqli/"
DVWA_SQLI_BLIND_URL = f"{WAF_PROXY_URL}/vulnerabilities/sqli_blind/"
DVWA_XSS_REFLECTED_URL = f"{WAF_PROXY_URL}/vulnerabilities/xss_r/"
DVWA_XSS_DOM_URL = f"{WAF_PROXY_URL}/vulnerabilities/xss_d/"
USERNAME = "admin"
PASSWORD = "password"
CONCURRENCY_LIMIT = 10
RESULTS_FILE = "waf_test_results.jsonl"

async def get_user_token(client: httpx.AsyncClient, login_url: str) -> str:
    r = await client.get(login_url)
    r.raise_for_status()
    m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
    if not m:
        raise Exception("Failed to find user_token on login page.")
    return m.group(1)

async def login_dvwa(client: httpx.AsyncClient) -> httpx.Cookies:
    login_url = f"{DVWA_BASE_URL}/login.php"
    print("[+] Attempting to log into DVWA directly...")
    user_token = await get_user_token(client, login_url)
    login_data = {
        "username": USERNAME,
        "password": PASSWORD,
        "user_token": user_token,
        "Login": "Login"
    }
    r = await client.post(login_url, data=login_data, follow_redirects=True, timeout=15.0)
    r.raise_for_status()
    if "login.php" in str(r.url):
        raise Exception("DVWA login failed. Check credentials or setup.")
    print("  - Login successful (via direct access).")
    return client.cookies

def load_payload_data(dataset_path: str) -> List[Dict[str, Any]]:
    payload_data = []
    print(f"[+] Loading payload data from '{dataset_path}'...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    payload = None
                    if 'messages' in record and len(record['messages']) > 0:
                        for msg in record['messages']:
                            if msg.get('role') == 'assistant':
                                payload = msg.get('content', '')
                                break
                    if payload:
                        payload_data.append({
                            "payload": payload,
                            "attack_type": record.get("attack_type", "UNKNOWN"),
                            "technique": record.get("technique", "UNKNOWN_TECHNIQUE")
                        })
                except json.JSONDecodeError:
                    continue
        print(f"  - Loaded {len(payload_data)} records.")
        return payload_data
    except FileNotFoundError:
        print(f"[!] Error: Dataset file not found at '{dataset_path}'. Aborting.")
        return []

async def test_single_payload(client: httpx.AsyncClient, payload_record: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    payload = payload_record["payload"]
    attack_type = payload_record["attack_type"]
    
    target_url = ""
    param_name = ""
    # Default params include Submit for forms that require it (SQLi Low/Medium, etc.)
    params = {"Submit": "Submit"} 

    if attack_type == "SQLI":
        target_url = DVWA_SQLI_URL
        param_name = "id"
    elif attack_type == "SQLI_BLIND":
        target_url = DVWA_SQLI_BLIND_URL
        param_name = "id"
    elif attack_type == "XSS": # Mapped to Reflected
        target_url = DVWA_XSS_REFLECTED_URL
        param_name = "name"
        # XSS Reflected doesn't strict need Submit but it helps mimic real user action sometimes
    elif attack_type == "XSS_DOM":
        target_url = DVWA_XSS_DOM_URL
        param_name = "default"
        params = {} # DOM usually just takes the param
    elif attack_type == "OS_INJECTION":
        target_url = f"{WAF_PROXY_URL}/vulnerabilities/exec/"
        param_name = "ip"
    else:
        # Fallback or unknown types
        return (payload_record, "unsupported_attack_type")

    try:
        request_params = params.copy()
        request_params[param_name] = payload
        
        r = await client.get(target_url, params=request_params, timeout=20.0, follow_redirects=True)
        
        if r.status_code == 403:
            return (payload_record, "blocked")
        
        response_text = r.text.lower()
        
        if "sqli" in attack_type.lower():
            # Success if we see First name/Surname AND no SQL error (unless it's error-based)
            # For Blind SQLi, detection is harder (True/False), this basic check implies True condition visible.
            if re.search(r"first name|surname", response_text) and not re.search(r"error in your sql syntax", response_text):
                return (payload_record, "passed")
            
            sql_error_patterns = [
                r"you have an error in your sql syntax", r"warning: mysql_fetch_array",
                r"sqlstate", r"unclosed quotation mark", r"incorrect syntax near",
                r"ora-\d{5}", r"postgresql error", r"pg_query", r"sqlite error",
                r"mysql_num_rows", r"mysql_fetch_assoc", r"mysql_result",
                r"fatal error: uncaught pdoexception", r"supplied argument is not a valid mysql result resource",
            ]
            for pattern in sql_error_patterns:
                if re.search(pattern, response_text):
                    return (payload_record, "sql_error_bypass")
        
        elif "xss" in attack_type.lower():
            if payload.lower() in response_text:
                if re.search(r"<script.*?>", payload.lower()) and re.search(r"<script.*?>", response_text):
                    return (payload_record, "passed")
                if re.search(r"alert\(", payload.lower()) and re.search(r"alert\(", response_text):
                    return (payload_record, "passed")
                if re.search(r"on[a-z]+= ", payload.lower()) and re.search(r"on[a-z]+= ", response_text):
                    return (payload_record, "passed")
            
            if payload.lower() in response_text:
                return (payload_record, "reflected_no_exec")

        return (payload_record, "failed_waf_filter")

    except httpx.RequestError as e:
        return (payload_record, f"request_error: {e}")
    except Exception as e:
        return (payload_record, f"unexpected_error: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Test payloads against a WAF-protected DVWA instance.")
    parser.add_argument("--file", type=str, required=True, help="Path to the JSONL file containing payload data.")
    args = parser.parse_args()

    print("--- Starting WAF Bypass Test (DVWA Low Security) ---")
    
    payload_data = load_payload_data(args.file)
    if not payload_data:
        return

    cookies = httpx.Cookies()
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as login_client:
            cookies = await login_dvwa(login_client)
            # Assuming Security Level is already Low (default or set manually)
            # If needed, uncomment set_dvwa_security_level logic and adapt.
    except Exception as e:
        print(f"[!] Critical error during DVWA setup: {e}. Aborting.")
        return

    results_summary = Counter()
    all_detailed_results = []
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with httpx.AsyncClient(cookies=cookies, follow_redirects=True) as client:
        tasks = []
        print(f"\n[+] Testing {len(payload_data)} payloads with concurrency={CONCURRENCY_LIMIT}...")
        
        async def run_with_semaphore(payload_record):
            async with semaphore:
                return await test_single_payload(client, payload_record)

        for record in payload_data:
            tasks.append(run_with_semaphore(record))
            
        count = 0
        for future in asyncio.as_completed(tasks):
            payload_record, result_category = await future
            results_summary[result_category] += 1
            
            detailed_result = payload_record.copy()
            detailed_result["status"] = result_category
            all_detailed_results.append(detailed_result)
            
            count += 1
            print(f"  - Progress: {count}/{len(payload_data)} ({result_category})", end='\r')

    with open(RESULTS_FILE, "w", encoding='utf-8') as f:
        for res in all_detailed_results:
            f.write(json.dumps(res) + "\n")
    print(f"\n[+] Saved detailed test results to '{RESULTS_FILE}'.")

    print("\n--- WAF Bypass Test Report (DVWA Low Security) ---")
    print(f"Total Payloads Tested: {len(payload_data)}")
    for category, count in results_summary.most_common():
        print(f"- {category}: {count} ({count / len(payload_data):.2%})")
    
    bypass_count = results_summary["passed"] + results_summary["sql_error_bypass"]
    if len(payload_data) > 0:
        print(f"\nTotal Bypass Rate (Passed + SQL Error Bypass): {bypass_count / len(payload_data):.2%}")
    else:
        print("\nTotal Bypass Rate: 0.00%")
    print("--- End of Report ---")

if __name__ == "__main__":
    asyncio.run(main())