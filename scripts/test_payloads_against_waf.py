import json
import re
import asyncio
import httpx
import argparse
from typing import List, Tuple

# --- Configuration ---
DVWA_BASE_URL = "http://localhost:8000"
DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
USERNAME = "admin"
PASSWORD = "password"
CONCURRENCY_LIMIT = 20 # Number of parallel requests

def extract_payload_from_text(text: str) -> str:
    """Extracts the payload from the 'text' field of a JSONL entry."""
    match = re.search(r"Payload:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def load_payloads(dataset_path: str) -> List[str]:
    """Loads all unique payloads from the specified dataset."""
    payloads = set()
    print(f"[+] Loading payloads from '{dataset_path}'...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Handle both direct 'payload' key and nested 'text' key
                    if 'payload' in data:
                        payload = data['payload']
                    elif 'text' in data:
                        payload = extract_payload_from_text(data.get('text', ''))
                    else:
                        continue
                    
                    if payload:
                        payloads.add(payload)
                except json.JSONDecodeError:
                    continue
        print(f"  - Loaded {len(payloads)} unique payloads.")
        return list(payloads)
    except FileNotFoundError:
        print(f"[!] Error: Dataset file not found at '{dataset_path}'. Aborting.")
        return []

async def test_single_payload(client: httpx.AsyncClient, payload: str) -> Tuple[str, str]:
    """Tests a single payload and returns the result."""
    try:
        r = await client.get(DVWA_SQLI_URL, params={"id": payload}, timeout=20.0)
        
        if r.status_code == 403:
            return (payload, "blocked")
        
        sql_error_patterns = [
            r"You have an error in your SQL syntax", r"Warning: mysql_fetch_array",
            r"SQLSTATE", r"Unclosed quotation mark", r"Incorrect syntax near",
            r"ORA-\d{5}", r"PostgreSQL error", r"pg_query", r"SQLite error",
            r"mysql_num_rows", r"mysql_fetch_assoc", r"mysql_result",
            r"Fatal error: Uncaught PDOException",
        ]
        
        response_text = r.text.lower()
        for pattern in sql_error_patterns:
            if re.search(pattern.lower(), response_text):
                return (payload, "sql_error_bypass")
        
        return (payload, "passed")
    except httpx.RequestError as e:
        # print(f"  [!] Request failed for payload: {payload[:50]}... ({e})")
        return (payload, "request_error")

async def main():
    """Main function to load and test all payloads."""
    parser = argparse.ArgumentParser(description="Test SQLi payloads against a WAF.")
    parser.add_argument("--file", type=str, required=True, help="Path to the JSONL file containing payloads.")
    args = parser.parse_args()

    print("--- Starting WAF Bypass Test ---")
    
    payloads = load_payloads(args.file)
    if not payloads:
        return

    # --- Login to DVWA ---
    cookies = httpx.Cookies()
    try:
        print("[+] Logging into DVWA...")
        DVWA_LOGIN_URL = f"{DVWA_BASE_URL}/login.php" # Define the variable here
        with httpx.Client(follow_redirects=True, cookies=cookies, timeout=15.0) as login_client:
            r = login_client.get(DVWA_LOGIN_URL)
            r.raise_for_status()
            m = re.search(r"user_token' value='([a-f0-9]{32})'", r.text, re.I)
            if not m:
                print("[!] Failed to find user_token on login page. Is DVWA running?")
                return
            token = m.group(1)
            data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
            r = login_client.post(DVWA_LOGIN_URL, data=data)
            r.raise_for_status()
            if "login.php" in str(r.url):
                print("[!] DVWA login failed. Check credentials.")
                return
        print("  - Login successful.")
    except Exception as e:
        print(f"[!] An error occurred during login: {e}. Is the WAF+DVWA container running?")
        return

    # --- Run tests in parallel ---
    results = {
        "blocked": 0,
        "passed": 0,
        "sql_error_bypass": 0,
        "request_error": 0,
    }
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with httpx.AsyncClient(cookies=cookies, follow_redirects=True) as client:
        tasks = []
        print(f"\n[+] Testing {len(payloads)} payloads with concurrency={CONCURRENCY_LIMIT}...")
        
        async def run_with_semaphore(payload):
            async with semaphore:
                return await test_single_payload(client, payload)

        for payload in payloads:
            tasks.append(run_with_semaphore(payload))
            
        # Process tasks as they complete to give some feedback
        count = 0
        all_results = []
        for future in asyncio.as_completed(tasks):
            payload, result_category = await future
            results[result_category] += 1
            all_results.append({"payload": payload, "status": result_category})
            count += 1
            print(f"  - Progress: {count}/{len(payloads)}", end='\r')

    # Save all results for detailed analysis
    with open("test_results.jsonl", "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")
    print(f"\n[+] Saved all 5 test results to 'test_results.jsonl'.")

    # Optionally, still save just the passed payloads
    passed_payloads = [res["payload"] for res in all_results if res["status"] == "passed"]
    if passed_payloads:
        with open("passed_payloads.jsonl", "w") as f:
            for payload in passed_payloads:
                f.write(json.dumps({"payload": payload}) + "\n")
        print(f"[+] Saved {len(passed_payloads)} passed payloads to 'passed_payloads.jsonl'.")

    total_payloads = len(payloads)
    blocked_count = results["blocked"]
    passed_count = results["passed"]
    sql_error_count = results["sql_error_bypass"]
    error_count = results["request_error"]

    print("\n--- WAF Bypass Test Report ---")
    print(f"Total Payloads Tested: {total_payloads}")
    print(f"- Blocked: {blocked_count} ({blocked_count / total_payloads:.2%})")
    print(f"- Passed: {passed_count} ({passed_count / total_payloads:.2%})")
    print(f"- Sql Error Bypass: {sql_error_count} ({sql_error_count / total_payloads:.2%})")
    print(f"- Request Error: {error_count} ({error_count / total_payloads:.2%})")
    print(f"\nTotal Bypass Rate (Passed + SQL Error): {(passed_count + sql_error_count) / total_payloads:.2%}")
    print("--- End of Report ---")

if __name__ == "__main__":
    asyncio.run(main())
