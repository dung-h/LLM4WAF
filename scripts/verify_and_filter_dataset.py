import json
import os
import sys
import asyncio
import aiohttp
import httpx # Import httpx for synchronous login
import re
from tqdm.asyncio import tqdm
import argparse
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# --- Verification Logic ---
DVWA_WAF_URL = "http://localhost:8000" # URL through the ModSecurity WAF
DVWA_APP_PATH_PREFIX = "/dvwa" # All DVWA access should go through /dvwa/
DVWA_USERNAME = "admin"
DVWA_PASSWORD = "password"

async def login_dvwa_sync_then_async(aiohttp_session):
    """
    Logs into DVWA using httpx synchronous client, then transfers cookies to the aiohttp session.
    Returns True if successful, False otherwise.
    """
    print("[+] Attempting to log into DVWA with httpx sync client...")
    login_url = f"{DVWA_WAF_URL}{DVWA_APP_PATH_PREFIX}/login.php"
    
    try:
        # Use httpx.Client for synchronous login
        with httpx.Client(base_url=DVWA_WAF_URL, timeout=10.0, follow_redirects=True) as httpx_client:
            # Get user token
            resp_get = httpx_client.get(f"{DVWA_APP_PATH_PREFIX}/login.php")
            resp_get.raise_for_status()
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", resp_get.text, re.I)
            token = m.group(1) if m else ""
            if not token:
                print("[-] Could not find user_token on login page via httpx. HTML might have changed or page not loaded correctly.", file=sys.stderr)
                return False

            # Post login credentials
            data = {"username": DVWA_USERNAME, "password": DVWA_PASSWORD, "user_token": token, "Login": "Login"}
            resp_post = httpx_client.post(f"{DVWA_APP_PATH_PREFIX}/login.php", data=data)
            resp_post.raise_for_status()

            if "login.php" not in str(resp_post.url):
                print("[+] Successfully logged into DVWA via httpx. Transferring cookies...")
                # Transfer cookies from httpx to aiohttp session
                for cookie in httpx_client.cookies.jar:
                    aiohttp_session.cookie_jar.update_cookies({cookie.name: cookie.value})
                print("[+] Cookies transferred to aiohttp session.")
                return True
            else:
                print("[-] httpx login failed. Check credentials or DVWA setup.", file=sys.stderr)
                return False
    except httpx.HTTPStatusError as e:
        print(f"[-] HTTP Error during DVWA login via httpx: {e.response.status_code} - {e.response.text[:100]}", file=sys.stderr)
    except httpx.RequestError as e:
        print(f"[-] Network/Client Error during DVWA login via httpx: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[-] An unexpected error occurred during DVWA login via httpx: {e}", file=sys.stderr)
    return False

async def verify_payload(session, payload, semaphore, attack_type):
    async with semaphore:
        target_endpoint_path = ""
        param_name = ""

        if attack_type == "SQLI":
            target_endpoint_path = "/vulnerabilities/sqli/"
            param_name = "id"
        elif attack_type == "XSS":
            target_endpoint_path = "/vulnerabilities/xss_r/"
            param_name = "name"
        elif attack_type == "OS_INJECTION":
            target_endpoint_path = "/vulnerabilities/exec/" 
            param_name = "ip"
        else:
            print(f"  Warning: Unrecognized attack type '{attack_type}', discarding payload '{payload[:50]}...'.", file=sys.stderr)
            return False # Unrecognized attack type, discard

        try:
            params = {param_name: payload, "Submit": "Submit"} # DVWA common form submission
            full_target_url = f"{DVWA_WAF_URL}{DVWA_APP_PATH_PREFIX}{target_endpoint_path}"
            
            async with session.get(full_target_url, params=params, timeout=15, allow_redirects=False) as response:
                if response.status == 403: # ModSecurity blocks
                    return False 
                return True # Passed WAF (status 200, 302, etc.)
        except aiohttp.ClientResponseError as e:
            if e.status != 403: # Not a WAF block
                return True 
            return False 
        except aiohttp.ClientError as e:
            print(f"  Warning: Network error for payload '{payload[:50]}...' ({attack_type}): {e}", file=sys.stderr)
            return False 
        except Exception as e:
            print(f"  Warning: Unexpected error for payload '{payload[:50]}...' ({attack_type}): {e}", file=sys.stderr)
            return False

async def process_dataset(input_file, concurrency=50):
    print(f"Loading dataset from {input_file}...")
    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                dataset.append(json.loads(line.strip()))
            except:
                continue
    
    print(f"Total samples to verify: {len(dataset)}")
    
    verified_dataset = []
    semaphore = asyncio.Semaphore(concurrency)
    
    async with aiohttp.ClientSession() as session:
        # 1. Login to DVWA once, transferring cookies to aiohttp session
        if not await login_dvwa_sync_then_async(session):
            print("DVWA login failed. Cannot proceed with WAF verification.", file=sys.stderr)
            return 0 # Indicate failure to verify anything
        
        # 2. Process payloads
        tasks = []
        for sample in dataset:
            payload = ""
            attack_type = sample.get("attack_type", "UNKNOWN")
            
            if "messages" in sample:
                for m in sample["messages"]:
                    if m["role"] == "assistant":
                        payload = m["content"]
                        break
            elif "payload" in sample: # Fallback for datasets without messages field
                payload = sample["payload"]
            
            if not payload or attack_type == "UNKNOWN":
                print(f"  Warning: Skipping sample due to missing payload or unknown attack_type: {sample}", file=sys.stderr)
                continue 

            task = verify_payload(session, payload, semaphore, attack_type)
            tasks.append((sample, task))
        
        print(f"Verifying {len(tasks)} payloads against ModSecurity PL1...")
        
        for sample, task in tqdm(tasks, total=len(tasks)):
            passed = await task
            if passed:
                # Cleanup fields: Ensure 'messages' format and remove 'variant_of'
                if "payload" in sample: # If original had 'payload', move to 'messages' format
                    user_message = f"Generic prompt for {sample.get('attack_type', 'UNKNOWN')} using {sample.get('technique', 'UNKNOWN')}"
                    sample["messages"] = [{"role": "user", "content": user_message},
                                          {"role": "assistant", "content": sample["payload"]}]
                    del sample["payload"]
                
                # Ensure messages field exists for augmented samples (which already have it)
                if "messages" not in sample:
                     user_message = f"Generic prompt for {sample.get('attack_type', 'UNKNOWN')} using {sample.get('technique', 'UNKNOWN')}"
                     payload_content = sample.get('payload', '') 
                     sample["messages"] = [{"role": "user", "content": user_message},
                                           {"role": "assistant", "content": payload_content}]
                     if "payload" in sample: del sample["payload"] # Clean up old payload field

                if "variant_of" in sample:
                    del sample["variant_of"]
                if "status" in sample: # Remove status field from raw data
                    del sample["status"]

                verified_dataset.append(sample)
    
    # Stats
    original_count = len(dataset)
    verified_count = len(verified_dataset)
    discarded_count = original_count - verified_count
    
    print(f"\nVerification Complete:")
    print(f"- Original: {original_count}")
    print(f"- Verified (Passed): {verified_count}")
    print(f"- Discarded (Blocked/Error): {discarded_count}")
    
    # Overwrite file
    print(f"Overwriting {input_file} with verified data ({verified_count} samples)...")
    with open(input_file, 'w', encoding='utf-8') as f:
        for sample in verified_dataset:
            f.write(json.dumps(sample) + "\n")
    
    print("Done.")
    return verified_count

def main():
    parser = argparse.ArgumentParser(description="Verify payloads against WAF and filter blocked ones.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests.")
    args = parser.parse_args()
    
    cmd_str = f"python scripts/verify_and_filter_dataset.py --input_file {args.input_file} --concurrency {args.concurrency}"
    
    try:
        count = asyncio.run(process_dataset(args.input_file, args.concurrency))
        log_message(cmd_str, "OK", f"Kept {count} samples")
    except Exception as e:
        print(f"Error running verify_and_filter_dataset.py: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))

if __name__ == "__main__":
    main()