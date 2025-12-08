#!/usr/bin/env python3
"""
RL Training with Adaptive Pipeline - Gemma 2 2B Phase 3
Full autonomous pipeline: WAF Detection → Probing → Adaptive Generation → RL Training
"""
import os
import sys
import logging
import random
import time
import httpx
import re
import torch
import yaml
import datetime
import json
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm

# --- Configuration ---
CONFIG_PATH = "configs/gemma2_2b_phase3_rl.yaml"
TRAINING_DATA_PATH = "data/processed/red_v40_balanced_final_v13.jsonl"

# DVWA Config - Use localhost for testing strict WAF
DVWA_BASE_URL = "http://localhost:8000/modsec_dvwa"
LOGIN_URL = f"{DVWA_BASE_URL}/login.php"
SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"

# Number of random probes per episode (mix of passed and blocked)
NUM_PROBES = 10  # Random sample each time
PROBE_MIX_RATIO = 0.5  # 50% passed, 50% blocked/unknown

# Logging
log_filename = f"rl_adaptive_pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_probe_payloads():
    """Load diverse payloads from training data for probing - mix of passed and blocked"""
    sqli_passed = []
    sqli_blocked = []
    xss_passed = []
    xss_blocked = []
    
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Extract payload from messages
            payload = None
            if 'messages' in data and len(data['messages']) > 1:
                payload = data['messages'][1]['content']
            elif 'payload' in data:
                payload = data['payload']
            
            if not payload:
                continue
            
            technique = data.get('technique', 'Unknown')
            result = data.get('result', 'unknown')
            
            if data['attack_type'] == 'SQLI':
                if result == 'passed':
                    sqli_passed.append({"payload": payload, "technique": technique, "result": result})
                else:  # blocked or unknown
                    sqli_blocked.append({"payload": payload, "technique": technique, "result": result})
            elif data['attack_type'] == 'XSS':
                if result == 'passed':
                    xss_passed.append({"payload": payload, "technique": technique, "result": result})
                else:  # blocked or unknown
                    xss_blocked.append({"payload": payload, "technique": technique, "result": result})
    
    logger.info(f"Loaded SQLI: {len(sqli_passed)} passed, {len(sqli_blocked)} blocked/unknown")
    logger.info(f"Loaded XSS: {len(xss_passed)} passed, {len(xss_blocked)} blocked/unknown")
    
    return {
        'sqli_passed': sqli_passed,
        'sqli_blocked': sqli_blocked,
        'xss_passed': xss_passed,
        'xss_blocked': xss_blocked
    }

# Load payload pools at startup
PAYLOAD_POOLS = load_probe_payloads()

class AdaptivePipelineEnv:
    """Environment that mimics Phase 3 adaptive learning with real WAF"""
    
    def __init__(self):
        self.client = httpx.Client(timeout=10.0, follow_redirects=True)
        self.attack_type = None
        self.probing_results = []
        
    def login(self):
        """Login to DVWA"""
        try:
            r = self.client.get(LOGIN_URL)
            token = None
            patterns = [
                r"user_token'\s*value='([a-f0-9]{32})'",
                r'user_token"\s*value="([a-f0-9]{32})"',
            ]
            for pattern in patterns:
                m = re.search(pattern, r.text, re.I)
                if m:
                    token = m.group(1)
                    break
            
            if not token:
                logger.warning("No token found, trying without CSRF")
                r = self.client.post(LOGIN_URL, data={"username": USERNAME, "password": PASSWORD, "Login": "Login"})
            else:
                r = self.client.post(LOGIN_URL, data={"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"})
            
            success = "login.php" not in str(r.url).lower()
            if success:
                logger.info("✅ Logged into DVWA")
            return success
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def probe_waf(self, attack_type):
        """Phase 1: Probe WAF with random diverse payloads - mix of passed and blocked"""
        logger.info(f"🔍 Probing WAF for {attack_type}...")
        
        # Get passed and blocked pools
        if attack_type == "SQLI":
            passed_pool = PAYLOAD_POOLS['sqli_passed']
            blocked_pool = PAYLOAD_POOLS['sqli_blocked']
        else:
            passed_pool = PAYLOAD_POOLS['xss_passed']
            blocked_pool = PAYLOAD_POOLS['xss_blocked']
        
        # Sample mix of passed and blocked payloads
        num_passed = int(NUM_PROBES * PROBE_MIX_RATIO)
        num_blocked = NUM_PROBES - num_passed
        
        probes = []
        if len(passed_pool) >= num_passed:
            probes.extend(random.sample(passed_pool, num_passed))
        else:
            probes.extend(passed_pool)
            
        if len(blocked_pool) >= num_blocked:
            probes.extend(random.sample(blocked_pool, num_blocked))
        else:
            probes.extend(blocked_pool)
        
        # Shuffle to mix them
        random.shuffle(probes)
        
        results = []
        
        for probe in probes:
            status = self._test_payload(probe["payload"], attack_type)
            results.append({
                "payload": probe["payload"],
                "technique": probe["technique"],
                "result": status,
                "original_result": probe.get("result", "unknown")
            })
            orig = probe.get("result", "?")
            logger.info(f"  [{orig}→{status}] {probe['technique'][:35]}")
        
        return results
    
    def _test_payload(self, payload, attack_type):
        """Test payload against WAF"""
        try:
            if attack_type == "SQLI":
                r = self.client.get(SQLI_URL, params={"id": payload, "Submit": "Submit"})
            else:  # XSS
                r = self.client.get(XSS_URL, params={"name": payload, "Submit": "Submit"})
            
            # Check for WAF block
            if r.status_code == 403 or "403 Forbidden" in r.text or "Not Acceptable" in r.text:
                return "BLOCKED"
            
            # Check for successful execution
            if attack_type == "SQLI":
                if "Surname:" in r.text or "First name:" in r.text:
                    return "PASSED"
            else:  # XSS
                if payload.replace("%20", " ").lower() in r.text.lower():
                    return "PASSED"
            
            return "BLOCKED"
        except Exception as e:
            logger.error(f"Test error: {e}")
            return "ERROR"
    
    def reset(self, attack_type):
        """Reset environment for new episode"""
        self.attack_type = attack_type
        self.probing_results = self.probe_waf(attack_type)
        return self._get_state()
    
    def _get_state(self):
        """Get current state for RL"""
        # Separate BLOCKED and PASSED
        blocked = [p for p in self.probing_results if p["result"] == "BLOCKED"]
        passed = [p for p in self.probing_results if p["result"] == "PASSED"]
        
        return {
            "attack_type": self.attack_type,
            "blocked_examples": blocked[:3],  # Limit to 3
            "passed_examples": passed[:2],    # Limit to 2
            "waf_type": "ModSecurity (Detected via Probing)",
        }
    
    def step(self, payload):
        """Execute generated payload and return reward"""
        result = self._test_payload(payload, self.attack_type)
        
        # Compute reward
        if result == "PASSED":
            reward = 1.0
            done = True
        elif result == "BLOCKED":
            reward = -0.5
            done = False
        else:  # ERROR
            reward = -1.0
            done = False
        
        return self._get_state(), reward, done
    
    def close(self):
        self.client.close()

def build_adaptive_prompt(state, technique):
    """Build Phase 3 style prompt with BLOCKED/PASSED examples"""
    attack_type = state["attack_type"]
    blocked = state["blocked_examples"]
    passed = state["passed_examples"]
    
    # Format examples
    blocked_str = "\n".join([f"  - {p['payload']}" for p in blocked]) if blocked else "  (none)"
    passed_str = "\n".join([f"  - {p['payload']}" for p in passed]) if passed else "  (none)"
    
    attack_str = "SQL injection" if attack_type == "SQLI" else "XSS"
    
    prompt = f"""Generate WAF-evasion payloads.

Target: {attack_str} on {state['waf_type']}.
Technique: {technique}

[Observations]
- BLOCKED: 
{blocked_str}
- PASSED:
{passed_str}

Instruction: Generate a NEW payload using the target technique, learning from the PASSED examples if available. Output ONLY the payload."""
    
    return prompt

def generate_payload(model, tokenizer, state, technique, max_length=384):
    """Generate payload using model"""
    prompt = build_adaptive_prompt(state, technique)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Extract payload
    lines = generated.split('\n')
    payload = lines[0].strip() if lines else generated.strip()
    
    return payload

def compute_log_probs(model, tokenizer, prompt_text, response_text, max_length):
    """Compute log probabilities for REINFORCE"""
    # Format with chat template
    messages = [{"role": "user", "content": prompt_text}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    full_text_ids = tokenizer(full_prompt + response_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(model.device)
    prompt_ids = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    
    # Forward pass
    outputs = model(full_text_ids)
    logits = outputs.logits
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = full_text_ids[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    
    # Only response part
    response_loss = loss[..., prompt_len-1:]
    log_probs = -response_loss
    
    return log_probs.sum()

def load_model_for_training(cfg):
    """Load model and adapter for RL training"""
    base_model = cfg["base_model"]
    adapter_path = cfg["adapter_path"]
    
    logger.info(f"Loading {base_model} + {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN")
    )
    
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model loaded and trainable")
    return model, tokenizer

def main():
    cfg = load_config()
    
    epochs = cfg.get("epochs", 50)
    lr = float(cfg.get("lr", 1e-6))
    max_context_length = cfg.get("max_context_length", 384)
    output_dir = cfg.get("output_dir", "experiments/gemma2_2b_phase3_rl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("🎮 RL Training: Adaptive Pipeline (Gemma 2 2B Phase 3)")
    logger.info("="*80)
    
    # Setup
    env = AdaptivePipelineEnv()
    if not env.login():
        logger.error("Failed to login to DVWA. Aborting.")
        return
    
    model, tokenizer = load_model_for_training(cfg)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    baseline_reward = 0.0
    alpha = 0.1
    
    techniques = ["Double URL Encode", "Comment Obfuscation", "Boolean OR", "UNION SELECT",
                  "Event Handler", "Script Tag", "IMG onerror", "SVG onload"]
    
    logger.info(f"\nStarting RL Training ({epochs} epochs)...")
    
    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # Select attack type
        attack_type = "SQLI" if random.random() < 0.7 else "XSS"
        technique = random.choice(techniques)
        
        # Reset env with probing
        state = env.reset(attack_type=attack_type)
        
        # Generate payload
        optimizer.zero_grad()
        
        prompt_text = build_adaptive_prompt(state, technique)
        payload = generate_payload(model, tokenizer, state, technique, max_context_length)
        
        logger.info(f"Generated: {payload[:60]}...")
        
        # Test payload
        next_state, reward, done = env.step(payload)
        
        logger.info(f"Result: Reward={reward:.2f}")
        
        # Compute loss
        log_prob = compute_log_probs(model, tokenizer, prompt_text, payload, max_context_length)
        advantage = reward - baseline_reward
        loss = -(log_prob * advantage)
        
        loss.backward()
        torch.cuda.empty_cache()
        
        optimizer.step()
        
        # Update baseline
        baseline_reward = (1 - alpha) * baseline_reward + alpha * reward
        
        logger.info(f"Baseline: {baseline_reward:.4f} | Loss: {loss.item():.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            model.save_pretrained(save_path)
            logger.info(f"💾 Saved checkpoint to {save_path}")
    
    # Final save
    model.save_pretrained(output_dir)
    logger.info(f"\n✅ Training complete. Model saved to {output_dir}")
    logger.info("="*80)
    
    env.close()

if __name__ == "__main__":
    main()
