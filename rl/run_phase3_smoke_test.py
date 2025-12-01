
import sys
import os
# Add project root to path to allow imports if needed, though we'll be self-contained
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from waf_env import WAFEnv

# --- Model Config ---
BASE_MODEL = "google/gemma-2-2b-it"
ADAPTER_PATH = "experiments/phase2_gemma2_2b_reasoning"
OUTPUT_FILE = "rl/trajectories_phase3_smoke.jsonl"

# --- Utils ---
def load_model():
    print(f"Loading model: {BASE_MODEL} + {ADAPTER_PATH}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN")
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    return model, tokenizer

def build_payload_history_list(payload_history):
    if not payload_history:
        return "None"
    history_str = ""
    for i, item in enumerate(payload_history):
        history_str += f"{i+1}) {item['payload']}\n"
    return history_str.strip()

def generate_prompt(state):
    waf_type = state["waf_type"]
    attack_type = state["attack_type"]
    injection_point = state["injection_point"]
    history_list = build_payload_history_list(state["payload_history"])
    target_technique = state["target_technique"]
    
    attack_type_str = "SQL injection" if attack_type == "SQLI" else ("XSS" if attack_type == "XSS" else attack_type)

    prompt = f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: {waf_type}.
- Attack type: {attack_type_str}.
- Injection point: {injection_point}.

Previously tried payloads against this target (all blocked by the WAF):
{history_list}

Your task:
Generate a NEW {attack_type_str} payload that has a higher chance of bypassing this WAF while still reaching and triggering the underlying vulnerability.
Use the following core technique as the main idea for the payload:
- Technique: {target_technique}

You may combine this technique with additional obfuscation tricks if it helps evade the filter, but keep the payload compact and realistic.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations or comments.
- Do NOT wrap it in code fences."""
    
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

def main():
    # Initialize Model
    model, tokenizer = load_model()
    
    # Initialize Env
    env = WAFEnv()
    
    episodes = 5
    print(f"\nStarting Smoke Test ({episodes} episodes)...")
    
    trajectories = []
    
    for ep in range(episodes):
        # Pick a random scenario
        attack_type = "SQLI" if ep % 2 == 0 else "XSS"
        target_technique = "Smoke Test Technique"
        
        state = env.reset(attack_type=attack_type, target_technique=target_technique)
        done = False
        
        print(f"\n--- Episode {ep+1} ({attack_type}) ---")
        
        while not done:
            prompt = generate_prompt(state)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
            
            payload = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Step Environment
            next_state, reward, done, info = env.step(payload)
            
            print(f"Step {env.current_step}: Reward={reward} | Status={info['status']}")
            
            # Log
            log_entry = {
                "episode": ep,
                "step": env.current_step,
                "attack_type": attack_type,
                "payload": payload,
                "reward": reward,
                "status": info["status"],
                "history_len": len(state["payload_history"])
            }
            trajectories.append(log_entry)
            
            state = next_state

    # Save trajectories
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")
    print(f"\nSmoke test complete. Trajectories saved to {OUTPUT_FILE}")
    env.close()

if __name__ == "__main__":
    main()
