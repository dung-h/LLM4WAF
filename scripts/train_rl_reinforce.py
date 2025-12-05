import argparse
import os
import sys
import json
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import numpy as np
import random
import logging
import datetime
import yaml 

# Add project root to path to import WAFEnv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl.waf_env import WAFEnv

# --- Setup Logging ---
log_filename = f"rl_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler(sys.stdout) 
    ]
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model_for_training(cfg):
    base_model = cfg["base_model"]
    adapter_path = cfg["adapter_path"]
    
    logger.info(f"Loading model: {base_model}")
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
    
    # Prepare for k-bit training (Critical for Gradient Checkpointing with LoRA)
    model = prepare_model_for_kbit_training(model)

    logger.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    
    # Enable Gradient Checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def build_payload_history_list(payload_history):
    if not payload_history: return "None"
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

def generate_prompt_from_state_mock(raw_prompt_text):
    return raw_prompt_text

def compute_log_probs(model, tokenizer, prompt_text, response_text, device, max_context_length):
    full_prompt_formatted = generate_prompt_from_state_mock(prompt_text)

    full_text_ids = tokenizer(full_prompt_formatted + response_text, return_tensors="pt", truncation=True, max_length=max_context_length).input_ids.to(device)
    prompt_ids = tokenizer(full_prompt_formatted, return_tensors="pt", truncation=True, max_length=max_context_length).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # Forward pass to get logits
    outputs = model(full_text_ids)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = full_text_ids[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    loss = loss.view(shift_labels.size())
    
    response_loss = loss[..., prompt_len-1:]
    
    log_probs = -response_loss
    sequence_log_prob = log_probs.sum()
    
    return sequence_log_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    
    args = parser.parse_args()
    cfg = load_config(args.config)

    epochs = cfg.get("epochs", 25)
    batch_size = cfg.get("batch_size", 1) # Default to 1
    lr = float(cfg.get("lr", 1e-6)) 
    max_new_tokens = cfg.get("max_new_tokens", 64)
    max_context_length = cfg.get("max_context_length", 512)
    output_dir = cfg.get("output_dir", "experiments/rl_output")

    model, tokenizer = load_model_for_training(cfg)
    optimizer = AdamW(model.parameters(), lr=lr)
    env = WAFEnv(max_steps=5) 
    
    baseline_reward = 0.0
    alpha = 0.1 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"\nStarting REINFORCE Training ({epochs} epochs, Batch {batch_size}, Total Episodes: {epochs * batch_size})...")
    
    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        batch_loss = 0.0
        batch_rewards = []
        
        optimizer.zero_grad()
        
        for i in range(batch_size):
            attack_type = "SQLI" if random.random() < 0.7 else "XSS" 
            target_technique = "RL Generated" 
            state = env.reset(attack_type=attack_type, target_technique=target_technique)
            
            episode_log_prob = 0.0
            episode_reward = 0.0
            done = False
            
            while not done:
                prompt_content = generate_prompt(state)
                
                inputs = tokenizer(prompt_content, return_tensors="pt", truncation=True, max_length=max_context_length - max_new_tokens).to(model.device)
                
                # GENERATION with use_cache=False for Gradient Checkpointing compatibility
                with torch.no_grad(): 
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, pad_token_id=tokenizer.pad_token_id, use_cache=False)
                
                response_ids = outputs[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                if "<start_of_turn>" in response:
                    response = response.split("<start_of_turn>")[0]
                
                next_state, reward, done, info = env.step(response)
                episode_reward += reward
                
                # LOG PROB calculation (gradient flows here)
                log_prob = compute_log_probs(model, tokenizer, prompt_content, response, model.device, max_context_length)
                episode_log_prob += log_prob
                
                state = next_state
            
            advantage = episode_reward - baseline_reward
            loss = -(episode_log_prob * advantage)
            
            loss = loss / batch_size 
            loss.backward()
            
            batch_loss += loss.item()
            batch_rewards.append(episode_reward)
            
            logger.info(f"  Ep {i+1}: Reward={episode_reward:.2f}")

        optimizer.step()
        
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        baseline_reward = (1 - alpha) * baseline_reward + alpha * avg_reward
        
        logger.info(f"Epoch {epoch+1}: Avg Reward = {avg_reward:.4f} | Baseline = {baseline_reward:.4f} | Loss = {batch_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            model.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    model.save_pretrained(output_dir)
    logger.info(f"Training complete. Model saved to {output_dir}")
    env.close()

if __name__ == "__main__":
    main()