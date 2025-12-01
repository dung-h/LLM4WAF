import argparse
import os
import sys
import json
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
import random
import logging
import datetime

# Add project root to path to import WAFEnv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl.waf_env import WAFEnv

# --- Config Defaults ---
BASE_MODEL = "google/gemma-2-2b-it"
ADAPTER_PATH = "experiments/phase2_gemma2_2b_reasoning" # Start from Phase 2 model
OUTPUT_DIR = "experiments/phase3_gemma2_2b_rl"
MAX_STEPS_PER_EPISODE = 5

# --- Setup Logging ---
log_filename = f"rl_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler(sys.stdout) # Also print to console
    ]
)
logger = logging.getLogger(__name__)

def load_model_for_training():
    logger.info(f"Loading model: {BASE_MODEL}")
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
    
    logger.info(f"Loading adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    tokenizer.padding_side = "left" # For generation
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

def compute_log_probs(model, tokenizer, prompt_text, response_text, device):
    # Truncate prompt_text if too long
    # max_seq_len for generation (used when tokenizing full_text) - response_len
    max_context_len = 512 - len(tokenizer.encode(response_text)) # A heuristic, need careful handling if exact max_length is crucial
    if len(tokenizer.encode(prompt_text)) > max_context_len:
        prompt_text = tokenizer.decode(tokenizer.encode(prompt_text)[:max_context_len], skip_special_tokens=True)
    
    full_text = prompt_text + response_text
    
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    outputs = model(**inputs)
    logits = outputs.logits
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs.input_ids[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    loss = loss.view(shift_labels.size())
    
    # Mask out prompt tokens
    response_loss = loss[..., prompt_len-1:]
    
    log_probs = -response_loss
    sequence_log_prob = log_probs.sum()
    
    return sequence_log_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Number of RL epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (episodes per update)") # Reduced batch size
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate per step") # Reduced max_new_tokens
    parser.add_argument("--max_context_length", type=int, default=512, help="Max length for tokenizer context")
    args = parser.parse_args()

    model, tokenizer = load_model_for_training()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    env = WAFEnv(max_steps=MAX_STEPS_PER_EPISODE)
    
    baseline_reward = 0.0
    alpha = 0.1 
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger.info(f"\nStarting REINFORCE Training ({args.epochs} epochs, Batch {args.batch_size}, Total Episodes: {args.epochs * args.batch_size})...")
    
    global_step = 0
    
    for epoch in range(args.epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        batch_loss = 0.0
        batch_rewards = []
        
        optimizer.zero_grad()
        
        for i in range(args.batch_size):
            attack_type = "SQLI" if random.random() < 0.7 else "XSS" 
            target_technique = "RL Generated"
            state = env.reset(attack_type=attack_type, target_technique=target_technique)
            
            episode_log_prob = 0.0
            episode_reward = 0.0
            done = False
            
            # Run Episode
            while not done:
                prompt = generate_prompt(state)
                
                # Truncate prompt for generation to ensure it fits with max_new_tokens
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_context_length - args.max_new_tokens).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.8, pad_token_id=tokenizer.pad_token_id)
                
                response_ids = outputs[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                # Clean up response (sometimes model adds extra text)
                if "<start_of_turn>" in response:
                    response = response.split("<start_of_turn>")[0]
                
                next_state, reward, done, info = env.step(response)
                episode_reward += reward
                
                # Compute Log Prob (with Gradient)
                log_prob = compute_log_probs(model, tokenizer, prompt, response, model.device)
                episode_log_prob += log_prob
                
                state = next_state
            
            advantage = episode_reward - baseline_reward
            loss = -(episode_log_prob * advantage)
            
            loss = loss / args.batch_size # Normalize by batch size
            loss.backward()
            
            batch_loss += loss.item()
            batch_rewards.append(episode_reward)
            global_step += 1
            
            logger.info(f"  Ep {i+1}: Reward={episode_reward:.2f}")

        optimizer.step()
        
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        baseline_reward = (1 - alpha) * baseline_reward + alpha * avg_reward
        
        logger.info(f"Epoch {epoch+1}: Avg Reward = {avg_reward:.4f} | Baseline = {baseline_reward:.4f} | Loss = {batch_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
            model.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    model.save_pretrained(OUTPUT_DIR)
    logger.info(f"Training complete. Model saved to {OUTPUT_DIR}")
    env.close()

if __name__ == "__main__":
    main()