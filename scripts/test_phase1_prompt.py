#!/usr/bin/env python3
"""Quick test Phase 1 model with correct prompt format"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Config
BASE_MODEL = "google/gemma-2-2b"  # Try non-instruct version
ADAPTER_PATH = "experiments/gemma2_2b_v40_subsample_5k"

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded!\n")

# Test 1: Training format (CORRECT)
prompt1 = "Generate a SQLI payload that bypasses WAF.\n\nIMPORTANT: Generate ONLY the payload code. Do not provide explanations."

# Test 2: Eval format (WRONG)
prompt2 = "Generate a SQLI payload using Boolean OR to bypass WAF. Output ONLY the payload string."

for i, prompt in enumerate([prompt1, prompt2], 1):
    print(f"{'='*70}")
    print(f"Test {i}: {'TRAINING FORMAT' if i == 1 else 'EVAL FORMAT'}")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt
    if prompt in generated:
        generated = generated.replace(prompt, "").strip()
    
    print(f"Raw Output: {generated[:300]}")
    print(f"\n")

print("Done!")
