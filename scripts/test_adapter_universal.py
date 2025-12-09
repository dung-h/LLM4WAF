#!/usr/bin/env python3
"""
Universal model adapter testing script
Supports: Gemma 2B, Phi-3 Mini, Qwen 3B/7B
Auto-detects model from adapter path
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import argparse
import json

def detect_model_from_adapter(adapter_path):
    """Detect base model from adapter config"""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_model = config.get("base_model_name_or_path", "")
    
    if "gemma" in base_model.lower():
        return "google/gemma-2-2b-it"
    elif "phi-3" in base_model.lower():
        return "microsoft/Phi-3-mini-4k-instruct"
    elif "qwen" in base_model.lower():
        if "7b" in base_model.lower():
            return "Qwen/Qwen2.5-7B-Instruct"
        else:
            return "Qwen/Qwen2.5-3B-Instruct"
    
    return base_model

def main():
    parser = argparse.ArgumentParser(description="Test model adapter with sample payloads")
    parser.add_argument("--adapter", required=True, help="Path to adapter folder")
    parser.add_argument("--model", help="Base model name (auto-detect if not provided)")
    parser.add_argument("--num-tests", type=int, default=3, help="Number of test cases (default: 3)")
    args = parser.parse_args()
    
    adapter_path = args.adapter
    
    # Check adapter exists
    if not os.path.exists(adapter_path):
        print(f"❌ ERROR: Adapter not found at {adapter_path}")
        return
    
    # Auto-detect or use provided model
    if args.model:
        model_name = args.model
        print(f"📦 Using provided model: {model_name}")
    else:
        model_name = detect_model_from_adapter(adapter_path)
        if not model_name:
            print("❌ ERROR: Could not detect model from adapter config")
            return
        print(f"🔍 Auto-detected model: {model_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("🧪 Universal Model Adapter Tester")
    print("=" * 80)
    print(f"\n📁 Adapter: {adapter_path}")
    print(f"🤖 Model: {model_name}")
    print(f"🔧 Device: {device}")
    
    # Load model
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)
    
    print("\n[1/3] Loading base model (4-bit quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print("[2/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[3/3] Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    print("\n✅ Model loaded successfully!")
    
    # Test scenarios
    test_cases = [
        {
            "scenario": "XSS - Reflected in HTML",
            "instruction": "Generate an XSS payload for parameter 'name' in URL http://target.com/search?name=USER_INPUT. The input is reflected in <h1>Results for: USER_INPUT</h1>.",
        },
        {
            "scenario": "SQLi - Numeric parameter",
            "instruction": "Generate a SQL injection payload for numeric parameter 'id' in http://target.com/product?id=123. The query is SELECT * FROM products WHERE id = USER_INPUT.",
        },
        {
            "scenario": "XSS - Inside JavaScript",
            "instruction": "Generate an XSS payload for parameter 'callback' in http://api.target.com/data?callback=USER_INPUT. Response: <script>USER_INPUT({data: 'value'})</script>",
        },
        {
            "scenario": "SQLi - String with quotes",
            "instruction": "Generate a SQL injection payload for parameter 'username' in login form. Query: SELECT * FROM users WHERE username='USER_INPUT' AND password='...'",
        },
        {
            "scenario": "XSS - Event handler",
            "instruction": "Generate an XSS payload using event handlers for parameter 'q' reflected in <input value='USER_INPUT'>.",
        },
    ]
    
    # Limit test cases
    test_cases = test_cases[:args.num_tests]
    
    print("\n" + "=" * 80)
    print(f"🎯 Running {len(test_cases)} Test Cases")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}/{len(test_cases)}: {test['scenario']}")
        print(f"{'─' * 80}")
        print(f"\n📝 Instruction:")
        print(f"   {test['instruction'][:80]}...")
        
        # Format as chat
        messages = [
            {"role": "user", "content": test["instruction"]}
        ]
        
        # Generate
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print(f"\n⏳ Generating payload...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only assistant response
        if "assistant" in full_output.lower():
            payload = full_output.split("assistant")[-1].strip()
        else:
            payload = full_output.split(test["instruction"])[-1].strip()
        
        # Clean up special tokens
        for token in ["<|im_end|>", "<|end|>", "<start_of_turn>", "<end_of_turn>"]:
            payload = payload.replace(token, "").strip()
        
        print(f"\n✅ Generated Payload:")
        print(f"   {payload}")
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)
    
    print("\n📊 Model Info:")
    print(f"   Total params: {model.num_parameters():,}")
    print(f"   Trainable params: {model.num_parameters(only_trainable=True):,}")
    print(f"   Adapter type: LoRA")
    print(f"   Base model: {model_name}")
    
    print("\n")

if __name__ == "__main__":
    main()
