#!/usr/bin/env python3
"""
Quick test for Qwen 2.5 7B Phase 2 adapter (lightweight enhanced)
Tests payload generation for a few XSS/SQLi scenarios
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Config
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "experiments/red_phase3_lightweight_enhanced_qwen3b"  # Phase 2 (old name phase3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🧪 Testing Qwen 2.5 7B Phase 2 Adapter (Lightweight Enhanced)")
print("=" * 80)
print(f"\n📦 Model: {MODEL_NAME}")
print(f"📁 Adapter: {ADAPTER_PATH}")
print(f"🔧 Device: {DEVICE}")

# Check adapter exists
if not os.path.exists(ADAPTER_PATH):
    print(f"\n❌ ERROR: Adapter not found at {ADAPTER_PATH}")
    print("\nAvailable adapters in experiments/:")
    for item in os.listdir("experiments"):
        if os.path.isdir(f"experiments/{item}") and "qwen" in item.lower():
            print(f"  - {item}")
    exit(1)

print("\n" + "=" * 80)
print("Loading model and adapter...")
print("=" * 80)

# Load base model
print("\n[1/3] Loading base model (4-bit quantization)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load tokenizer
print("[2/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load adapter
print(f"[3/3] Loading adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print("\n✅ Model loaded successfully!")

# Test scenarios
test_cases = [
    {
        "scenario": "XSS - Reflected in HTML",
        "instruction": "Generate an XSS payload for parameter 'name' in URL http://target.com/search?name=USER_INPUT. The input is reflected in <h1>Results for: USER_INPUT</h1>.",
        "expected_technique": "XSS via event handler or script tag"
    },
    {
        "scenario": "SQLi - Numeric parameter",
        "instruction": "Generate a SQL injection payload for numeric parameter 'id' in http://target.com/product?id=123. The query is SELECT * FROM products WHERE id = USER_INPUT.",
        "expected_technique": "Union-based or boolean-based SQLi"
    },
    {
        "scenario": "XSS - Inside JavaScript",
        "instruction": "Generate an XSS payload for parameter 'callback' in http://api.target.com/data?callback=USER_INPUT. Response: <script>USER_INPUT({data: 'value'})</script>",
        "expected_technique": "JavaScript context escape"
    },
    {
        "scenario": "SQLi - String with quotes",
        "instruction": "Generate a SQL injection payload for parameter 'username' in login form. Query: SELECT * FROM users WHERE username='USER_INPUT' AND password='...'",
        "expected_technique": "Quote escape or comment injection"
    }
]

print("\n" + "=" * 80)
print("🎯 Running Test Cases")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'─' * 80}")
    print(f"Test {i}/{len(test_cases)}: {test['scenario']}")
    print(f"{'─' * 80}")
    print(f"\n📝 Instruction:")
    print(f"   {test['instruction'][:100]}...")
    
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
            max_new_tokens=150,
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
    
    print(f"\n✅ Generated Payload:")
    print(f"   {payload}")
    print(f"\n💡 Expected: {test['expected_technique']}")

print("\n" + "=" * 80)
print("✅ All tests completed!")
print("=" * 80)

print("\n📊 Model Info:")
print(f"   Total params: {model.num_parameters():,}")
print(f"   Trainable params: {model.num_parameters(only_trainable=True):,}")
print(f"   Adapter type: LoRA")

print("\n💾 Adapter Location:")
print(f"   {os.path.abspath(ADAPTER_PATH)}")

print("\n")
