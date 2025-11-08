#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json

print('🚀 Loading v5_fixed model...')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
base = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it', quantization_config=bnb_config, device_map='auto')
model = PeftModel.from_pretrained(base, 'experiments/red_gemma2_v5_fixed/adapter')
model.eval()

print('✅ Model loaded!')
print('\n📝 Generating 30 test payloads...\n')

# Load test prompts
with open('data/processed/red_test_v5_simple.jsonl', 'r') as f:
    prompts = [json.loads(line) for line in f][:30]

payloads = []
for i, data in enumerate(prompts, 1):
    prompt_text = f"Instruction: {data['instruction']}\n\nPayload:"
    inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True, top_p=0.9)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract payload
    if 'Payload:' in text:
        payload = text.split('Payload:')[-1].strip().split('\n')[0].strip()
    else:
        payload = text.strip()
    
    payloads.append(payload)
    print(f'[{i}/30] {payload[:80]}')

# Save
with open('results/v5_fixed_payloads_30.txt', 'w') as f:
    for p in payloads:
        f.write(p + '\n')

print(f'\n✅ Saved {len(payloads)} payloads!')
print('📁 File: results/v5_fixed_payloads_30.txt')

# Quick preview first 5
print('\n📋 First 5 payloads:')
for i, p in enumerate(payloads[:5], 1):
    print(f'{i}. {p}')
