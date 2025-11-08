#!/usr/bin/env python3
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json

# Clear GPU cache BEFORE loading anything
print('🧹 Clearing GPU cache...')
torch.cuda.empty_cache()
gc.collect()

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

print('📦 Loading base model...')
base = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2-2b-it', 
    quantization_config=bnb_config, 
    device_map='auto',
    torch_dtype=torch.float16
)

# Clear cache again before loading adapter
print('🧹 Clearing cache before PEFT adapter...')
torch.cuda.empty_cache()
gc.collect()

print('🔧 Loading PEFT adapter...')
model = PeftModel.from_pretrained(
    base, 
    'experiments/red_gemma2_v5_fixed/adapter',
    torch_dtype=torch.float16
)
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
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.7, 
            do_sample=True, 
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract payload
    if 'Payload:' in text:
        payload = text.split('Payload:')[-1].strip().split('\n')[0].strip()
    else:
        payload = text.strip()
    
    payloads.append(payload)
    print(f'[{i}/30] {payload[:80]}')
    
    # Clear cache every 10 generations to prevent memory buildup
    if i % 10 == 0:
        torch.cuda.empty_cache()

# Save
with open('results/v5_fixed_payloads_30.txt', 'w') as f:
    for p in payloads:
        f.write(p + '\n')

print(f'\n✅ Saved 30 payloads to results/v5_fixed_payloads_30.txt')
