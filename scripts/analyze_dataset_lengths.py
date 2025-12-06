#!/usr/bin/env python3
"""Analyze average token lengths across different datasets"""
import json
from transformers import AutoTokenizer

# Load tokenizer (using Gemma as reference)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

datasets = {
    "Phase 1 (Basic)": "data/processed/red_phase1_enriched_v2.jsonl",
    "Phase 2 (Reasoning)": "data/processed/red_phase2_reasoning.jsonl",
    "Phase 3 10k": "data/processed/red_phase3_lightweight_10k.jsonl",
    "Phase 3 20k": "data/processed/red_phase3_lightweight.jsonl",
}

print("="*80)
print("Dataset Length Analysis (Token Count)")
print("="*80)

for name, path in datasets.items():
    try:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample first 100 for speed
                    break
                samples.append(json.loads(line))
        
        if not samples:
            continue
        
        # Calculate token lengths
        prompt_lengths = []
        response_lengths = []
        total_lengths = []
        
        for sample in samples:
            if 'messages' in sample:
                prompt = sample['messages'][0]['content']
                response = sample['messages'][1]['content']
            else:
                continue
            
            prompt_tokens = len(tokenizer.encode(prompt))
            response_tokens = len(tokenizer.encode(response))
            total_tokens = prompt_tokens + response_tokens
            
            prompt_lengths.append(prompt_tokens)
            response_lengths.append(response_tokens)
            total_lengths.append(total_tokens)
        
        avg_prompt = sum(prompt_lengths) / len(prompt_lengths)
        avg_response = sum(response_lengths) / len(response_lengths)
        avg_total = sum(total_lengths) / len(total_lengths)
        max_total = max(total_lengths)
        
        print(f"\n{name}:")
        print(f"  Samples analyzed: {len(samples)}")
        print(f"  Avg prompt length: {avg_prompt:.0f} tokens")
        print(f"  Avg response length: {avg_response:.0f} tokens")
        print(f"  Avg total length: {avg_total:.0f} tokens")
        print(f"  Max total length: {max_total} tokens")
        
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")

print("\n" + "="*80)
print("VRAM Usage Explanation:")
print("="*80)
print("VRAM = Model Size (4bit) + Optimizer States + Gradients + Activations + Batch")
print("- Phase 1 (~150 tokens): Minimal activations → ~8GB VRAM (Gemma 2B)")
print("- Phase 2 (~700 tokens): 4.6x more activations → ~12GB VRAM")
print("- Phase 3 (~300 tokens): 2x more than Phase 1 → ~10GB VRAM")
print("\nLonger sequences = more intermediate activations stored = more VRAM!")
print("="*80)
