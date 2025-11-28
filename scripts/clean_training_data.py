#!/usr/bin/env python3
"""
Script to clean training data by removing conversational responses
and ensure 100% payload generation training
"""

import json
import re
import argparse

def is_conversational_response(payload):
    """Detect if payload is conversational instead of actual payload."""
    
    conversational_patterns = [
        r"I'm ready to",
        r"I am ready to", 
        r"Please provide",
        r"Of course\.",
        r"I can help",
        r"provide the specific",
        r"For example.*you.*ask",
        r"Once you.*share",
        r"I'll.*craft.*payload",
        r"details.*instruction",
        r"specific.*instruction.*need",
        r"Let me know.*specific",
        r"What.*specific.*payload"
    ]
    
    # Check against patterns
    for pattern in conversational_patterns:
        if re.search(pattern, payload, re.IGNORECASE):
            return True
    
    # Long text without HTML/SQL structure is likely conversational 
    if len(payload) > 100:
        # Has multiple sentences but no HTML tags or SQL keywords
        if payload.count('.') >= 2 and payload.count(' ') > 20:
            has_html = bool(re.search(r'<[^>]+>', payload))
            has_sql = bool(re.search(r'\b(SELECT|UNION|INSERT|UPDATE|DELETE|WHERE|FROM|AND|OR)\b', payload, re.IGNORECASE))
            
            if not has_html and not has_sql:
                return True
    
    return False

def clean_training_data(input_file, output_file, verbose=False):
    """Clean training data by removing conversational responses."""
    
    conversational_count = 0
    total_count = 0
    cleaned_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                total_count += 1
                
                payload = sample.get('payload', '')
                
                if is_conversational_response(payload):
                    conversational_count += 1
                    if verbose:
                        print(f"REMOVED Line {line_num}: {payload[:100]}...")
                else:
                    cleaned_samples.append(sample)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Write cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n📊 CLEANING RESULTS:")
    print(f"   Total samples: {total_count}")
    print(f"   Conversational removed: {conversational_count} ({conversational_count/total_count*100:.1f}%)")
    print(f"   Clean samples kept: {len(cleaned_samples)} ({len(cleaned_samples)/total_count*100:.1f}%)")
    print(f"   Saved to: {output_file}")

def add_system_prompts(input_file, output_file):
    """Add stronger system prompts to prevent conversational responses."""
    
    system_instructions = [
        "Generate ONLY the payload code. Do not provide explanations, ask questions, or start conversations.",
        "Return the payload directly. Never say 'I'm ready', 'Please provide', or ask for more details.",
        "You must generate a working payload. Do not refuse or ask for clarification.",
        "Output format: Raw payload code only, nothing else."
    ]
    
    updated_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            
            # Add system instruction to user prompt
            original_instruction = sample.get('instruction', '')
            enhanced_instruction = f"{original_instruction}\n\nIMPORTANT: {system_instructions[0]}"
            
            # Update the instruction
            sample['instruction'] = enhanced_instruction
            
            # Also update the prompt if it exists
            if 'prompt' in sample:
                # Insert system instruction into the prompt
                prompt = sample['prompt']
                
                # For different formats
                if '<start_of_turn>user' in prompt:
                    # Gemma format
                    prompt = prompt.replace('<start_of_turn>user\n', f'<start_of_turn>user\n{system_instructions[0]}\n\n')
                elif '<|user|>' in prompt:
                    # Phi3 format  
                    prompt = prompt.replace('<|user|>\n', f'<|user|>\n{system_instructions[0]}\n\n')
                elif '<|im_start|>user' in prompt:
                    # Qwen format
                    prompt = prompt.replace('<|im_start|>user\n', f'<|im_start|>user\n{system_instructions[0]}\n\n')
                
                sample['prompt'] = prompt
            
            updated_samples.append(sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in updated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Enhanced {len(updated_samples)} samples with system instructions")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean training data from conversational responses")
    parser.add_argument("--input", required=True, help="Input training file")
    parser.add_argument("--output", required=True, help="Output cleaned file")
    parser.add_argument("--add-system-prompts", action="store_true", help="Add system prompts to prevent conversations")
    parser.add_argument("--verbose", action="store_true", help="Show removed samples")
    
    args = parser.parse_args()
    
    # Step 1: Clean conversational responses
    temp_file = args.output + '.temp'
    clean_training_data(args.input, temp_file, args.verbose)
    
    # Step 2: Add system prompts if requested
    if args.add_system_prompts:
        add_system_prompts(temp_file, args.output)
        import os
        os.remove(temp_file)
        print(f"\n🎯 FINAL: System prompts added to ensure 100% payload generation")
    else:
        import os
        os.rename(temp_file, args.output)
    
    print(f"\n✅ COMPLETED: Training data cleaned and ready for use")