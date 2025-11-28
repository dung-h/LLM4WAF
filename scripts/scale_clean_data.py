#!/usr/bin/env python3
"""
Script to clean v29 enriched data format (messages structure)
"""

import json
import re

def is_conversational_response(content):
    """Detect if assistant content is conversational."""
    
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
        if re.search(pattern, content, re.IGNORECASE):
            return True
    
    # Long text without HTML/SQL structure is likely conversational 
    if len(content) > 100:
        if content.count('.') >= 2 and content.count(' ') > 20:
            has_html = bool(re.search(r'<[^>]+>', content))
            has_sql = bool(re.search(r'\b(SELECT|UNION|INSERT|UPDATE|DELETE|WHERE|FROM|AND|OR)\b', content, re.IGNORECASE))
            
            if not has_html and not has_sql:
                return True
    
    return False

def clean_v29_data():
    """Clean v29 enriched data format."""
    
    input_file = "data/processed/red_v29_enriched.jsonl"
    output_file = "data/processed/red_v29_cleaned_messages.jsonl"
    
    conversational_count = 0
    total_count = 0
    cleaned_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            sample = json.loads(line.strip())
            total_count += 1
            
            # Extract assistant content
            messages = sample.get('messages', [])
            assistant_content = None
            
            for msg in messages:
                if msg.get('role') == 'assistant':
                    assistant_content = msg.get('content', '')
                    break
            
            if assistant_content and is_conversational_response(assistant_content):
                conversational_count += 1
                print(f"REMOVED Line {line_num}: {assistant_content[:100]}...")
            else:
                # Add system instruction to user message
                for msg in messages:
                    if msg.get('role') == 'user':
                        original_content = msg.get('content', '')
                        enhanced_content = f"{original_content}\n\nIMPORTANT: Generate ONLY the payload code. Do not provide explanations, ask questions, or start conversations."
                        msg['content'] = enhanced_content
                        break
                
                cleaned_samples.append(sample)
    
    # Write cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n📊 CLEANING RESULTS:")
    print(f"   Total samples: {total_count}")
    print(f"   Conversational removed: {conversational_count} ({conversational_count/total_count*100:.1f}%)")
    print(f"   Clean samples kept: {len(cleaned_samples)} ({len(cleaned_samples)/total_count*100:.1f}%)")
    print(f"   Saved to: {output_file}")
    
    return len(cleaned_samples)

def convert_to_training_formats(clean_count):
    """Convert cleaned messages to model-specific training formats."""
    
    input_file = "data/processed/red_v29_cleaned_messages.jsonl"
    
    # Output files for different models
    output_files = {
        "gemma": "data/processed/train_full_gemma_cleaned.jsonl",
        "phi3": "data/processed/train_full_phi3_cleaned.jsonl", 
        "qwen": "data/processed/train_full_qwen_cleaned.jsonl"
    }
    
    for model_type, output_file in output_files.items():
        converted_samples = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                
                # Extract user and assistant content
                user_content = ""
                assistant_content = ""
                
                for msg in sample['messages']:
                    if msg['role'] == 'user':
                        user_content = msg['content']
                    elif msg['role'] == 'assistant':
                        assistant_content = msg['content']
                
                # Convert to model-specific format
                if model_type == "gemma":
                    prompt = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n{assistant_content}<end_of_turn>"
                elif model_type == "phi3":
                    prompt = f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n{assistant_content}<|end|>"
                else:  # qwen
                    prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
                
                # Create training sample
                training_sample = {
                    "prompt": prompt,
                    "instruction": user_content.split("\n\nIMPORTANT:")[0],  # Remove system prompt for instruction field
                    "payload": assistant_content,
                    "type": sample.get('attack_type', ''),
                    "source": "v29_cleaned"
                }
                
                converted_samples.append(training_sample)
        
        # Write converted data
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in converted_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✅ {model_type.upper()}: {len(converted_samples)} samples → {output_file}")
    
    return len(converted_samples)

if __name__ == "__main__":
    print("🧹 CLEANING FULL V29 DATASET...")
    clean_count = clean_v29_data()
    
    print(f"\n🔄 CONVERTING TO TRAINING FORMATS...")
    final_count = convert_to_training_formats(clean_count)
    
    print(f"\n🎯 SCALING COMPLETED:")
    print(f"   From: 4K samples (with 131 conversational)")
    print(f"   To: {final_count:,} samples (0 conversational)")
    print(f"   Scale factor: {final_count/4000:.1f}x")
    print(f"   Ready for 7B training! 🚀")