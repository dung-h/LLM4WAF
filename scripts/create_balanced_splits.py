"""
Create balanced train/val/test splits for SFT experiment
Generates separate files for each model format (Gemma, Phi-3, Qwen)
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict


def build_prompt_gemma(sample: Dict) -> str:
    """Build Gemma 2 format prompt"""
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    constraints = sample.get("constraints", "")
    payload = sample.get("payload", "")
    
    user_parts = []
    if instruction:
        user_parts.append(instruction)
    if context:
        user_parts.append(f"Context: {context}")
    if constraints:
        user_parts.append(f"Constraints: {constraints}")
    
    user_block = "\n\n".join(user_parts)
    
    return f"<start_of_turn>user\n{user_block}<end_of_turn>\n<start_of_turn>model\n{payload}<end_of_turn>"


def build_prompt_phi3(sample: Dict) -> str:
    """Build Phi-3 format prompt"""
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    constraints = sample.get("constraints", "")
    payload = sample.get("payload", "")
    
    user_parts = []
    if instruction:
        user_parts.append(instruction)
    if context:
        user_parts.append(f"Context: {context}")
    if constraints:
        user_parts.append(f"Constraints: {constraints}")
    
    user_block = "\n\n".join(user_parts)
    
    return f"<|user|>\n{user_block}<|end|>\n<|assistant|>\n{payload}<|end|>"


def build_prompt_qwen(sample: Dict) -> str:
    """Build Qwen 2.5 format prompt"""
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    constraints = sample.get("constraints", "")
    payload = sample.get("payload", "")
    
    user_parts = []
    if instruction:
        user_parts.append(instruction)
    if context:
        user_parts.append(f"Context: {context}")
    if constraints:
        user_parts.append(f"Constraints: {constraints}")
    
    user_block = "\n\n".join(user_parts)
    
    return f"<|im_start|>user\n{user_block}<|im_end|>\n<|im_start|>assistant\n{payload}<|im_end|>"


def load_dataset(filepath: str) -> List[Dict]:
    """Load JSONL dataset and convert to unified format"""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                
                # Handle v29_enriched format (messages array)
                if 'messages' in item:
                    instruction = item['messages'][0]['content']
                    payload = item['messages'][1]['content']
                    waf_result = item.get('result', 'unknown')
                    
                    samples.append({
                        'instruction': instruction,
                        'payload': payload,
                        'status': 'blocked' if waf_result == 'blocked' else 'passed',
                        'source': 'v29_enriched'
                    })
                # Handle old format (payload field)
                elif 'payload' in item and item.get('payload'):
                    samples.append(item)
    
    return samples


def create_balanced_split(samples: List[Dict], size: int, passed_ratio: float = 0.5) -> List[Dict]:
    """Create balanced split with specified passed/blocked ratio"""
    # Separate by status (assuming 'status' field or 'type' field indicates passed/blocked)
    passed = [s for s in samples if s.get('status') == 'passed' or 'passed' in s.get('source', '').lower()]
    blocked = [s for s in samples if s.get('status') == 'blocked' or 'blocked' in s.get('source', '').lower()]
    
    # If no status field, use 50/50 random split
    if not passed and not blocked:
        random.shuffle(samples)
        mid = len(samples) // 2
        passed = samples[:mid]
        blocked = samples[mid:]
    
    # Calculate target counts
    num_passed = int(size * passed_ratio)
    num_blocked = size - num_passed
    
    # Sample with replacement if needed
    if len(passed) < num_passed:
        selected_passed = random.choices(passed, k=num_passed)
    else:
        selected_passed = random.sample(passed, num_passed)
    
    if len(blocked) < num_blocked:
        selected_blocked = random.choices(blocked, k=num_blocked)
    else:
        selected_blocked = random.sample(blocked, num_blocked)
    
    # Combine and shuffle
    split = selected_passed + selected_blocked
    random.shuffle(split)
    
    return split


def save_formatted_dataset(samples: List[Dict], output_path: Path, format_type: str):
    """Save dataset in specific model format"""
    format_funcs = {
        "gemma": build_prompt_gemma,
        "phi3": build_prompt_phi3,
        "qwen": build_prompt_qwen
    }
    
    format_func = format_funcs.get(format_type)
    
    if not format_func:
        raise ValueError(f"Unknown format: {format_type}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Build formatted prompt
            formatted = {
                "prompt": format_func(sample),
                "instruction": sample.get("instruction", ""),
                "payload": sample.get("payload", ""),
                "type": sample.get("type", ""),
                "source": sample.get("source", "")
            }
            
            f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
    
    print(f"  OK {output_path.name}: {len(samples)} samples")


def main():
    parser = argparse.ArgumentParser(description="Create balanced splits for SFT experiment")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2000, 4000, 8000], help="Training set sizes")
    parser.add_argument("--test_size", type=int, default=200, help="Test set size")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--passed_ratio", type=float, default=0.5, help="Passed samples ratio")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("="*80)
    print("CREATING BALANCED SPLITS")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading: {args.input}")
    samples = load_dataset(args.input)
    print(f"  Total samples: {len(samples)}")
    
    # Count by type
    types = Counter(s.get('type', 'unknown') for s in samples)
    print(f"  By type: {dict(types)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test set (shared across all experiments)
    print(f"\nCreating test set ({args.test_size} samples)...")
    test_samples = create_balanced_split(samples, args.test_size, args.passed_ratio)
    
    # Remove test samples from pool
    test_ids = set(id(s) for s in test_samples)
    remaining_samples = [s for s in samples if id(s) not in test_ids]
    
    print(f"  Remaining samples: {len(remaining_samples)}")
    
    # Save test set in all formats
    for fmt in ["gemma", "phi3", "qwen"]:
        test_path = output_dir / f"test_200_{fmt}.jsonl"
        save_formatted_dataset(test_samples, test_path, fmt)
    
    # Create train/val splits for each size
    for size in args.sizes:
        size_k = size // 1000
        print(f"\nCreating {size_k}K splits...")
        
        # Create training set
        train_samples = create_balanced_split(remaining_samples, size, args.passed_ratio)
        
        # Create validation set
        val_size = int(size * args.val_ratio)
        val_samples = create_balanced_split(remaining_samples, val_size, args.passed_ratio)
        
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val: {len(val_samples)} samples")
        
        # Save in all formats
        for fmt in ["gemma", "phi3", "qwen"]:
            train_path = output_dir / f"train_{size_k}k_{fmt}.jsonl"
            val_path = output_dir / f"val_{size_k}k_{fmt}.jsonl"
            
            save_formatted_dataset(train_samples, train_path, fmt)
            save_formatted_dataset(val_samples, val_path, fmt)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Test set: {args.test_size} samples (3 formats)")
    print(f"Training sets: {len(args.sizes)} sizes × 3 formats = {len(args.sizes) * 3} files")
    print(f"Validation sets: {len(args.sizes)} sizes × 3 formats = {len(args.sizes) * 3} files")
    print(f"\nTotal files created: {3 + len(args.sizes) * 3 * 2}")
    
    # List files
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")
    
    print("\nSplits created successfully!")


if __name__ == "__main__":
    main()
