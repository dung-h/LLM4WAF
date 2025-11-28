import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel

def build_prompt_gemma(sample: dict) -> str:
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    constraints = sample.get("constraints", "")
    user_parts = [p for p in [instruction, f"Context: {context}" if context else "", f"Constraints: {constraints}" if constraints else ""] if p]
    user_block = "\n\n".join(user_parts)
    return f"<start_of_turn>user\n{user_block}<end_of_turn>\n<start_of_turn>model\n"

def build_prompt_phi3(sample: dict) -> str:
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    constraints = sample.get("constraints", "")
    user_parts = [p for p in [instruction, f"Context: {context}" if context else "", f"Constraints: {constraints}" if constraints else ""] if p]
    user_block = "\n\n".join(user_parts)
    return f"<|user|>\n{user_block}<|end|>\n<|assistant|>\n"

def build_prompt_phi3(sample: dict) -> str:
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    constraints = sample.get("constraints", "")
    user_parts = [p for p in [instruction, f"Context: {context}" if context else "", f"Constraints: {constraints}" if constraints else ""] if p]
    user_block = "\n\n".join(user_parts)
    return f"<|user|>\n{user_block}<|end|>\n<|assistant|>\n"

def build_prompt_qwen(sample: dict) -> str:
    """Build Qwen chat format prompt"""
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    constraints = sample.get("constraints", "")
    user_parts = [p for p in [instruction, f"Context: {context}" if context else "", f"Constraints: {constraints}" if constraints else ""] if p]
    user_block = "\n\n".join(user_parts)
    return f"<|im_start|>user\n{user_block}<|im_end|>\n<|im_start|>assistant\n"

def strip_assistant_from_prompt(raw_prompt: str, fmt: str) -> str:
    """Return only user portion by removing assistant/answer section, then add assistant prefix."""
    if fmt == "gemma":
        if "<start_of_turn>model" in raw_prompt:
            user_part = raw_prompt.split("<start_of_turn>model")[0].strip()
            return user_part + "\n<start_of_turn>model\n"
    elif fmt == "phi3":
        if "<|assistant|>" in raw_prompt:
            user_part = raw_prompt.split("<|assistant|>")[0].strip()
            return user_part + "\n<|assistant|>\n"
    else:  # qwen
        if "<|im_start|>assistant" in raw_prompt:
            user_part = raw_prompt.split("<|im_start|>assistant")[0].strip()
            return user_part + "\n<|im_start|>assistant\n"
    return raw_prompt

def main():
    parser = argparse.ArgumentParser(description="Standard tool to evaluate RED models.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--format", choices=["gemma", "phi3", "qwen"], required=True)
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base model (be careful with 4bit).")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    
    print(f"Loading base model: {args.base_model}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    
    if args.merge:
        print("Merging adapter...")
        model = model.merge_and_unload()
    
    model.eval()

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset("json", data_files=args.dataset, split="train")
    
    num = min(args.num_samples, len(ds))
    for i in range(num):
        sample = ds[i]
        
        # Get reference payload for debugging
        reference = sample.get("payload") or sample.get("answer") or None
        
        # Prefer provided prompt, but strip prefilled answer
        if "prompt" in sample and sample["prompt"]:
            raw_prompt = sample["prompt"]
            prompt = strip_assistant_from_prompt(raw_prompt, args.format)
            # Fallback if strip produces empty
            if not prompt or not prompt.strip():
                if args.format == "gemma":
                    prompt = build_prompt_gemma(sample)
                elif args.format == "phi3":
                    prompt = build_prompt_phi3(sample)
                else:  # qwen
                    prompt = build_prompt_qwen(sample)
        else:
            if args.format == "gemma":
                prompt = build_prompt_gemma(sample)
            elif args.format == "phi3":
                prompt = build_prompt_phi3(sample)
            else:  # qwen
                prompt = build_prompt_qwen(sample)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8
            )
            
        raw = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        
        # Cleanup based on format
        if args.format == "gemma":
            clean = raw.replace("<end_of_turn>", "").replace("<eos>", "")
        elif args.format == "phi3":
            clean = raw.replace("<|end|>", "").replace("<|assistant|>", "").replace(tokenizer.eos_token, "")
        else:  # qwen
            clean = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace(tokenizer.eos_token, "")
            
        print("\n" + "="*80)
        print(f"SAMPLE {i}")
        print(f"PROMPT:\n{prompt}")
        gen_out = clean.strip()
        print(f"GENERATED:\n{gen_out}")
        
        if not gen_out:
            print("--- NOTE: generation empty. Reference payload (if available):")
            if reference:
                print(reference)
            else:
                print("<no reference in dataset>")

if __name__ == "__main__":
    main()
