import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import argparse
import json
import os

def generate_payloads(model, tokenizer, prompt, num_payloads=10, max_new_tokens=100):
    """
    Generates SQL injection payloads using the fine-tuned model.
    Generates in batches to manage memory.
    """
    all_generated_texts = []
    batch_size = 10 # Generate 10 payloads at a time
    
    for i in range(0, num_payloads, batch_size):
        current_batch_size = min(batch_size, num_payloads - i)
        if current_batch_size == 0:
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=current_batch_size, # Use current_batch_size
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_texts = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        all_generated_texts.extend([text.strip() for text in generated_texts])
        print(f"Generated {len(all_generated_texts)}/{num_payloads} payloads...")

    return all_generated_texts

def main():
    parser = argparse.ArgumentParser(description="Generate SQLi payloads using a fine-tuned SFT model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the SFT configuration YAML file.")
    parser.add_argument("--num_payloads", type=int, default=100, help="Number of payloads to generate.")
    parser.add_argument("--output_file", type=str, default="generated_sft_payloads.jsonl", help="Output file for generated payloads.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_model_id = config["model_name"]
    sft_adapter_path = config["output_dir"]

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        # device_map="auto" # Removed device_map="auto" to handle device placement explicitly
    )
    # Explicitly move model to device after loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading SFT adapter from: {sft_adapter_path}")
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    model.eval() # Set model to evaluation mode

    # --- DEBUG PRINTS ---
    print("\n--- Debugging Adapter Loading ---")
    print(f"Model PEFT config: {model.peft_config}")
    print("Keys in model.state_dict():")
    for k in model.state_dict().keys():
        print(f"- {k}")
    print("--- End Debugging Adapter Loading ---\n")


    # Define the prompt for generation
    # This should match the prompt format used during SFT
    prompt_template = "<|system|> You are a helpful AI assistant. <|end|><|user|> Generate a single, effective MySQL SQL injection payload. Output only the payload itself, with no other text or explanation. <|end|><|assistant|> Payload:"

    print(f"Generating {args.num_payloads} payloads...")
    generated_payloads = generate_payloads(
        model,
        tokenizer,
        prompt_template,
        num_payloads=args.num_payloads
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w") as f:
        for payload in generated_payloads:
            json.dump({"payload": payload}, f)
            f.write("\n")
    print(f"Generated payloads saved to {args.output_file}")

if __name__ == "__main__":
    main()
