import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria

def main():
    parser = argparse.ArgumentParser(description="Validate a fine-tuned SFT model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct", # Updated default model name
        help="The base model name.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for private models.",
    )
    args = parser.parse_args()

    print(f"Loading base model: {args.model_name}")
    # Load model to CPU first to avoid meta device issues with adapter loading
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None, # Do not use auto device map initially
        token=args.hf_token,
    )

    print(f"Loading adapter from: {args.adapter_path}")
    model.load_adapter(args.adapter_path)

    # Move model to GPU if available, otherwise keep on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model moved to device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Set padding side for generation

    # Diverse test prompts, including the strict output instruction
    test_prompts = [
        "Generate a time-based blind SQL injection payload for a MySQL database to determine if a user exists. Output only the payload, with no other text or explanation.",
        "Craft an error-based SQL injection payload for a PostgreSQL database to extract the current database name. Output only the payload, with no other text or explanation.",
        "Provide a union-based SQL injection payload for an MSSQL server to retrieve all table names. Output only the payload, with no other text or explanation.",
        "Generate a reflected XSS payload that executes `alert(document.domain)`. Output only the payload, with no other text or explanation.",
        "Develop an XSS payload that uses a `<img>` tag with an `onerror` event. Output only the payload, with no other text or explanation.",
    ]

    print("\n--- Generating Payloads ---")
    model.eval()
    with torch.no_grad():
        for i, prompt_text in enumerate(test_prompts):
            # Phi-3 instruct format
            messages = [
                {"role": "user", "content": f"{prompt_text}\n"}, # Add newline
            ]
            # Manually construct the prompt string to match the training format
            # The model is trained to generate the assistant's response, which starts after <|assistant|>\n
            # So, the input to the model should be everything up to <|assistant|>\n
            prompt_for_model = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
            input_ids = tokenizer.encode(prompt_for_model, return_tensors="pt", add_special_tokens=False).to(model.device)
            
            print(f"--- Decoded Input IDs for Prompt {i+1} ---")
            print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
            print("------------------------------------------")

            # Stopping criteria should include the eos_token_id
            outputs = model.generate(
                input_ids,
                max_new_tokens=100, # Reduced for quicker testing
                do_sample=False, # Use greedy decoding
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id, # Simplify back to just eos_token_id
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode and print
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract the text after the first "<|assistant|> " and aggressively truncate
            try:
                assistant_tag = "<|assistant|> "
                # Find the start of the *first* assistant response
                first_assistant_start_index = decoded_output.find(assistant_tag)
                
                if first_assistant_start_index != -1:
                    # Adjust to get the content *after* the assistant tag
                    payload_start = first_assistant_start_index + len(assistant_tag)
                    content_after_first_assistant = decoded_output[payload_start:]

                    # Find the first newline, or the start of a new user/assistant turn
                    stop_indices = []
                    newline_index = content_after_first_assistant.find('\n')
                    if newline_index != -1:
                        stop_indices.append(newline_index)
                    
                    user_tag_index = content_after_first_assistant.find("<|user|>")
                    if user_tag_index != -1:
                        stop_indices.append(user_tag_index)
                    
                    next_assistant_tag_index = content_after_first_assistant.find("<|assistant|>")
                    if next_assistant_tag_index != -1:
                        stop_indices.append(next_assistant_tag_index)

                    if stop_indices:
                        # Take the minimum of all found stop indices
                        truncation_index = min(stop_indices)
                        generated_text = content_after_first_assistant[:truncation_index].strip()
                    else:
                        # If no stop markers found, take everything until max_new_tokens limit
                        generated_text = content_after_first_assistant.strip()
                else:
                    generated_text = "Error: Could not find '<|assistant|> ' tag in generated output."
            except Exception as e:
                generated_text = f"Error during parsing: {e}"


            print(f"\n--- Prompt {i+1} ---")
            print(f"Instruction: {prompt_text}")
            print(f"Full Decoded Output: {decoded_output}") # Print full output
            print(f"Generated Payload (extracted): {generated_text}")
    
    print("\n--- Validation Complete ---")


if __name__ == "__main__":
    main()
