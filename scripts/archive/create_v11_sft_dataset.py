import sys
import logging
from datetime import datetime

import openai
import json
import os
from transformers import AutoTokenizer

# Replace with your actual DeepSeek API key
DEEPSEEK_API_KEY = "sk-ffc6fa18776a475c8aba7d5457df2824" 

# Initialize the OpenAI client with DeepSeek's API base URL and your API key
client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

DATA_DIR = "data/processed"
V9_MIXED_DATASET_PATH = os.path.join(DATA_DIR, "red_train_v9_mixed.jsonl")
V11_SFT_DATASET_PATH = os.path.join(DATA_DIR, "v11_sft_data.jsonl")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def refine_sqli_prompt(sqli_example):
    original_instruction = sqli_example.get("instruction", "")
    payload = sqli_example.get("payload", "")
    context = sqli_example.get("context", "")
    constraints = sqli_example.get("constraints", "")

    # Construct a prompt for the LLM to refine the instruction
    llm_prompt = f"""The following is an SQL Injection example:
Original Instruction: "{original_instruction}"
Context: "{context}"
Constraints: "{constraints}"
Payload: "{payload}""

    This original instruction is too generic. Please generate a new, more specific, and descriptive instruction for this SQL Injection payload. The new instruction should clearly describe the type of SQLi attack (e.g., error-based, time-based, union-based, boolean-based, stacked queries, out-of-band) and the specific technique used, based on the provided payload. Focus on making the instruction highly informative and actionable for a language model to generate similar payloads.

    New Instruction:"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in cybersecurity and SQL injection. Your task is to refine generic SQLi instructions into highly specific and descriptive ones."},
                {"role": "user", "content": llm_prompt}
            ],
            temperature=0.7, # A bit of creativity to make prompts diverse
            max_tokens=200
        )
        refined_instruction = response.choices[0].message.content.strip()
        return refined_instruction
    except openai.APIStatusError as e:
        logging.error(f"DeepSeek API Error during prompt refinement: {e.status_code} - {e.response}")
        return original_instruction # Fallback to original
    except openai.APIConnectionError as e:
        logging.error(f"DeepSeek API Connection Error during prompt refinement: {e}")
        return original_instruction # Fallback to original
    except Exception as e:
        logging.error(f"An unexpected error occurred during prompt refinement: {e}")
        return original_instruction # Fallback to original

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout) # Only log to console
                        ])
    
    logging.info("DEBUG: Running create_v11_sft_dataset.py - Version with 'chosen' field and logging fix.") # Added debug message
    logging.info(f"Starting v11 SFT dataset creation at {datetime.now()}")

    # Load tokenizer to get eos_token
    logging.info("DEBUG: Attempting to load tokenizer.") # NEW DEBUG
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    logging.info(f"DEBUG: Tokenizer loaded: {tokenizer}") # NEW DEBUG
    eos_token = tokenizer.eos_token
    logging.info(f"Loaded tokenizer and eos_token: {eos_token}")
    logging.info(f"DEBUG: Value of eos_token after assignment: '{eos_token}'") # NEW DEBUG LOG

    logging.info(f"Loading data from {V9_MIXED_DATASET_PATH}...")
    v9_data = load_jsonl(V9_MIXED_DATASET_PATH)
    logging.info(f"Loaded {len(v9_data)} entries.")

    # Separate SQLi and XSS examples
    all_sqli_examples = [entry for entry in v9_data if entry.get("attack_type") == "SQLi"]
    all_xss_examples = [entry for entry in v9_data if entry.get("attack_type") == "XSS"]

    # Limit to 100 examples of each type for testing
    sqli_examples = all_sqli_examples[:100]
    xss_examples = all_xss_examples[:100]

    logging.info(f"Processing {len(sqli_examples)} SQLi examples (out of {len(all_sqli_examples)} total).")
    logging.info(f"Processing {len(xss_examples)} XSS examples (out of {len(all_xss_examples)} total).")

    logging.info("DEBUG: Before refined_sqli_examples initialization.") # NEW DEBUG LOG
    refined_sqli_examples = []
    logging.info("DEBUG: Before 'Refining SQLi prompts...' log.") # NEW DEBUG LOG
    logging.info("\nRefining SQLi prompts...") # This is the line that should print next
    logging.info("DEBUG: Before SQLi examples loop.") # NEW DEBUG LOG
    for i, example in enumerate(sqli_examples):
        logging.info(f"DEBUG: Entering SQLi example loop for example {i+1}/{len(sqli_examples)}") # NEW DEBUG LOG
        logging.info(f"Processing SQLi example {i+1}/{len(sqli_examples)}") # Original log
        logging.info(f"DEBUG: Before calling refine_sqli_prompt for example {i+1}.") # NEW DEBUG LOG
        refined_instruction = refine_sqli_prompt(example)
        logging.info(f"DEBUG: After calling refine_sqli_prompt for example {i+1}.") # NEW DEBUG LOG
        new_example = example.copy()
        new_example["instruction"] = refined_instruction
        instruction_content = new_example.get('instruction', '')
        payload_content = new_example.get('payload', '')

        # Build the prompt string using triple quotes for multi-line strings
        prompt_str = f"""<|user|>
{instruction_content}
<|assistant|>
{payload_content}"""

        new_example["prompt"] = prompt_str
        new_example["chosen"] = prompt_str + f"\n{eos_token}"
        refined_sqli_examples.append(new_example)
        logging.info(f"DEBUG: After processing SQLi example {i+1}.") # NEW DEBUG LOG
    logging.info("DEBUG: After SQLi examples loop.") # NEW DEBUG LOG
    logging.info("\nSQLi prompt refinement complete.") # Original log
    logging.info("DEBUG: After 'SQLi prompt refinement complete' log.") # NEW DEBUG LOG

    # For now, we'll just use the existing XSS examples.
    # Add the 'prompt' and 'chosen' fields for SFTTrainer to XSS examples as well
    logging.info("DEBUG: Before final_xss_examples_with_chosen initialization.") # NEW DEBUG LOG
    final_xss_examples_with_chosen = []
    logging.info("DEBUG: Before XSS examples loop.") # NEW DEBUG LOG
    for example in xss_examples:
        logging.info("DEBUG: Entering XSS example loop.") # NEW DEBUG LOG
        new_example = example.copy()
        instruction_content = new_example.get('instruction', '')
        payload_content = new_example.get('payload', '')

        # Build the prompt string using triple quotes for multi-line strings
        prompt_str = f"""<|user|>
{instruction_content}
<|assistant|>
{payload_content}"""

        new_example["prompt"] = prompt_str
        new_example["chosen"] = prompt_str + f"\n{eos_token}"
        final_xss_examples_with_chosen.append(new_example)
        logging.info("DEBUG: After processing XSS example.") # NEW DEBUG LOG
    logging.info("DEBUG: After XSS examples loop.") # NEW DEBUG LOG
    final_xss_examples = final_xss_examples_with_chosen
    logging.info("DEBUG: After final_xss_examples assignment.") # NEW DEBUG LOG

    # Combine everything into a single v11_sft_data.jsonl file.
    logging.info("DEBUG: Before combining data.") # NEW DEBUG LOG
    v11_data = refined_sqli_examples + final_xss_examples
    logging.info("DEBUG: Before saving data.") # NEW DEBUG LOG

    logging.info(f"\nSaving {len(v11_data)} entries to {V11_SFT_DATASET_PATH}...")
    with open(V11_SFT_DATASET_PATH, 'w', encoding='utf-8') as f:
        for entry in v11_data:
            f.write(json.dumps(entry) + '\n')
    logging.info("DEBUG: After saving data file.") # NEW DEBUG LOG
    logging.info("v11 SFT dataset created successfully!")
    logging.info(f"Finished v11 SFT dataset creation at {datetime.now()}")

if __name__ == "__main__":
    main()