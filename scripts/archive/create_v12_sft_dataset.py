import sys
import logging
from datetime import datetime
import json
import os
import asyncio
import itertools
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
import openai

# --- Configuration ---
LOG_FILE = "v12_generation.log"
DATA_DIR = "data/processed"
V9_MIXED_DATASET_PATH = os.path.join(DATA_DIR, "red_train_v9_mixed.jsonl")
V12_SFT_DATASET_PATH = os.path.join(DATA_DIR, "v12_sft_data.jsonl")
XSS_LIMIT = 500

# A list of your DeepSeek API keys
DEEPSEEK_API_KEYS = [
    "sk-ffc6fa18776a475c8aba7d5457df2824", "sk-a6eae5b6fe93460380793444d1677478",
    "sk-92ba51e5f1cb4034b2cdcf0ec444f5dd", "sk-5c06b1843f2c40b9a7612f6f8cfd0afa",
    "sk-5624a8638a7442b4a98aa03382eb72ce", "sk-2098c65133c8421689da4bb227eb65e5",
    "sk-5d82535cd52f46948ee9d6b6363696ec", "sk-eb47b3807ffb45bf821de90c06e6d5e4",
    "sk-afd8f7921f60497fa28129fae291f3c7", "sk-afd7ed4ba2ce45639fd5ecf931f8a6b3"
]

# Create a cycle of asynchronous clients
clients = itertools.cycle([
    openai.AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    for key in DEEPSEEK_API_KEYS
])

# --- Utility Functions ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_jsonl(file_path):
    if not os.path.exists(file_path):
        return []
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_example_id(example):
    # Create a unique identifier for an example to check for its existence.
    return (
        example.get("instruction", ""),
        example.get("context", ""),
        example.get("attack_type", "")
    )

# --- Synchronous Functions for SQLi (Unchanged) ---
def refine_sqli_prompt(sqli_example):
    sync_client = openai.OpenAI(api_key=DEEPSEEK_API_KEYS[0], base_url="https://api.deepseek.com/v1")
    original_instruction = sqli_example.get("instruction", "")
    # ... (rest of the function is the same as before)
    llm_prompt = f"The following is an SQL Injection example..." # Truncated for brevity
    try:
        # ... (API call logic is the same)
        return "refined instruction" # Placeholder
    except Exception as e:
        logging.error(f"An unexpected error occurred during SQLi prompt refinement: {e}")
        return original_instruction

def generate_sqli_payload(refined_instruction):
    sync_client = openai.OpenAI(api_key=DEEPSEEK_API_KEYS[0], base_url="https://api.deepseek.com/v1")
    # ... (rest of the function is the same)
    try:
        # ... (API call logic is the same)
        return "generated payload" # Placeholder
    except Exception as e:
        logging.error(f"An unexpected error occurred during SQLi payload generation: {e}")
        return ""

# --- Asynchronous Functions for XSS (Unchanged) ---
async def refine_xss_prompt_async(client, xss_example):
    original_instruction = xss_example.get("instruction", "")
    # ... (rest of the function is the same)
    llm_prompt = f"The following is a Cross-Site Scripting (XSS) example..." # Truncated
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in cybersecurity and Cross-Site Scripting (XSS). Your task is to refine generic XSS instructions into highly specific and descriptive ones."},
                {"role": "user", "content": llm_prompt}
            ],
            temperature=0.7, max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"An unexpected error occurred during XSS prompt refinement: {e}")
        return original_instruction

async def generate_xss_payload_async(client, refined_instruction):
    # ... (rest of the function is the same)
    llm_prompt = f"Based on the following specific Cross-Site Scripting (XSS) instruction..." # Truncated
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in cybersecurity and Cross-Site Scripting (XSS). Your task is to generate a precise XSS payload based on a given instruction."},
                {"role": "user", "content": llm_prompt}
            ],
            temperature=0.7, max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"An unexpected error occurred during XSS payload generation: {e}")
        return ""

# --- Main Processing Logic ---
async def process_xss_example_worker(example, eos_token):
    client = next(clients)
    refined_instruction = await refine_xss_prompt_async(client, example)
    generated_payload = await generate_xss_payload_async(client, refined_instruction)

    new_example = example.copy()
    new_example["instruction"] = refined_instruction
    new_example["payload"] = generated_payload
    # ... (rest of the function is the same)
    prompt_str = f"<|user|>\n{new_example.get('instruction', '')}\n<|assistant|>\n{new_example.get('payload', '')}"
    new_example["prompt"] = prompt_str
    new_example["chosen"] = prompt_str + f"\n{eos_token}"
    return new_example

async def main():
    setup_logging()
    logging.info(f"--- Starting v12 SFT dataset creation at {datetime.now()} ---")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    eos_token = tokenizer.eos_token
    logging.info(f"Loaded tokenizer and eos_token.")

    # --- Load Data and Determine State ---
    logging.info(f"Loading source data from {V9_MIXED_DATASET_PATH}...")
    v9_data = load_jsonl(V9_MIXED_DATASET_PATH)
    logging.info(f"Loaded {len(v9_data)} total entries from source.")

    logging.info(f"Checking for existing data in {V12_SFT_DATASET_PATH} to resume...")
    existing_v12_data = load_jsonl(V12_SFT_DATASET_PATH)
    processed_ids = {get_example_id(ex) for ex in existing_v12_data}
    
    final_sqli_examples = [ex for ex in existing_v12_data if ex.get("attack_type") == "SQLi"]
    final_xss_examples = [ex for ex in existing_v12_data if ex.get("attack_type") == "XSS"]
    
    logging.info(f"Found {len(existing_v12_data)} already processed examples.")
    logging.info(f"({len(final_sqli_examples)} SQLi, {len(final_xss_examples)} XSS)")

    # --- SQLi Processing (if needed) ---
    all_sqli_examples = [entry for entry in v9_data if entry.get("attack_type") == "SQLi"]
    if not final_sqli_examples:
        logging.info(f"No existing SQLi data found. Processing {len(all_sqli_examples)} SQLi examples...")
        # This part remains synchronous as per the original logic
        for example in tqdm_asyncio(all_sqli_examples, desc="Processing SQLi"):
            refined_instruction = refine_sqli_prompt(example)
            generated_payload = generate_sqli_payload(refined_instruction)
            # ... (create new_example as before)
            final_sqli_examples.append(new_example)
        logging.info("SQLi processing complete.")
    else:
        logging.info("Skipping SQLi processing as data already exists.")

    # --- XSS Processing (if needed) ---
    num_xss_needed = XSS_LIMIT - len(final_xss_examples)
    if num_xss_needed > 0:
        all_xss_examples = [entry for entry in v9_data if entry.get("attack_type") == "XSS"]
        
        # Filter out examples that have already been processed
        # MODIFICATION: We are intentionally re-processing to generate more data variants.
        # The uniqueness check is removed.
        logging.info("Taking needed examples from the top of the source list to generate new variants.")
        xss_to_process_now = all_xss_examples[:num_xss_needed]
        
        logging.info(f"Need to process {num_xss_needed} more XSS examples. Found {len(xss_to_process_now)} available unique examples to process.")
        
        if xss_to_process_now:
            logging.info("Refining XSS prompts and generating new payloads concurrently...")
            tasks = [process_xss_example_worker(ex, eos_token) for ex in xss_to_process_now]
            newly_processed_xss = await tqdm_asyncio.gather(*tasks, desc="Processing XSS")
            final_xss_examples.extend(newly_processed_xss)
            logging.info("XSS processing complete.")
    else:
        logging.info(f"XSS limit of {XSS_LIMIT} already met or exceeded. Skipping XSS processing.")

    # --- Combine and Save ---
    final_data = final_sqli_examples + final_xss_examples
    logging.info(f"\nTotal examples to save: {len(final_data)}. Writing to {V12_SFT_DATASET_PATH}...")
    
    with open(V12_SFT_DATASET_PATH, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry) + '\n')
    
    logging.info("v12 SFT dataset creation/update complete!")
    logging.info(f"--- Finished at {datetime.now()} ---")

if __name__ == "__main__":
    asyncio.run(main())