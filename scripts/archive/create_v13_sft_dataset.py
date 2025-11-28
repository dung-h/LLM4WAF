import sys
import logging
import json
import os
import asyncio
import itertools
import random
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
import openai

# --- Configuration ---
LOG_FILE = "v13_generation.log"
DATA_DIR = "data/processed"
V12_SFT_DATASET_PATH = os.path.join(DATA_DIR, "v12_sft_data.jsonl")
V13_SFT_DATASET_PATH = os.path.join(DATA_DIR, "v13_sft_data.jsonl")
SQLI_ENRICHMENT_COUNT = 500  # Number of new SQLi variations to generate

DEEPSEEK_API_KEYS = [
    "sk-ffc6fa18776a475c8aba7d5457df2824", "sk-a6eae5b6fe93460380793444d1677478",
    "sk-92ba51e5f1cb4034b2cdcf0ec444f5dd", "sk-5c06b1843f2c40b9a7612f6f8cfd0afa",
    "sk-5624a8638a7442b4a98aa03382eb72ce", "sk-2098c65133c8421689da4bb227eb65e5",
    "sk-5d82535cd52f46948ee9d6b6363696ec", "sk-eb47b3807ffb45bf821de90c06e6d5e4",
    "sk-afd8f7921f60497fa28129fae291f3c7", "sk-afd7ed4ba2ce45639fd5ecf931f8a6b3"
]

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
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# --- Data Cleaning and Enrichment Functions ---

def is_valid_xss_payload(payload):
    """A simple heuristic to filter out conversational/bad XSS payloads."""
    if not payload or len(payload) > 500: return False
    bad_phrases = ["i'm ready to assist", "please provide", "scenario", "attack vector"]
    if any(phrase in payload.lower() for phrase in bad_phrases): return False
    if not any(c in payload for c in '<>"`()'): return False # Must have some payload-like characters
    return True

async def enrich_sqli_example(client, sqli_example):
    """Generates a new, more complex SQLi variation."""
    technique = random.choice(["time-based blind", "union-based", "error-based", "boolean-based"])
    
    llm_prompt = f"""Given the following SQL Injection example:
Instruction: "{sqli_example['instruction']}"
Payload: "{sqli_example['payload']}"

Generate a new, more advanced SQL injection instruction and a corresponding payload.
The new attack should be a '{technique}' attack.
The instruction must be specific, actionable, and describe the goal of the attack.
The payload must be different from the original.

Provide the response as a JSON object with two keys: "new_instruction" and "new_payload".
Do not include any other text, explanations, or markdown.

JSON Response:"""

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert specializing in SQL injection. Your task is to create new, advanced SQLi examples based on existing ones. You only output JSON."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=400
        )
        content = response.choices[0].message.content
        new_data = json.loads(content)
        
        # Basic validation of the returned JSON
        if "new_instruction" in new_data and "new_payload" in new_data:
            return {
                "instruction": new_data["new_instruction"],
                "payload": new_data["new_payload"],
                "attack_type": "SQLi",
                "source": "enriched"
            }
        return None
    except Exception as e:
        logging.error(f"An error occurred during SQLi enrichment: {e}")
        return None

async def main():
    setup_logging()
    logging.info(f"--- Starting v13 SFT dataset creation at {datetime.now()} ---")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    eos_token = tokenizer.eos_token
    logging.info("Loaded tokenizer.")

    # 1. Load v12 data
    logging.info(f"Loading data from {V12_SFT_DATASET_PATH}...")
    v12_data = load_jsonl(V12_SFT_DATASET_PATH)
    logging.info(f"Loaded {len(v12_data)} entries from v12.")

    # 2. Clean XSS data
    logging.info("Cleaning XSS data...")
    cleaned_xss = [ex for ex in v12_data if ex.get("attack_type") == "XSS" and is_valid_xss_payload(ex.get("payload", ""))]
    original_sqli = [ex for ex in v12_data if ex.get("attack_type") == "SQLi"]
    logging.info(f"Kept {len(cleaned_xss)} valid XSS samples and {len(original_sqli)} original SQLi samples.")

    # 3. Enrich SQLi data
    logging.info(f"Enriching SQLi data by generating {SQLI_ENRICHMENT_COUNT} new samples...")
    # Use a random sample of original SQLi data as seeds
    seed_examples = random.choices(original_sqli, k=SQLI_ENRICHMENT_COUNT)
    
    tasks = [enrich_sqli_example(next(clients), ex) for ex in seed_examples]
    enriched_sqli_results = await tqdm_asyncio.gather(*tasks, desc="Enriching SQLi")
    
    newly_enriched_sqli = [res for res in enriched_sqli_results if res is not None]
    logging.info(f"Successfully generated {len(newly_enriched_sqli)} new SQLi examples.")

    # 4. Combine and save v13 data
    final_data = original_sqli + cleaned_xss + newly_enriched_sqli
    
    # Add 'prompt' and 'chosen' fields to all entries
    for entry in final_data:
        instruction = entry.get('instruction', '')
        payload = entry.get('payload', '')
        prompt_str = f"<|user|>\n{instruction}\n<|assistant|>\n{payload}"
        entry["prompt"] = prompt_str
        entry["chosen"] = prompt_str + f"{eos_token}"

    logging.info(f"Total examples for v13 dataset: {len(final_data)}. Writing to {V13_SFT_DATASET_PATH}...")
    with open(V13_SFT_DATASET_PATH, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry) + '\n')

    logging.info("v13 SFT dataset created successfully!")
    logging.info(f"--- Finished at {datetime.now()} ---")

if __name__ == "__main__":
    asyncio.run(main())
