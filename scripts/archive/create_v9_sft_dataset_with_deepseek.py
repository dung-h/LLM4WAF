import json
from pathlib import Path
import random
import requests
import os
import time

# --- Configuration ---
DEEPSEEK_API_KEY = "sk-ffc6fa18776a475c8aba7d5457df2824" # Use one of the user's keys
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_SQLI_DIR = PROJECT_ROOT / "data" / "raw" / "payloadbox_sqli"
PAYLOAD_XSS_DIR = PROJECT_ROOT / "data" / "raw" / "payloadbox_xss"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "v9_sft_data_deepseek.jsonl"
MAX_PAYLOADS_PER_CATEGORY = 20 # Limit the number of payloads to process to avoid long run times

# --- Payload Categorization ---
def categorize_sqli_payload(payload):
    payload = payload.lower()
    if "union select" in payload:
        return "union_based_sqli"
    if "if(" in payload and ("sleep(" in payload or "benchmark(" in payload):
        return "time_based_sqli"
    if "if(" in payload and ("ascii(" in payload or "substring(" in payload):
        return "boolean_based_sqli"
    if "extractvalue(" in payload or "updatexml(" in payload:
        return "error_based_sqli"
    return "other_sqli"

def categorize_xss_payload(payload):
    payload = payload.lower()
    if "<script>" in payload or "javascript:" in payload:
        return "script_tag_xss"
    if "onerror=" in payload or "onload=" in payload or "alert(" in payload:
        return "event_handler_xss"
    if "<img" in payload or "<svg" in payload:
        return "image_xss"
    return "other_xss"

# --- DeepSeek API Interaction ---
def get_refined_payload_from_deepseek(payload, category):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    
    system_prompt = """You are an expert in web application security. Your task is to analyze a given payload, identify its attack type, and generate a clear, concise instruction that would produce this payload. You will also refine the payload to be more effective and concise if possible.

    You must output a JSON object with the following structure:
    {
      "instruction": "A clear and concise instruction for generating the payload.",
      "refined_payload": "The refined, more effective payload."
    }
    """
    
    user_prompt = f"Analyze and refine the following payload for the category '{category}':\n\nPayload: {payload}"

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.5,
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {e}")
        return None

# --- Main Script ---
def create_v9_finetune_data():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    all_payloads = {
        "union_based_sqli": [],
        "time_based_sqli": [],
        "boolean_based_sqli": [],
        "error_based_sqli": [],
        "other_sqli": [],
        "script_tag_xss": [],
        "event_handler_xss": [],
        "image_xss": [],
        "other_xss": []
    }

    # Read SQLi payloads
    for filepath in PAYLOAD_SQLI_DIR.rglob("*.txt"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
            for line in infile:
                payload = line.strip()
                if payload and not payload.startswith('#') and len(payload) > 5:
                    category = categorize_sqli_payload(payload)
                    if len(all_payloads[category]) < MAX_PAYLOADS_PER_CATEGORY:
                        all_payloads[category].append(payload)

    # Read XSS payloads
    for filepath in PAYLOAD_XSS_DIR.rglob("*.txt"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
            for line in infile:
                payload = line.strip()
                if payload and not payload.startswith('#') and len(payload) > 5:
                    category = categorize_xss_payload(payload)
                    if len(all_payloads[category]) < MAX_PAYLOADS_PER_CATEGORY:
                        all_payloads[category].append(payload)

    finetune_data = []
    total_processed = 0
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
        for category, payloads in all_payloads.items():
            if not payloads:
                continue
            
            print(f"Processing {len(payloads)} payloads for category: {category}")
            for payload in payloads:
                print(f"  - Refining payload: {payload[:80]}...")
                
                refined_data = get_refined_payload_from_deepseek(payload, category)
                
                if refined_data and 'choices' in refined_data and refined_data['choices']:
                    try:
                        content = json.loads(refined_data['choices'][0]['message']['content'])
                        instruction = content.get("instruction")
                        refined_payload = content.get("refined_payload")

                        if instruction and refined_payload:
                            chat_format_entry = {
                                "prompt": f"""<|user|>
{instruction}
<|assistant|>""",
                                "chosen": f"""{refined_payload}
<|endoftext|>""",
                                "rejected": ""
                            }
                            outfile.write(json.dumps(chat_format_entry) + '\n')
                            total_processed += 1
                        else:
                            print("    - Skipping: Missing instruction or refined_payload in DeepSeek response.")
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"    - Skipping: Error parsing DeepSeek response: {e}")
                else:
                    print("    - Skipping: Invalid or empty response from DeepSeek API.")
                
                time.sleep(1) # Rate limit to avoid overwhelming the API

    print(f"\nSuccessfully created {OUTPUT_PATH} with {total_processed} records.")

if __name__ == "__main__":
    create_v9_finetune_data()