import json
from pathlib import Path
import random
from transformers import AutoTokenizer

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

def create_v8_finetune_data():
    project_root = Path(__file__).resolve().parents[1]
    payload_sqli_dir = project_root / "data" / "raw" / "payloadbox_sqli"
    payload_xss_dir = project_root / "data" / "raw" / "payloadbox_xss"
    output_path = project_root / "data" / "processed" / "v8_sft_data.jsonl"

    output_path.parent.mkdir(exist_ok=True)

    # Load tokenizer to get eos_token
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    eos_token = tokenizer.eos_token

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
    for filepath in payload_sqli_dir.rglob("*.txt"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
            for line in infile:
                payload = line.strip()
                if payload and not payload.startswith('#') and len(payload) > 5:
                    category = categorize_sqli_payload(payload)
                    all_payloads[category].append(payload)

    # Read XSS payloads
    for filepath in payload_xss_dir.rglob("*.txt"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
            for line in infile:
                payload = line.strip()
                if payload and not payload.startswith('#') and len(payload) > 5:
                    category = categorize_xss_payload(payload)
                    all_payloads[category].append(payload)

    # Define instructions for each category
    instructions = {
        "union_based_sqli": [
            "Generate a union-based SQL injection payload for MySQL.",
            "Craft a SQLi payload using UNION SELECT for data exfiltration.",
            "Provide a MySQL UNION-based SQL injection string."
        ],
        "time_based_sqli": [
            "Create a time-based blind SQL injection payload for MySQL.",
            "Generate a SQLi payload that causes a time delay if true.",
            "Provide a MySQL time-based SQL injection string using SLEEP or BENCHMARK."
        ],
        "boolean_based_sqli": [
            "Generate a boolean-based blind SQL injection payload for MySQL.",
            "Craft a SQLi payload that returns true or false based on a condition.",
            "Provide a MySQL boolean-based SQL injection string using ASCII or SUBSTRING."
        ],
        "error_based_sqli": [
            "Generate an error-based SQL injection payload for MySQL.",
            "Craft a SQLi payload that reveals information through database errors.",
            "Provide a MySQL error-based SQL injection string using EXTRACTVALUE or UPDATEXML."
        ],
        "other_sqli": [
            "Generate a generic SQL injection payload for MySQL.",
            "Provide a basic SQL injection string.",
            "Craft a simple SQLi payload."
        ],
        "script_tag_xss": [
            "Generate an XSS payload using a script tag.",
            "Craft a cross-site scripting payload that executes JavaScript via <script>.",
            "Provide an XSS payload with an embedded script."
        ],
        "event_handler_xss": [
            "Generate an XSS payload using an event handler (e.g., onerror, onload).",
            "Craft a cross-site scripting payload that triggers JavaScript on an event.",
            "Provide an XSS payload that uses an HTML event attribute."
        ],
        "image_xss": [
            "Generate an XSS payload using an image tag.",
            "Craft a cross-site scripting payload that leverages an <img> or <svg> tag.",
            "Provide an XSS payload embedded in an image context."
        ],
        "other_xss": [
            "Generate a generic XSS payload.",
            "Craft a simple cross-site scripting payload.",
            "Provide a basic XSS string."
        ]
    }

    # Determine the minimum number of samples per category for balancing
    min_samples = min(len(payloads) for payloads in all_payloads.values() if payloads)
    if min_samples == 0:
        print("Warning: One or more categories have no payloads. Cannot create a balanced dataset.")
        return

    print(f"Balancing dataset to {min_samples} samples per category.")

    finetune_data = []
    for category, payloads in all_payloads.items():
        if payloads:
            # Sample min_samples from each category
            sampled_payloads = random.sample(payloads, min_samples)
            for payload in sampled_payloads:
                # Choose a random instruction for the category
                instruction = random.choice(instructions[category])
                
                # Format in chat format
                chat_format_entry = {
                    "prompt": f"""<|user|>
{instruction}
<|assistant|>
{payload}""",
                    "chosen": f"""<|user|>
{instruction}
<|assistant|>
{payload}\n{eos_token}""", # Add newline before eos_token
                    "rejected": "" # Not applicable for SFT
                }
                finetune_data.append(chat_format_entry)

    random.shuffle(finetune_data) # Shuffle the entire dataset

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry in finetune_data:
            outfile.write(json.dumps(entry) + '\n')
    
    print(f"Successfully created {output_path} with {len(finetune_data)} records.")

if __name__ == "__main__":
    create_v8_finetune_data()
