import os
import json
import random
from typing import List, Dict, Any
import httpx
from tqdm import tqdm
import logging

# --- Setup Logging ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("create_v10_sft_dataset.log", mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def call_deepseek_api(instruction: str, client: httpx.Client) -> str:
    """Calls the DeepSeek API to generate a payload based on the instruction."""
    system_prompt = (
        "You are an expert in web application security. Your task is to generate a single, "
        "concise, and effective payload for the given web attack scenario. "
        "You must only output the raw payload itself, with no explanations, "
        "no markdown, no code blocks, and no additional text whatsoever. "
        "Focus on creating a high-quality, realistic payload that a security professional would use."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    try:
        response = client.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "deepseek-coder",
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.4,
                "stream": False,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        payload = data['choices'][0]['message']['content'].strip()
        
        # Basic cleanup to remove potential markdown
        if payload.startswith("```") and payload.endswith("```"):
            payload = payload[3:-3].strip()
            # Remove language specifier if present
            if '\n' in payload:
                payload = payload.split('\n', 1)[1]

        return payload

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling DeepSeek API: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred calling DeepSeek API: {e}")
    
    return ""


def create_sqli_instructions() -> List[str]:
    """Creates a diverse set of instructions for generating SQLi payloads."""
    instructions = [
        # --- Error-Based ---
        "Generate an error-based SQL injection payload for MySQL that uses the `EXTRACTVALUE` function to disclose the database version.",
        "Craft an error-based SQL injection for PostgreSQL that leverages the `query_to_xml` function to reveal user tables.",
        "Create an error-based SQLi payload for MSSQL using `CONVERT(int, ...)` to force an error and show the system user (`SUSER_SNAME()`).",
        "Construct an error-based SQL injection for Oracle that uses `CTXSYS.DRITHSX.SN` to display the current user.",
        "Generate a double-query error-based SQL injection payload for MySQL to extract the output of `@@version`.",

        # --- Time-Based Blind ---
        "Create a time-based blind SQL injection payload for MySQL that causes a 7-second delay if the first character of the database name is 's'.",
        "Generate a time-based blind SQLi for PostgreSQL using `pg_sleep()` that delays for 10 seconds.",
        "Craft a time-based blind SQL injection for MSSQL using `WAITFOR DELAY` that pauses for 5 seconds.",
        "Construct a time-based blind SQLi for Oracle using `DBMS_PIPE.RECEIVE_MESSAGE` to introduce a 8-second delay.",
        "Generate a time-based blind payload for SQLite that uses `LIKE` and a large number of `AND` conditions to cause a noticeable delay.",

        # --- Boolean-Based Blind ---
        "Create a boolean-based blind SQL injection payload that returns true if the length of the database name is greater than 5.",
        "Generate a boolean-based blind SQLi to check if the first character of the first table name is 'u'.",
        "Craft a boolean-based blind payload to confirm if the current user's name starts with 'a'.",
        "Construct a boolean-based blind SQL injection that verifies if the database version contains the number '10'.",
        "Generate a boolean-based blind payload that checks if the `users` table exists in the database.",

        # --- UNION-Based ---
        "Provide a UNION-based SQL injection payload for a 4-column query to retrieve all database names from `information_schema.schemata`.",
        "Craft a UNION-based SQLi to get the `table_name` and `column_name` from `information_schema.columns` where the table is 'users'. Assume 3 columns.",
        "Generate a UNION-based payload to extract usernames and passwords from a `users` table, assuming 2 columns in the original query.",
        "Construct a UNION-based SQL injection to read the content of `/etc/passwd` using `load_file()` in a 3-column query.",
        "Create a UNION-based SQLi payload that combines `NULL` values with the database version and current user for a 5-column query.",
        
        # --- Stacked Queries ---
        "Generate a stacked query SQL injection payload to insert a new admin user with username 'pwned' and password 'pwned' into a `users` table.",
        "Craft a stacked query SQLi to update the password of the 'admin' user to 'hacked'.",
        "Create a stacked query payload to create a new table named `hacked_data` with a single text column.",
        "Construct a stacked query SQL injection to drop a table named `temp_users`.",
        "Generate a stacked query SQLi that executes a `SHUTDOWN` command on the database server (for MSSQL).",
    ]
    return instructions

def main():
    output_file = "data/processed/v10_sft_data.jsonl"
    
    logger.info("Starting dataset generation for v10.")
    
    sqli_instructions = create_sqli_instructions()
    all_instructions = sqli_instructions # For now, just SQLi
    
    logger.info(f"Generated {len(all_instructions)} unique instructions.")

    with httpx.Client() as client, open(output_file, 'w', encoding='utf-8') as f:
        for instruction in tqdm(all_instructions, desc="Generating Payloads"):
            payload = call_deepseek_api(instruction, client)
            
            if payload:
                # Format for SFT training
                prompt = f"""<|user|>
{instruction}
<|assistant|>"""
                chosen = f"{payload}\n<|endoftext|>"
                
                record = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": "" # No rejections in this SFT dataset
                }
                f.write(json.dumps(record) + '\n')
                logger.info(f"Successfully generated and wrote payload for instruction: {instruction[:60]}...")
            else:
                logger.warning(f"Failed to generate payload for instruction: {instruction}")

    logger.info(f"Dataset generation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()