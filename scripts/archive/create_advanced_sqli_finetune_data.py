import json
from pathlib import Path

def create_finetune_data():
    """
    Reads SQLi payloads from the payloadbox vendor directory and creates
    a clean, high-quality JSONL dataset for fine-tuning.
    """
    # Define paths
    project_root = Path(__file__).resolve().parents[1]
    payload_sqli_dir = project_root / "data" / "raw" / "payloadbox_sqli"
    payload_xss_dir = project_root / "data" / "raw" / "payloadbox_xss"
    output_path = project_root / "data" / "processed" / "advanced_sqli_finetune_data.jsonl"

    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True)

    instruction = "Generate a single, effective MySQL SQL injection payload."
    count = 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        print(f"Searching for payloads in: {payload_sqli_dir} and {payload_xss_dir}")
        
        # Find all .txt files containing payloads in both SQLi and XSS directories
        payload_files = list(payload_sqli_dir.rglob("*.txt")) + list(payload_xss_dir.rglob("*.txt"))
        if not payload_files:
            print(f"Warning: No payload files found in {payload_dir}. Dataset will be empty.")
            return

        print(f"Found {len(payload_files)} payload files. Processing...")

        for filepath in payload_files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                for line in infile:
                    payload = line.strip()
                    # Filter out comments, empty lines, and very short/trivial payloads
                    if payload and not payload.startswith('#') and len(payload) > 5:
                        # Create the JSONL entry
                        record = {
                            "instruction": instruction,
                            "context": "", # No context needed for this simple format
                            "payload": payload
                        }
                        outfile.write(json.dumps(record) + '\n')
                        count += 1
    
    print(f"Successfully created {output_path} with {count} records.")

if __name__ == "__main__":
    create_finetune_data()