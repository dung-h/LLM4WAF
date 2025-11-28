import json

def create_sft_dataset(input_path="passed_payloads.jsonl", output_path="data/processed/sft_train.jsonl"):
    """
    Creates a new SFT training dataset from the passed payloads.
    The output format will be a JSONL file with each line containing a "text" field
    in the format: "Payload: <payload>".
    """
    print(f"[+] Creating SFT dataset from '{input_path}'...")
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            count = 0
            for line in infile:
                try:
                    data = json.loads(line)
                    payload = data.get("payload")
                    if payload:
                        sft_entry = {"text": f"Payload: {payload}"}
                        outfile.write(json.dumps(sft_entry) + "\n")
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  - Created SFT dataset with {count} entries at '{output_path}'.")
    except FileNotFoundError:
        print(f"[!] Error: Input file not found at '{input_path}'. Aborting.")

if __name__ == "__main__":
    create_sft_dataset()

