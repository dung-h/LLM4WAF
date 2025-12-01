
import json
from transformers import AutoTokenizer
import os
from tqdm import tqdm

INPUT_FILE = "data/processed/red_v40_phase2_reasoning.jsonl"
OUTPUT_FILE = "data/processed/red_v40_phase2_reasoning_ready.jsonl"
MODEL_ID = "google/gemma-2-2b-it"

def main():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    
    print(f"Processing {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        for line in tqdm(lines):
            try:
                record = json.loads(line.strip())
                messages = record.get("messages")
                
                if messages:
                    # Apply chat template to convert messages to a single string
                    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                    
                    # Write as a JSON object with a "text" field
                    new_record = {"text": formatted_text}
                    f_out.write(json.dumps(new_record) + "\n")
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
