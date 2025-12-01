
import json
from transformers import AutoTokenizer
import os
from tqdm import tqdm

INPUT_FILE = "data/processed/red_v40_phase2_reasoning.jsonl"
OUTPUT_FILE = "data/processed/red_v40_phase2_phi3_1500.jsonl"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SAMPLE_LIMIT = 1500

def main():
    print(f"Loading tokenizer for {MODEL_ID}...")
    # Use HF_TOKEN if available
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, trust_remote_code=True)
    
    print(f"Processing {INPUT_FILE} (Limit: {SAMPLE_LIMIT} samples)...")
    count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in):
            if count >= SAMPLE_LIMIT:
                break
                
            try:
                record = json.loads(line.strip())
                messages = record.get("messages")
                
                if messages:
                    # Apply chat template to convert messages to a single string
                    # Phi 3 template usually handles <|user|>, <|assistant|>, <|end|>
                    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                    
                    # Write as a JSON object with a "text" field
                    new_record = {"text": formatted_text}
                    f_out.write(json.dumps(new_record) + "\n")
                    count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    print(f"Done. Saved {count} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
