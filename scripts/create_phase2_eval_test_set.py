
import json
import random
import os
from datetime import datetime

# --- Configuration ---
INPUT_FILE = "data/processed/red_v40_phase2_reasoning.jsonl"
OUTPUT_FILE = "data/processed/red_v40_phase2_eval_test.jsonl"
TEST_SIZE = 100

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def main():
    cmd_str = f"python scripts/create_phase2_eval_test_set.py"
    try:
        print(f"Loading samples from {INPUT_FILE} for eval test set creation...")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_samples = [json.loads(line.strip()) for line in f]
        
        print(f"  - Total Phase 2 samples loaded: {len(all_samples)}")

        if len(all_samples) < TEST_SIZE:
            print(f"Warning: Only {len(all_samples)} samples available. Using all for test.")
            test_set = all_samples
        else:
            test_set = random.sample(all_samples, TEST_SIZE)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for record in test_set:
                f.write(json.dumps(record) + '\n')
                
        print(f"  - Eval test set created with {len(test_set)} records at {OUTPUT_FILE}")
        log_message(cmd_str, "OK", OUTPUT_FILE)

    except Exception as e:
        print(f"Error creating Phase 2 eval test set: {e}")
        log_message(cmd_str, "FAIL")

if __name__ == "__main__":
    main()
