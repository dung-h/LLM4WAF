import json
import os
from tqdm import tqdm

def get_avg_length(filepath):
    user_lengths = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'messages' in item:
                    user_lengths.append(len(item['messages'][0]['content']))
            except: continue
            
    avg = sum(user_lengths) / len(user_lengths)
    return avg, max(user_lengths)

if __name__ == "__main__":
    avg, max_l = get_avg_length("data/processed/red_phase3_lightweight.jsonl")
    print(f"Lightweight Dataset:\n  Avg Length: {avg:.2f} chars\n  Max Length: {max_l} chars")

