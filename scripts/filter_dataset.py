
import json

input_file = "data/processed/red_train_v6_multi_clean.jsonl"
output_file = "data/processed/red_train_v6_multi_clean_filtered.jsonl"
blocked_keywords = ["or", "union"] # Keywords to filter out

filtered_count = 0
total_count = 0

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        total_count += 1
        entry = json.loads(line)
        payload = entry.get("payload", "").lower()
        
        should_filter = False
        for keyword in blocked_keywords:
            if keyword in payload:
                should_filter = True
                break
        
        if not should_filter:
            outfile.write(line)
        else:
            filtered_count += 1

print(f"Total entries: {total_count}")
print(f"Entries filtered (containing '{', '.join(blocked_keywords)}'): {filtered_count}")
print(f"Entries remaining: {total_count - filtered_count}")
print(f"Filtered dataset saved to: {output_file}")
