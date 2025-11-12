
import json

input_file = "data/processed/red_train_v6_multi_clean.jsonl"
output_file = "data/processed/red_train_v6_multi_clean_small.jsonl"
num_samples = 10

count = 0
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if count < num_samples:
            outfile.write(line)
            count += 1
        else:
            break

print(f"Extracted {count} samples to: {output_file}")
