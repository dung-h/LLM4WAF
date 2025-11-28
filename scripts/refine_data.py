
import json

def refine_and_combine_data():
    sqli_data_path = 'data/processed/red_train_v8_final.jsonl'
    xss_data_path = 'data/processed/red_train_xss_only.jsonl'
    output_path = 'data/processed/red_train_v9_mixed.jsonl'

    refined_sqli_data = []
    with open(sqli_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                payload = data.get('text', '')
                
                # Extract payload from the text
                if 'Payload: ' in payload:
                    payload_content = payload.split('Payload: ')[1]
                else:
                    # Handle cases where the format is different
                    # This is a simple heuristic, might need adjustment
                    parts = payload.split('<|assistant|>')
                    if len(parts) > 1:
                        payload_content = parts[1].strip()
                    else:
                        continue

                # Filter out simple payloads
                if 'UNION' in payload_content.upper() or ' OR ' in payload_content.upper():
                    continue

                refined_sqli_data.append({
                    "instruction": "Generate a single, effective MySQL SQL injection payload.",
                    "context": "Target: Generic MySQL database. Environment: Generic WAF.",
                    "constraints": "Output only the payload itself, with no other text or explanation.",
                    "payload": payload_content,
                    "reasoning": "Generated as part of a refined SQLi dataset.",
                    "attack_type": "SQLi"
                })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

    xss_data = []
    with open(xss_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                xss_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

    combined_data = refined_sqli_data + xss_data

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')

    print(f"Combined data saved to {output_path}")
    print(f"Total SQLi payloads: {len(refined_sqli_data)}")
    print(f"Total XSS payloads: {len(xss_data)}")
    print(f"Total combined payloads: {len(combined_data)}")

if __name__ == '__main__':
    refine_and_combine_data()
