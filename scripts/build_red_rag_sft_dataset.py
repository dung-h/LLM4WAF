import json
import os
import sys
import random
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from red.rag_internal_client import get_client

# Config
INPUT_DATASET = "data/processed/red_phase1_enriched_v2.jsonl"
OUTPUT_DATASET = "data/processed/red_phase2_rag_sft_train.jsonl"
CORPUS_VERSION = "v2"
SAMPLES_TO_GENERATE = 4000 

def build_rag_aware_prompt(attack_type, technique, rag_docs):
    waf_type = "ModSecurity + OWASP CRS 3.3 (PL1)"
    injection_point = "query parameter"
    
    rag_context_str = ""
    if rag_docs:
        rag_context_str += "\nRelevant Knowledge (RAG):\n"
        for i, doc in enumerate(rag_docs):
            t_info = doc.get('technique') or doc.get('family_name') or 'N/A'
            if isinstance(t_info, tuple): t_info = str(t_info)
            
            rag_context_str += f"- [{doc['kind']}] {doc['doc_id']}: {doc.get('short_desc', '')} (Technique: {t_info})\n"
            if doc.get('example_payloads'):
                rag_context_str += f"  Example: {doc['example_payloads'][0]}\n"

    prompt = f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: {waf_type}
- Attack type: {attack_type}
- Injection point: {injection_point}
{rag_context_str}
Previously tried payloads: None (Fresh attempt)

Your task:
Generate a NEW {attack_type} payload that successfully bypasses the WAF.
Target Technique: {technique}

Instructions:
1. Analyze the provided RAG Knowledge patterns.
2. Adapt the technique to fit the injection point.
3. Apply obfuscation if necessary.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations.
"""
    return prompt

def main():
    print(f"Loading dataset from {INPUT_DATASET}...")
    if not os.path.exists(INPUT_DATASET):
        print("Input dataset not found!")
        return

    data = []
    with open(INPUT_DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except: continue
    
    random.shuffle(data)
    selected_data = data[:SAMPLES_TO_GENERATE]
    
    print(f"Initializing RAG Client ({CORPUS_VERSION})...")
    rag_client = get_client(corpus_version=CORPUS_VERSION)
    
    output_data = []
    print(f"Building RAG-SFT samples for {len(selected_data)} entries...")

    for sample in tqdm(selected_data):
        attack_type = sample.get("attack_type", "UNKNOWN")
        technique = sample.get("technique", "Unknown")
        
        payload = None
        if "messages" in sample:
            for m in sample["messages"]:
                if m["role"] == "assistant":
                    payload = m["content"]
                    break
        if not payload:
            continue

        # Use client to search (hybrid search implicitly)
        # Here we manually craft a query to ensure we find relevant docs for training
        query_text = technique
        retrieved_docs = rag_client.indexer.search(query_text, top_k=3)
        
        doc_dicts = retrieved_docs

        user_prompt = build_rag_aware_prompt(attack_type, technique, doc_dicts)

        training_entry = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": payload}
            ],
            "meta": {
                "attack_type": attack_type,
                "technique": technique,
                "rag_docs_ids": [d['doc_id'] for d in doc_dicts]
            }
        }
        output_data.append(training_entry)

    print(f"Saving {len(output_data)} samples to {OUTPUT_DATASET}...")
    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + "\n")
    
    if output_data:
        print("\n--- Example RAG-SFT Sample ---")
        print(output_data[0]["messages"][0]["content"])
        print("-" * 20)
        print(f"Target Output: {output_data[0]['messages'][1]['content']}")

if __name__ == "__main__":
    main()