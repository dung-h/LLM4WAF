import json
import os
import sys
import re
from collections import defaultdict
from datetime import datetime
import argparse
import pickle

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

class RedInternalIndex:
    def __init__(self):
        self.documents = []
        self.inverted_index = defaultdict(list) # word -> list of doc_ids
        self.metadata = {} # doc_id -> metadata

    def build(self, corpus_file):
        print(f"Building index from {corpus_file}...")
        self.documents = [] # Clear existing documents
        self.inverted_index = defaultdict(list)
        self.metadata = {}
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())
                    doc_id = doc['doc_id']
                    self.documents.append(doc)
                    
                    # Metadata
                    self.metadata[doc_id] = {
                        "kind": doc.get("kind"),
                        "attack_type": doc.get("attack_type"),
                        "technique": doc.get("technique"),
                        "waf_profile": doc.get("waf_profile"),
                        "text_preview": doc.get("text_for_embedding")[:100] if doc.get("text_for_embedding") else ""
                    }

                    # Tokenize and Index
                    text = doc.get("text_for_embedding", "").lower()
                    tokens = re.findall(r'\b\w+\b', text)
                    for token in set(tokens): # Unique tokens per doc
                        self.inverted_index[token].append(idx) # Store integer index for speed
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"Indexed {len(self.documents)} documents.")

    def save(self, index_filepath):
        os.makedirs(os.path.dirname(index_filepath), exist_ok=True)
        
        with open(index_filepath, 'wb') as f:
            pickle.dump({
                "inverted_index": self.inverted_index,
                "documents": self.documents, 
                "metadata": self.metadata
            }, f)
            
        print(f"Index saved to {index_filepath}")

    def load(self, index_filepath):
        pkl_path = index_filepath # Use index_filepath directly, it's already full path
        if not os.path.exists(pkl_path):
             print(f"Index file not found: {pkl_path}")
             return

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            self.inverted_index = data["inverted_index"]
            self.documents = data["documents"]
            self.metadata = data["metadata"]

    def search(self, query, top_k=5, filter_attack_type=None):
        tokens = re.findall(r'\b\w+\b', query.lower())
        scores = defaultdict(int)
        
        for token in tokens:
            if token in self.inverted_index:
                for doc_idx in self.inverted_index[token]:
                    # Optional filter
                    if filter_attack_type:
                        if self.documents[doc_idx].get("attack_type") != filter_attack_type:
                            continue
                    scores[doc_idx] += 1
        
        # Sort
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(self.documents[idx])
            
        return results

def build_index_script(corpus_file, index_dir, output_prefix):
    indexer = RedInternalIndex()
    indexer.build(corpus_file)
    
    index_filepath = os.path.join(index_dir, f"{output_prefix}_index.pkl")
    indexer.save(index_filepath)
    
    # Quick Test
    print("\n--- Quick Search Test ---")
    query = "XSS event handler obfuscation"
    results = indexer.search(query, top_k=3, filter_attack_type="XSS")
    for r in results:
        print(f"DocID: {r['doc_id']} | Kind: {r['kind']}")

    return index_filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/rag/red_corpus_internal_v2.jsonl")
    parser.add_argument("--index_dir", default="data/rag/index")
    parser.add_argument("--output_prefix", default="red_rag_v2")
    args = parser.parse_args()
    
    cmd_str = f"python scripts/red_rag_build_index.py --input {args.input} --output_prefix {args.output_prefix}"
    
    try:
        output = build_index_script(args.input, args.index_dir, args.output_prefix)
        log_message(cmd_str, "OK", output)
    except Exception as e:
        print(f"Error: {e}")
        log_message(cmd_str, "FAIL", str(e))