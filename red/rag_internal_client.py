import sys
import os
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.red_rag_build_index import RedInternalIndex # Reuse class definition

class RedRAGClient:
    def __init__(self, index_dir="data/rag/index", corpus_version="v2"):
        self.indexer = RedInternalIndex()
        # Adjusted index_file path logic
        if corpus_version == "v1":
            index_file = os.path.join(index_dir, "red_rag_index.pkl") # Old default
        else: # Default to v2
            index_file = os.path.join(index_dir, f"red_rag_{corpus_version}_index.pkl")
            
        self.indexer.load(index_file)
        print(f"RedRAGClient loaded index from {index_file} (version: {corpus_version})")

    def retrieve_red_knowledge(self, attack_type: str, signals: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves knowledge relevant to the current attack context.
        """
        # Build a simple query string from signals
        query_parts = [attack_type]
        
        if signals:
            if "blocked_keywords" in signals and signals["blocked_keywords"]:
                if isinstance(signals["blocked_keywords"], list):
                    query_parts.extend(signals["blocked_keywords"])
                else: # Assume string
                    query_parts.append(signals["blocked_keywords"])
            
            if "waf_profile_hint" in signals and signals["waf_profile_hint"]:
                query_parts.append(signals["waf_profile_hint"])
                
            # Add other signal processing logic here if needed (e.g., status codes, reflection)
            if "status_codes" in signals and signals["status_codes"]:
                query_parts.extend([str(s) for s in signals["status_codes"]])
            if "reflection" in signals and signals["reflection"]:
                query_parts.append("reflected")

        query_text = " ".join(query_parts)
        # print(f"DEBUG: RAG Query: {query_text}")
        
        results = self.indexer.search(query_text, top_k=top_k, filter_attack_type=attack_type)
        return results

# Singleton instance pattern for default client (v2 corpus)
_client_instance_v2 = None
_client_instance_v1 = None

def get_client(corpus_version: str = "v2"):
    global _client_instance_v1, _client_instance_v2
    if corpus_version == "v2":
        if _client_instance_v2 is None:
            _client_instance_v2 = RedRAGClient(corpus_version="v2")
        return _client_instance_v2
    elif corpus_version == "v1":
        if _client_instance_v1 is None:
            _client_instance_v1 = RedRAGClient(corpus_version="v1")
        return _client_instance_v1
    else:
        raise ValueError(f"Unknown corpus version: {corpus_version}")

if __name__ == "__main__":
    # Quick Test V2
    client_v2 = get_client(corpus_version="v2")
    
    print("\n--- Test Retrieval (V2 Corpus): XSS with blocked <script> ---")
    signals_xss = {"blocked_keywords": ["<script>", "onerror"], "waf_profile_hint": "modsec"}
    docs_xss = client_v2.retrieve_red_knowledge("XSS", signals_xss, top_k=3)
    
    for doc in docs_xss:
        print(f"Found: {doc['doc_id']} ({doc['kind']})")
        if 'example_payloads' in doc and doc['example_payloads']:
            print(f"  Example payload: {doc['example_payloads'][0][:50]}...")
        elif doc['kind'] == 'attack_strategy':
            print(f"  Idea: {doc['idea']}")

    print("\n--- Test Retrieval (V2 Corpus): SQLI time-based blind ---")
    signals_sqli = {"blocked_keywords": ["OR", "UNION"], "waf_profile_hint": "modsec", "reflection": False}
    docs_sqli = client_v2.retrieve_red_knowledge("SQLI", signals_sqli, top_k=3)
    
    for doc in docs_sqli:
        print(f"Found: {doc['doc_id']} ({doc['kind']})")
        if 'example_payloads' in doc and doc['example_payloads']:
            print(f"  Example payload: {doc['example_payloads'][0][:50]}...")
        elif doc['kind'] == 'attack_strategy':
            print(f"  Idea: {doc['idea']}")

    # Quick Test V1 (for comparison)
    client_v1 = get_client(corpus_version="v1") # This will load from red_rag_index.pkl
    
    print("\n--- Test Retrieval (V1 Corpus): XSS with blocked <script> ---")
    signals_xss_v1 = {"blocked_keywords": ["<script>", "onerror"], "waf_profile_hint": "modsec"}
    docs_xss_v1 = client_v1.retrieve_red_knowledge("XSS", signals_xss_v1, top_k=3)
    
    for doc in docs_xss_v1:
        print(f"Found: {doc['doc_id']} ({doc['kind']})")
        if 'example_payloads' in doc and doc['example_payloads']:
            print(f"  Example payload: {doc['example_payloads'][0][:50]}...")
        elif doc['kind'] == 'attack_strategy':
            print(f"  Idea: {doc['idea']}")