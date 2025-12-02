import json
import os
import re
from typing import List, Dict, Any
from collections import defaultdict # Import defaultdict

class CRSKnowledgeBase:
    def __init__(self, kb_path: str = "data/blue/blue_phase1_crs_kb.jsonl"):
        self.kb_path = kb_path
        self.entries = []
        self.index = {} # Simple keyword-based index
        self._load_crs_entries()
        self._build_index()

    def _load_crs_entries(self):
        """Loads CRS entries from the JSONL file."""
        if not os.path.exists(self.kb_path):
            print(f"Warning: CRS KB file not found at {self.kb_path}. KB will be empty.")
            return

        with open(self.kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.entries.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error loading CRS KB entry: {e} in line: {line.strip()}")
    
    def _build_index(self):
        """Builds a simple keyword-based index for retrieval."""
        for i, entry in enumerate(self.entries):
            searchable_text = []
            if entry.get("rule_id"):
                searchable_text.append(str(entry["rule_id"]))
            if entry.get("test_description"):
                searchable_text.append(entry["test_description"])
            if entry.get("operator"):
                searchable_text.append(entry["operator"])
            if entry.get("example_payload"):
                searchable_text.append(entry["example_payload"])
            
            # Combine all searchable text
            full_text = " ".join(searchable_text).lower()
            
            # Simple tokenization and indexing
            for word in re.findall(r'\b\w+\b', full_text):
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(i) # Store index of the entry

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieves top_k relevant CRS entries based on a keyword search.
        """
        query_words = re.findall(r'\b\w+\b', query_text.lower())
        
        # Score entries based on number of matching keywords
        scores = defaultdict(int)
        for word in query_words:
            if word in self.index:
                for entry_idx in self.index[word]:
                    scores[entry_idx] += 1
        
        # Sort by score and return top_k entries
        sorted_entries_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for idx in sorted_entries_indices:
            results.append(self.entries[idx])
            if len(results) >= top_k:
                break
        
        return results

if __name__ == "__main__":
    # Simple test
    kb = CRSKnowledgeBase()
    print(f"Loaded {len(kb.entries)} CRS KB entries.")
    
    test_query = "ruby error message"
    results = kb.query(test_query, top_k=2)
    print(f"\nQuery: '{test_query}'")
    for r in results:
        print(f"- Rule ID: {r.get('rule_id')}, Description: {r.get('test_description')}")

    test_query = "union select SQLI"
    results = kb.query(test_query, top_k=2)
    print(f"\nQuery: '{test_query}'")
    for r in results:
        print(f"- Rule ID: {r.get('rule_id')}, Description: {r.get('test_description')}")