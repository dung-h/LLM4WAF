
import json
from typing import List, Dict, Any
from blue.rag_index import CRSKnowledgeBase

def retrieve_for_episode(episode: Dict, kb: CRSKnowledgeBase, top_k: int = 5) -> List[Dict]:
    """
    Constructs a query from a blue_episode and retrieves top_k relevant CRS entries from the KB.
    """
    query_parts = []

    # Prioritize error messages and vulnerability effects
    if episode["app_observation"]["resp_body_snippet"]:
        query_parts.append(episode["app_observation"]["resp_body_snippet"])
    if episode["blue_label"]["vuln_effect"] and episode["blue_label"]["vuln_effect"] != "no-effect":
        query_parts.append(episode["blue_label"]["vuln_effect"].replace("-", " "))
    
    # Add attack type
    if episode["attack"]["attack_type"]:
        query_parts.append(episode["attack"]["attack_type"])
    
    # Add technique
    if episode["attack"]["technique"] and episode["attack"]["technique"] != "UNKNOWN":
        query_parts.append(episode["attack"]["technique"].replace("_", " "))

    # Add relevant WAF context (e.g., engine, paranoia level)
    if episode["waf_env"]["engine"] != "UNKNOWN":
        query_parts.append(episode["waf_env"]["engine"])
    if episode["waf_env"]["paranoia_level"]:
        query_parts.append(f"PL{episode['waf_env']['paranoia_level']}")

    query_text = " ".join(query_parts)
    if not query_text:
        return [] # Cannot form a query

    return kb.query(query_text, top_k=top_k)

if __name__ == "__main__":
    # Simple test with a dummy episode
    dummy_kb = CRSKnowledgeBase(kb_path="data/blue/blue_phase1_crs_kb.jsonl")
    
    # Example 1: SQL error
    example_episode_sql = {
        "app_observation": {"resp_body_snippet": "SQL syntax error near 'union select'"},
        "blue_label": {"vuln_effect": "sql-error-disclosure"},
        "attack": {"attack_type": "SQLI", "technique": "union_select"},
        "waf_env": {"engine": "modsecurity", "paranoia_level": 1}
    }
    retrieved_sql = retrieve_for_episode(example_episode_sql, dummy_kb, top_k=2)
    print("\nRetrieved for SQL Episode:")
    for r in retrieved_sql:
        print(f"- {r.get('rule_id')}: {r.get('test_description')}")

    # Example 2: Ruby error (from KB example)
    example_episode_ruby = {
        "app_observation": {"resp_body_snippet": "SyntaxError (/path/to/your/app/controllers/reflects_controller.rb:10: ...)"},
        "blue_label": {"vuln_effect": "error-disclosure"},
        "attack": {"attack_type": "UNKNOWN", "technique": "UNKNOWN"},
        "waf_env": {"engine": "modsecurity", "paranoia_level": 1}
    }
    retrieved_ruby = retrieve_for_episode(example_episode_ruby, dummy_kb, top_k=2)
    print("\nRetrieved for Ruby Episode:")
    for r in retrieved_ruby:
        print(f"- {r.get('rule_id')}: {r.get('test_description')}")

