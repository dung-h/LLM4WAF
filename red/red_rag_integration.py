import sys
import os
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from red.rag_internal_client import get_client

def build_red_prompt_with_rag(
    attack_type: str,
    target_desc: str, # e.g., "DVWA SQLi, input 'id'"
    signals: Dict[str, Any], # e.g., {"blocked_keywords": ["script"], "waf_profile_hint": "modsec_pl1"}
    history_payloads: List[Dict], # List of {"payload": "...", "status": "..."}
    corpus_version: str = "v2"
) -> Tuple[str, List[Dict[str, Any]]]: # Changed return type
    """
    Builds a detailed prompt for the RED LLM, incorporating context from the RAG system.
    Returns a tuple: (prompt_text, list_of_rag_docs_metadata).
    """
    rag_client = get_client(corpus_version=corpus_version)
    
    # Retrieve relevant knowledge from RAG
    # Prioritize strategies and patterns related to current attack_type and signals
    retrieved_docs = rag_client.retrieve_red_knowledge(
        attack_type=attack_type,
        signals=signals,
        top_k=5 # Retrieve top 5 relevant documents
    )

    rag_context = ""
    rag_docs_used = [] # List to store metadata of used RAG docs
    if retrieved_docs:
        rag_context += "\n--- RAG Context (Past Learnings & Patterns) ---\n"
        for i, doc in enumerate(retrieved_docs):
            rag_context += f"Doc {i+1} (Kind: {doc['kind']}, Type: {doc['attack_type']}):\n"
            rag_context += f"  ID: {doc['doc_id']}\n"
            rag_context += f"  Summary: {doc.get('short_desc') or doc.get('idea') or doc.get('text_preview')}\n"
            if doc.get('example_payloads'):
                rag_context += f"  Example Payload (partial): {doc['example_payloads'][0][:80]}...\n"
            rag_context += "\n"
            
            # Add metadata of used doc to rag_docs_used
            technique_info = doc.get('technique')
            if technique_info is None:
                # If 'technique' is not present, check for 'family_name' (for attack_strategy docs)
                technique_info = doc.get('family_name')
            
            if technique_info is None:
                technique_info = "N/A" # Default if neither technique nor family_name is found
            elif isinstance(technique_info, tuple):
                technique_info = str(technique_info) # Ensure it's a string, not a tuple
            
            rag_docs_used.append({
                "doc_id": doc['doc_id'],
                "kind": doc['kind'],
                "attack_type": doc['attack_type'],
                "technique": technique_info, # Use the safely determined technique_info
                "waf_profile": doc.get('waf_profile')
            })
        rag_context += "----------------------------------------------\n"
    
    # Build history section
    history_str = ""
    if history_payloads:
        history_str += "\n--- Attack History (against current target) ---\n"
        for i, entry in enumerate(history_payloads):
            history_str += f"Attempt {i+1}: Payload '{entry['payload'][:80]}...' resulted in '{entry['status']}'\n"
        history_str += "---------------------------------------------\n"

    # Construct the final prompt for the RED LLM
    final_prompt = f"""You are an expert Red Teaming LLM, specialized in generating highly effective WAF evasion payloads.
Your goal is to bypass the target WAF and trigger the underlying vulnerability.

Target description: {target_desc}

{history_str}

{rag_context}

Based on the above context and your expertise, generate a NEW payload to bypass the WAF.
Focus on techniques and patterns suggested by the RAG context if applicable.
Output ONLY the payload string. Do NOT add any explanations or comments.
"""
    return final_prompt, rag_docs_used # Return both prompt and rag_docs_used