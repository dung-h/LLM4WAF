import json
import os
import re
import sys
from typing import Dict, Any
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the prompt template
from blue.prompts import BLUE_PROMPT_TEMPLATE

# --- Backend Enum ---
class BlueModelBackend(str, Enum):
    GEMMA2 = "gemma2"
    PHI3_MINI = "phi3_mini"

def get_blue_backend() -> BlueModelBackend:
    """
    Đọc từ config hoặc env: BLUE_BACKEND=gemma2 hoặc BLUE_BACKEND=phi3_mini
    Mặc định: gemma2
    """
    backend_str = os.environ.get("BLUE_BACKEND", "gemma2").lower()
    if backend_str == "phi3_mini":
        return BlueModelBackend.PHI3_MINI
    return BlueModelBackend.GEMMA2

# --- Global Model/Tokenizer instances for lazy loading ---
_gemma2_model = None
_gemma2_tokenizer = None
_phi3_mini_model = None
_phi3_mini_tokenizer = None

# --- Model IDs ---
GEMMA2_MODEL_ID = "google/gemma-2-2b-it"
PHI3_MINI_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# LLM_ADAPTER_PATH for Gemma2 was "experiments/phase3_gemma2_2b_rl" # DISABLED

# --- Model Loading Utilities ---
def _load_model(model_id: str, is_gemma: bool = True):
    print(f"Loading LLM: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if is_gemma else tokenizer.unk_token # Phi-3 might not have eos_token as pad_token
        if tokenizer.pad_token is None: # Fallback if unk_token also none
            tokenizer.pad_token = tokenizer.convert_tokens_to_ids("[PAD]")
            if tokenizer.pad_token is None: # Add if it still doesn't exist
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"LLM {model_id} loaded successfully.")
    return model, tokenizer

# --- Generation Utilities ---
def _generate_response(model, tokenizer, prompt_text: str, model_type: BlueModelBackend) -> str:
    """
    Generates a text response from the loaded LLM based on its specific chat template.
    """
    if model_type == BlueModelBackend.GEMMA2:
        formatted_prompt = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
    elif model_type == BlueModelBackend.PHI3_MINI:
        # Phi-3 instruct format: <|user|>\n{message}<|end|>\n<|assistant|>\n
        # No system message for now, keep it simple and consistent with Gemma's user prompt approach
        formatted_prompt = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

# --- JSON Parsing Utilities ---
def _safe_parse_json(raw_response: str) -> Dict[str, Any]:
    """
    Applies aggressive cleaning and robust parsing to LLM raw output to get a valid JSON object.
    """
    cleaned_response = raw_response
    
    # Unescape underscores (common LLM markdown artifact: vuln\_effect -> vuln_effect)
    cleaned_response = cleaned_response.replace(r'\_', '_')
    
    # Remove Markdown bold/italic markers if they appear around keys or values
    cleaned_response = cleaned_response.replace('**', '') 
    
    # Standardize quotes (Smart quotes -> straight quotes)
    quote_map = {
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '‘': "'", '’': "'", '‚': "'", '‛': "'"
    }
    for k, v in quote_map.items():
        cleaned_response = cleaned_response.replace(k, v)
    
    # Replace unescaped backslashes with double backslashes for JSON compatibility
    # This regex looks for a single backslash not followed by a valid JSON escape character
    # (", \, /, b, f, n, r, t, uXXXX)
    cleaned_response = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cleaned_response)

    # Try to find a JSON block (e.g., within ```json ... ```)
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned_response, re.IGNORECASE)
    if json_match:
        json_string = json_match.group(1)
    else:
        # If no code block, try to find the first JSON-like object
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_string = cleaned_response[json_start : json_end + 1]
        else:
            # Fallback: Use raw text if it looks like JSON (risky)
            json_string = cleaned_response

    # Further clean JSON string (remove trailing commas, comments, and problematic control chars)
    json_string = re.sub(r',\s*([\}\]])', r'\1', json_string) # Remove trailing commas before } or ]
    json_string = re.sub(r'//.*', '', json_string)           # Remove single-line comments
    json_string = re.sub(r'/\*[\s\S]*?\*/', '', json_string) # Remove multi-line comments
    json_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_string) # Remove C0 and C1 control codes

    # Attempt to parse
    parsed_response = json.loads(json_string)

    # Filter out unexpected keys and ensure schema adherence
    expected_keys = ["vuln_effect", "is_false_negative", "recommended_rules", "recommended_actions", "notes"]
    filtered_response = {k: parsed_response.get(k) for k in expected_keys}

    # Validate basic schema of the response (after filtering)
    if not isinstance(filtered_response, dict) or \
       "vuln_effect" not in filtered_response or \
       "is_false_negative" not in filtered_response or \
       "recommended_rules" not in filtered_response:
        raise ValueError("LLM response missing required fields or invalid structure even after filtering.")

    # Ensure recommended_rules is a list, and each item is a dict with 'rule_id'
    if not isinstance(filtered_response.get("recommended_rules"), list):
        filtered_response["recommended_rules"] = []
    else:
        filtered_response["recommended_rules"] = [
            r for r in filtered_response["recommended_rules"] 
            if isinstance(r, dict) and "rule_id" in r and "engine" in r # Ensure 'engine' is also present
        ]
    
    # Ensure recommended_actions is a list
    if not isinstance(filtered_response.get("recommended_actions"), list):
        filtered_response["recommended_actions"] = []

    return filtered_response

# --- Specific Backend Call Functions ---
def _call_blue_gemma2(prompt: str) -> Dict[str, Any]:
    """
    Logic for calling Gemma 2 2B IT (base) for BLUE.
    """
    global _gemma2_model, _gemma2_tokenizer
    if _gemma2_model is None or _gemma2_tokenizer is None:
        _gemma2_model, _gemma2_tokenizer = _load_model(GEMMA2_MODEL_ID, is_gemma=True)
    
    raw_response = _generate_response(_gemma2_model, _gemma2_tokenizer, prompt, BlueModelBackend.GEMMA2)
    return _safe_parse_json(raw_response)

def _call_blue_phi3_mini(prompt: str) -> Dict[str, Any]:
    """
    Logic for calling Phi-3 Mini for BLUE.
    """
    global _phi3_mini_model, _phi3_mini_tokenizer
    if _phi3_mini_model is None or _phi3_mini_tokenizer is None:
        _phi3_mini_model, _phi3_mini_tokenizer = _load_model(PHI3_MINI_MODEL_ID, is_gemma=False)
    
    raw_response = _generate_response(_phi3_mini_model, _phi3_mini_tokenizer, prompt, BlueModelBackend.PHI3_MINI)
    return _safe_parse_json(raw_response)


# --- Main Wrapper Function ---
def call_blue_llm(prompt: str) -> Dict[str, Any]:
    """
    Wrapper chính: chọn backend LLM theo get_blue_backend() và gọi hàm tương ứng.
    Bao gồm logic retry nếu JSON không hợp lệ.
    """
    backend = get_blue_backend()
    
    response_dict = {}
    raw_llm_output_first_try = "" # Store for logging

    for attempt in range(2): # Try twice
        try:
            if backend == BlueModelBackend.GEMMA2:
                response_dict = _call_blue_gemma2(prompt)
            elif backend == BlueModelBackend.PHI3_MINI:
                response_dict = _call_blue_phi3_mini(prompt)
            
            # If successful, return the response
            if response_dict.get("recommended_actions") != ["LLM_ERROR_OR_INVALID_JSON"] and \
               response_dict.get("recommended_actions") != ["LLM_RESPONSE_VALIDATION_ERROR"] and \
               response_dict.get("recommended_actions") != ["LLM_UNEXPECTED_ERROR"]:
                return response_dict
            else:
                # If it's a fallback error, store raw output of first try
                if attempt == 0:
                    if backend == BlueModelBackend.GEMMA2:
                         raw_llm_output_first_try = _generate_response(_gemma2_model, _gemma2_tokenizer, prompt, BlueModelBackend.GEMMA2)
                    elif backend == BlueModelBackend.PHI3_MINI:
                         raw_llm_output_first_try = _generate_response(_phi3_mini_model, _phi3_mini_tokenizer, prompt, BlueModelBackend.PHI3_MINI)

                # If first attempt failed JSON parsing, try again with a stricter prompt
                if attempt == 0:
                    print(f"[{backend}] JSON parsing failed on first attempt. Retrying with stricter prompt...", file=sys.stderr)
                    stricter_prompt = prompt + "\nReturn ONLY the JSON object. Do not include any explanation or text outside the JSON."
                    prompt = stricter_prompt # Use stricter prompt for next attempt
                    
        except Exception as e:
            # Catching exceptions during the _call_blue_XXX functions themselves
            if attempt == 0:
                print(f"[{backend}] LLM call or initial JSON parsing failed on first attempt: {e}", file=sys.stderr)
                if backend == BlueModelBackend.GEMMA2:
                     raw_llm_output_first_try = _generate_response(_gemma2_model, _gemma2_tokenizer, prompt, BlueModelBackend.GEMMA2) # Re-generate to capture
                elif backend == BlueModelBackend.PHI3_MINI:
                     raw_llm_output_first_try = _generate_response(_phi3_mini_model, _phi3_mini_tokenizer, prompt, BlueModelBackend.PHI3_MINI)
                
                stricter_prompt = prompt + "\nReturn ONLY the JSON object. Do not include any explanation or text outside the JSON."
                prompt = stricter_prompt # Use stricter prompt for next attempt
            else:
                print(f"[{backend}] LLM call or JSON parsing failed on second attempt: {e}", file=sys.stderr)

    # If both attempts fail or return an error fallback
    fallback_actions = [f"LLM_ERROR_OR_INVALID_JSON_{backend.value.upper()}"]
    fallback_notes = f"Failed to get valid JSON from {backend.value} after 2 attempts. Raw output (first try): {raw_llm_output_first_try[:200]}..." if raw_llm_output_first_try else f"Failed to get valid JSON from {backend.value} after 2 attempts."
    
    return {
        "vuln_effect": "unknown",
        "is_false_negative": False,
        "recommended_rules": [],
        "recommended_actions": fallback_actions,
        "notes": fallback_notes
    }

if __name__ == "__main__":
    # Example usage for testing
    # Note: Model loading might take a moment the first time this is run.
    print("--- Testing BLUE LLM Client Multi-Backend ---")

    # Example EPISODE_JSON and KB_SNIPPETS (simplified for a quick test)
    example_episode_json = json.dumps({
        "app_context": {"app_name": "dvwa", "http_method": "GET", "injection_point": "query_param"},
        "attack": {"attack_type": "SQLI", "payload": "1' OR '1'='1'"},
        "waf_observation": {"blocked": False, "http_status": 200},
        "app_observation": {"http_status": 200, "resp_body_snippet": "You are logged in!", "error_class": None},
        "blue_label": {"vuln_effect": "sql-executed", "is_false_negative": True}
    }, indent=2)

    example_kb_snippets = json.dumps([
        {"rule_id": "942100", "test_description": "SQL Injection Attack Detected", "operator": "@rx ..."},
        {"rule_id": "942150", "test_description": "SQL Comment Sequence Detected", "operator": "@rx ..."}
    ], indent=2)

    test_prompt = BLUE_PROMPT_TEMPLATE.format(
        EPISODE_JSON=example_episode_json,
        KB_SNIPPETS=example_kb_snippets
    )

    # Test Gemma2
    os.environ["BLUE_BACKEND"] = "gemma2"
    print("\n--- Testing Gemma2 Backend ---")
    response_gemma = call_blue_llm(test_prompt)
    print("\nBLUE LLM Response (Gemma2):")
    print(json.dumps(response_gemma, indent=2))

    # Test Phi-3 Mini
    os.environ["BLUE_BACKEND"] = "phi3_mini"
    print("\n--- Testing Phi-3 Mini Backend ---")
    response_phi3 = call_blue_llm(test_prompt)
    print("\nBLUE LLM Response (Phi-3 Mini):")
    print(json.dumps(response_phi3, indent=2))