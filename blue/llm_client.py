import json
import os
import re
import sys
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the prompt template
from blue.prompts import BLUE_PROMPT_TEMPLATE

# Global variables for lazy loading the model
_model = None
_tokenizer = None
LLM_MODEL_ID = "google/gemma-2-2b-it"
# LLM_ADAPTER_PATH = "experiments/phase3_gemma2_2b_rl" # DISABLED: Do not use RED brain for BLUE task

def _load_llm_model():
    """
    Loads the LLM model and tokenizer for inference.
    This function will be called only once.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"Loading BLUE LLM (Base): {LLM_MODEL_ID}")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            # Load Base Model Only
            _model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                token=os.environ.get("HF_TOKEN")
            )
            
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=os.environ.get("HF_TOKEN"))
            _tokenizer.padding_side = "left"
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            print("BLUE LLM loaded successfully.")
        except Exception as e:
            print(f"Error loading BLUE LLM: {e}", file=sys.stderr)
            _model = None
            _tokenizer = None
            raise

def _generate_llm_response(prompt_text: str) -> str:
    """
    Generates a text response from the loaded LLM.
    """
    if _model is None or _tokenizer is None:
        _load_llm_model() # Ensure model is loaded

    # Gemma chat template format
    formatted_prompt = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
    inputs = _tokenizer(formatted_prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,    # Lower temperature for more deterministic output
            top_p=0.9,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id
        )
    response = _tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

def call_blue_llm(prompt: str) -> Dict[str, Any]:
    """
    Calls the BLUE LLM to get recommendations based on the prompt.
    """
    raw_response = "" # Initialize raw_response
    try:
        raw_response = _generate_llm_response(prompt)
        
        # --- Robust JSON Extraction and Cleaning (Nuclear Option) ---
        
        # 1. Aggressive Pre-cleaning
        cleaned_response = raw_response
        
        # Unescape underscores (common LLM markdown artifact: vuln\_effect -> vuln_effect)
        cleaned_response = cleaned_response.replace(r'\_', '_')
        
        # Remove Markdown formatting artifacts often found in code blocks
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

        # 2. Try to find a JSON block (e.g., within ```json ... ```)
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned_response, re.IGNORECASE)
        if json_match:
            json_string = json_match.group(1)
        else:
            # If no code block, try to find the first JSON-like object
            json_start = cleaned_response.find('{')
            # Find the last '}' that roughly balances or is at the end
            json_end = cleaned_response.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = cleaned_response[json_start : json_end + 1]
            else:
                # Fallback: Use raw text if it looks like JSON
                json_string = cleaned_response

        # 3. Further clean JSON string (remove comments and problematic control chars)
        json_string = re.sub(r',\s*([\}\]])', r'\1', json_string) # Remove trailing commas before } or ]
        json_string = re.sub(r'//.*', '', json_string)           # Remove single-line comments
        json_string = re.sub(r'/\*[\s\S]*?\*/', '', json_string) # Remove multi-line comments
        json_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_string) # Remove C0 and C1 control codes

        # Attempt to parse
        parsed_response = json.loads(json_string)

        # 4. Filter out unexpected keys and ensure schema adherence
        expected_keys = ["vuln_effect", "is_false_negative", "recommended_rules", "recommended_actions", "notes"]
        filtered_response = {k: parsed_response.get(k) for k in expected_keys if k in parsed_response}

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
                if isinstance(r, dict) and "rule_id" in r
            ]
        
        # Ensure recommended_actions is a list
        if not isinstance(filtered_response.get("recommended_actions"), list):
            filtered_response["recommended_actions"] = []

        return filtered_response

    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}", file=sys.stderr)
        print(f"Raw LLM response:\n{raw_response}", file=sys.stderr)
        return {
            "vuln_effect": "unknown",
            "is_false_negative": False,
            "recommended_rules": [],
            "recommended_actions": ["LLM_ERROR_OR_INVALID_JSON"],
            "notes": f"Error parsing LLM response: {e}. Raw response: {raw_response[:200]}..."
        }
    except ValueError as e:
        print(f"LLM response validation error: {e}", file=sys.stderr)
        print(f"Raw LLM response:\n{raw_response}", file=sys.stderr)
        return {
            "vuln_effect": "unknown",
            "is_false_negative": False,
            "recommended_rules": [],
            "recommended_actions": ["LLM_RESPONSE_VALIDATION_ERROR"],
            "notes": f"LLM response validation error: {e}. Raw response: {raw_response[:200]}..."
        }
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}", file=sys.stderr)
        return {
            "vuln_effect": "unknown",
            "is_false_negative": False,
            "recommended_rules": [],
            "recommended_actions": ["LLM_UNEXPECTED_ERROR"],
            "notes": f"Unexpected error during LLM call: {e}"
        }

if __name__ == "__main__":
    # Example usage for testing
    # Note: Model loading might take a moment the first time this is run.
    print("--- Testing BLUE LLM Client ---")

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

    print("\nAttempting to call BLUE LLM with example SQLI prompt...")
    response = call_blue_llm(test_prompt)
    print("\nBLUE LLM Response (SQLI example):")
    print(json.dumps(response, indent=2))

    # Another example for XSS
    example_episode_xss_json = json.dumps({
        "app_context": {"app_name": "dvwa", "http_method": "GET", "injection_point": "query_param"},
        "attack": {"attack_type": "XSS", "payload": "<script>alert(1)</script>"},
        "waf_observation": {"blocked": False, "http_status": 200},
        "app_observation": {"http_status": 200, "resp_body_snippet": "Hello <script>alert(1)</script>", "error_class": None},
        "blue_label": {"vuln_effect": "xss-reflected", "is_false_negative": True}
    }, indent=2)

    example_kb_xss_snippets = json.dumps([
        {"rule_id": "941100", "test_description": "XSS Attack Detected (Generic)", "operator": "@rx ..."}
    ], indent=2)

    test_prompt_xss = BLUE_PROMPT_TEMPLATE.format(
        EPISODE_JSON=example_episode_xss_json,
        KB_SNIPPETS=example_kb_xss_snippets
    )

    print("\nAttempting to call BLUE LLM with example XSS prompt...")
    response_xss = call_blue_llm(test_prompt_xss)
    print("\nBLUE LLM Response (XSS example):")
    print(json.dumps(response_xss, indent=2))
