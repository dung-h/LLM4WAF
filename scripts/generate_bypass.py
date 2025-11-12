
import asyncio
import httpx
import re
from urllib.parse import urlsplit
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langchain_core.prompts import PromptTemplate

# --- LLM & Path Config ---
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ADAPTER_DIR = "experiments/red_gemma2_v6_multi_clean/adapter"
BASE_MODEL_NAME = "google/gemma-2-2b-it"

# --- New Prompt Template ---
BYPASS_SQLI_PROMPT = PromptTemplate.from_template(
    """
You are a WAF penetration testing expert.

**Analysis of Target WAF:**
- The WAF blocks requests containing common SQL keywords like 'OR', 'UNION', 'AND'.
- The WAF *does not* block requests with simple SQL punctuation like a single quote (').
- The target is a standard MySQL database.

**Previous Attempt Analysis:**
- A previous attempt using the payload `1' or '1[' or 1 = 1 -- 1` failed.
- It was blocked because it still contained the keyword 'OR'. The obfuscation was not enough.

**Your New Goal:**
Generate a single, effective SQL injection payload that **completely avoids** the keywords 'OR', 'UNION', 'AND'.

**Chain of Thought Rules:**
1.  First, think step-by-step inside a <thought> block about how to achieve this.
2.  Your primary focus is to **not use the blocked keywords**.
3.  Consider time-based blind techniques (e.g., using `SLEEP()`, `BENCHMARK()`) or boolean-based techniques using different syntax (e.g., `&&`, `||`, `XOR`, `LIKE`).
4.  After your thought process, write the final payload on a new line, prefixed with "Payload:".

<thought>
</thought>

Payload:
""".strip()
)

def load_adapter(adapter_dir: str):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map='auto', quantization_config=bnb, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.float16)
    model.eval()
    return tok, model

def get_payload_from_text(text: str) -> str:
    """
    Extracts the payload from the model's full text output.
    It splits by "Payload:" and takes the last occurrence.
    """
    try:
        # Take the part after the LAST "Payload:" marker
        payload_part = text.split("Payload:")[-1]
        
        # Find the first non-empty line in that part
        lines = [line.strip() for line in payload_part.splitlines() if line.strip()]
        if lines:
            return lines[0]
        return ""
    except IndexError:
        return ""

async def main():
    print("--- Starting LLM Bypass Payload Generation (Fixed) ---")
    
    tok, model = load_adapter(ADAPTER_DIR)
    
    generated_payloads = []
    
    print(f"[+] Generating 1 SQLi bypass payload...")
    for i in range(1):
        prompt = BYPASS_SQLI_PROMPT.format()
        inputs = tok(prompt, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
            
        text = tok.decode(out[0], skip_special_tokens=True)
        
        payload = get_payload_from_text(text)

        if payload:
            print(f"  - Generated Payload {i+1}: {payload}")
            generated_payloads.append(payload)
        else:
            print(f"  - FAILED: Could not extract payload for generation #{i+1}.")

    output_file = RESULTS / "v8_sqli_bypass_payloads.txt"
    output_file.write_text("\n".join(generated_payloads), encoding="utf-8")
    
    print(f"\n[+] Saved {len(generated_payloads)} payloads to {output_file}")
    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
