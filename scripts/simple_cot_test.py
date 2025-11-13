import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import yaml
from pathlib import Path

# --- Config ---
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "red_llm_dora_8gb.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

BASE_MODEL_NAME = config.get("model_name")
ADAPTER_PATH = config.get("adapter_path")

# --- Model Loading ---
print(f"[Test Script] Loading base model: {BASE_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

if ADAPTER_PATH:
    print(f"[Test Script] Loading and merging LoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    print("[Test Script] Fine-tuned model loaded successfully.")
else:
    print("[Test Script] Base model loaded (no adapter).")

model.eval()

# --- Simple CoT Prompt ---
prompt = (
    "<|system|>\n"
    "You are a helpful AI assistant.<|end|>\n"
    "<|user|>\n"
    "Generate a MySQL SQL injection payload to bypass a WAF. Think step-by-step and then write the final payload on a new line prefixed with 'Payload:'.<|end|>\n"
    "<|assistant|>\n"
)

print("\n--- Prompt ---")
print(prompt)

# --- Generation ---
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract only the assistant's response
assistant_response_start = response_text.rfind("<|assistant|>")
if assistant_response_start != -1:
    assistant_response = response_text[assistant_response_start + len("<|assistant|>"):].strip()
else:
    # Fallback if the marker isn't found (less likely but safe)
    user_prompt_end = response_text.rfind("<|end|>")
    if user_prompt_end != -1:
        assistant_response = response_text[user_prompt_end + len("<|end|>"):].strip()
    else:
        assistant_response = "Could not parse assistant response."


print("\n--- Full Model Response ---")
print(assistant_response)
