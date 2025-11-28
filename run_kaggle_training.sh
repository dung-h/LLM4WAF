#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- USER CONFIGURATION ---
# IMPORTANT: Before running, please edit the DATASET_NAME variable below
# to match the name of the Kaggle dataset where you uploaded your .jsonl files.
# For example, if your dataset is named 'my-waf-data', set it to "my-waf-data".
DATASET_NAME="your-dataset-name-here"

# --- SCRIPT START ---

# Check if DATASET_NAME is set
if [ "$DATASET_NAME" == "your-dataset-name-here" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! ERROR: Please edit the DATASET_NAME variable in this script before running. !!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi

# Check if a model argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {deepseek|olmoe|openhermes}"
    exit 1
fi
MODEL_CHOICE=$1

echo "--- Step 1: Installing dependencies ---"
# Suppress pip progress bar to keep logs clean
export PIP_PROGRESS_BAR=off
pip install --upgrade pip > /dev/null
pip install -q \
    "torch" \
    "transformers>=4.41.0" \
    "accelerate>=0.30.0" \
    "datasets>=2.19.0" \
    "bitsandbytes>=0.43.0" \
    "peft>=0.11.1" \
    "trl>=0.9.6" \
    "pyyaml"
echo "Dependencies installed."

echo "--- Step 2: Creating training script and configuration files ---"
mkdir -p scripts configs

# Create the Python training script using a heredoc
cat > scripts/kaggle_train.py <<'EOF'
import argparse
import json
import os
import time
from dataclasses import dataclass

import yaml
from typing import Dict, Any, List

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def require_env_token(var: str = "HF_TOKEN") -> None:
    """Check for HF token from Kaggle secrets."""
    from kaggle_secrets import UserSecretsClient
    try:
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret(var)
        os.environ[var] = token
        print(f"✅ Successfully loaded {var} from Kaggle Secrets.")
    except Exception as e:
        print(f"❌ Could not load {var} from Kaggle Secrets. Please ensure it is set.", e)
        raise SystemExit("Aborting for safety.")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_bnb_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
    )

def build_lora_config(cfg: Dict[str, Any]) -> LoraConfig:
    # Mistral/Hermes models use different module names
    model_name = cfg.get("model_name", "").lower()
    if "mistral" in model_name or "hermes" in model_name:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else: # Default for Llama/DeepSeek
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    return LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=cfg.get("target_modules", target_modules),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
    )

def format_example(example: Dict[str, Any], fields: Dict[str, str], prompt_format_type: str = "default") -> str:
    payload = (example.get(fields["payload"]) or "").strip()
    reasoning = (example.get(fields["reasoning"]) or "").strip()
    instr = (example.get(fields.get("instruction")) or "").strip()
    ctx = (example.get(fields.get("context")) or "").strip()
    cons = (example.get(fields.get("constraints")) or "").strip()
    
    user_message = f"Instruction: {instr}\nContext: {ctx}\nConstraints: {cons}"
    assistant_message = f"Payload: {payload}\nReasoning: {reasoning}"

    if prompt_format_type == "openhermes":
        return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_message}<|im_end|>"
    else: # Default for base models
        return f"{user_message}\n\n{assistant_message}"

def formatting_func(examples: Dict[str, List[str]], fields, prompt_format_type: str) -> List[str]:
    fields_dict = {
        "instruction": "instruction", "context": "context", "constraints": "constraints",
        "payload": "payload", "reasoning": "reasoning"
    }
    out = []
    for i in range(len(examples[fields_dict["payload"]])):
        ex = {k: examples[k][i] for k in examples}
        out.append(format_example(ex, fields_dict, prompt_format_type))
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    require_env_token("HF_TOKEN")

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    auth_token = os.environ.get("HF_TOKEN")
    prompt_format_type = cfg.get("prompt_format_type", "default")

    bnb_cfg = build_bnb_config(cfg)

    print(f"Loading tokenizer for {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
    tok.padding_side = cfg.get("padding_side", "left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading 4-bit model: {model_name} (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print("Preparing k-bit training and applying LoRA adapters…")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)

    data_files = {"train": cfg["train_path"]}
    if cfg.get("eval_path") and os.path.exists(cfg["eval_path"]):
        data_files["validation"] = cfg["eval_path"]

    ds = load_dataset("json", data_files=data_files)

    def _fmt_func(batch):
        return {"text": formatting_func(batch, cfg["text_fields"], prompt_format_type)}

    ds_proc = ds["train"].map(_fmt_func, batched=True, remove_columns=ds["train"].column_names)

    out_dir = cfg.get("output_dir")
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        fp16=False, # Not recommended with bfloat16
        bf16=True,  # Use bfloat16 for better performance on modern GPUs
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=targs,
        train_dataset=ds_proc,
        max_seq_length=int(cfg.get("max_seq_length", 1024)),
        packing=False,
        dataset_text_field="text",
    )

    print("Starting training…")
    trainer.train()
    
    final_adapter_path = os.path.join(out_dir, "adapter")
    print(f"Saving final adapter to {final_adapter_path}...")
    trainer.model.save_pretrained(final_adapter_path)
    print("Done.")

if __name__ == "__main__":
    main()
EOF

# Create config for DeepSeek-7B-Base
cat > configs/kaggle_deepseek_7b_base.yaml <<EOF
model_name: "deepseek-ai/deepseek-llm-7b-base"
prompt_format_type: "default"
train_path: "/kaggle/input/$DATASET_NAME/red_train_v6_multi_clean.jsonl"
text_fields: ["instruction", "context", "constraints", "payload", "reasoning"]
lora_r: 16
lora_alpha: 32
num_train_epochs: 3
output_dir: "/kaggle/working/deepseek_7b_base_adapter"
EOF

# Create config for OLMoE-7B
cat > configs/kaggle_olmoe_7b.yaml <<EOF
model_name: "allenai/OLMoE-1B-7B-0125"
prompt_format_type: "default"
train_path: "/kaggle/input/$DATASET_NAME/red_train_v6_multi_clean.jsonl"
text_fields: ["instruction", "context", "constraints", "payload", "reasoning"]
lora_r: 16
lora_alpha: 32
num_train_epochs: 3
output_dir: "/kaggle/working/olmoe_7b_adapter"
EOF

# Create config for OpenHermes-2.5-Mistral-7B
cat > configs/kaggle_openhermes_7b.yaml <<EOF
model_name: "teknium/OpenHermes-2.5-Mistral-7B"
prompt_format_type: "openhermes" # Specific format for this model
train_path: "/kaggle/input/$DATASET_NAME/red_train_v6_multi_clean.jsonl"
text_fields: ["instruction", "context", "constraints", "payload", "reasoning"]
lora_r: 16
lora_alpha: 32
num_train_epochs: 3
output_dir: "/kaggle/working/openhermes_7b_adapter"
EOF

echo "✅ All files created successfully."
echo ""

# --- Training Logic ---

train_model() {
  CONFIG_PATH=$1
  ADAPTER_DIR_NAME=$(grep 'output_dir:' "$CONFIG_PATH" | awk '{print $2}' | tr -d '"')
  
  echo "--- Step 3: Starting Training for $MODEL_CHOICE ---"
  echo "Using config: $CONFIG_PATH"
  
  python scripts/kaggle_train.py --config "$CONFIG_PATH"
  
  echo "--- Step 4: Packaging Adapter ---"
  # Use the directory name from the config for packaging
  PACKAGE_NAME=$(basename "$ADAPTER_DIR_NAME")
  tar -czf "${PACKAGE_NAME}.tar.gz" -C "$ADAPTER_DIR_NAME" .
  
  echo "✅ Training complete."
  echo "✅ Adapter packaged to /kaggle/working/${PACKAGE_NAME}.tar.gz"
  echo "You can now download this file from the 'Output' section of your Kaggle notebook."
}

# --- Model Selection ---

case "$MODEL_CHOICE" in
  deepseek)
    train_model "configs/kaggle_deepseek_7b_base.yaml"
    ;;
  olmoe)
    train_model "configs/kaggle_olmoe_7b.yaml"
    ;;
  openhermes)
    train_model "configs/kaggle_openhermes_7b.yaml"
    ;;
  *)
    echo "Invalid model choice: $MODEL_CHOICE"
    echo "Usage: $0 {deepseek|olmoe|openhermes}"
    exit 1
    ;;
esac
