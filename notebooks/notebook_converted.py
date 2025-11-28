# Cell 1
# ===================================
#  Ô CẤU HÌNH - VUI LÒNG CHỈNH SỬA!  #
# ===================================

# 1. Đặt tên dataset của bạn ở đây (phần sau /kaggle/input/).
# Ví dụ: nếu bạn tạo dataset 'my-v13-sft-data' chứa 'v13_sft_data.jsonl', hãy đặt là "my-v13-sft-data"
DATASET_NAME = "llm4waf" 

# 2. Chọn MỘT mô hình để huấn luyện từ danh sách.
# Các tùy chọn hợp lệ: "deepseek", "olmoe", "openhermes"
MODEL_TO_TRAIN = "deepseek"

# In ra để xác nhận
print(f"✅ Dataset được đặt thành: /kaggle/input/{DATASET_NAME}")
print(f"✅ Mô hình được chọn để huấn luyện: {MODEL_TO_TRAIN}")

# Cell 3
print("--- Installing dependencies ---")
%pip install -q --upgrade pip
%pip install -q \
    "torch" \
    "transformers>=4.41.0" \
    "accelerate>=0.30.0" \
    "datasets>=2.19.0" \
    "bitsandbytes>=0.43.0" \
    "peft>=0.11.1" \
    "trl>=0.9.6" \
    "pyyaml"
print("✅ Dependencies installed.")

# Cell 5
%%writefile scripts/kaggle_train.py
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
from trl import SFTTrainer, SFTConfig

def require_env_token(var: str = "HF_TOKEN") -> None:
    '''Check for HF token from Kaggle secrets.'''
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
    model_name = cfg.get("model_name", "").lower()
    if "mistral" in model_name or "hermes" in model_name:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
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

    # Tuyệt đối không xuống dòng trong string literal
    user_message = (
        "Instruction: " + instr + "\n"
        + "Context: " + ctx + "\n"
        + "Constraints: " + cons
    )

    assistant_message = (
        "Payload: " + payload + "\n"
        + "Reasoning: " + reasoning
    )

    if prompt_format_type == "openhermes":
        return (
            "<|im_start|>user\n"
            + user_message
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
            + assistant_message
            + "<|im_end|>")
    else:
        return user_message + "\n\n" + assistant_message

def formatting_func(examples: Dict[str, List[str]], fields: Dict[str, str] | None, prompt_format_type: str) -> List[str]:
    # Nếu không truyền mapping từ config, dùng default
    if fields is None:
        fields = {
            "instruction": "instruction",
            "context": "context",
            "constraints": "constraints",
            "payload": "payload",
            "reasoning": "reasoning",
        }

    out = []
    payload_col = fields["payload"]  # tên cột thực trong dataset (thường là "payload")

    for i in range(len(examples[payload_col])):
        ex = {k: examples[k][i] for k in examples}
        out.append(format_example(ex, fields, prompt_format_type))
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

    text_fields = cfg.get("text_fields", None)

    def _fmt_func(batch):
        return {"text": formatting_func(batch, text_fields, prompt_format_type)}

    ds_proc = ds["train"].map(_fmt_func, batched=True, remove_columns=ds["train"].column_names)

    out_dir = cfg.get("output_dir")
    sft_config = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        fp16=False,
        bf16=True,
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to=[],
    )


    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds_proc,
        processing_class=tok,  # bạn đã đổi từ tokenizer -> processing_class trước đó
    )

    print("Starting training…")
    trainer.train()
    
    final_adapter_path = os.path.join(out_dir, "adapter")
    print(f"Saving final adapter to {final_adapter_path}...")
    trainer.model.save_pretrained(final_adapter_path)
    print("Done.")

if __name__ == "__main__":
    main()

# Cell 6
import os

# Tạo các thư mục cần thiết
os.makedirs("scripts", exist_ok=True)
os.makedirs("configs", exist_ok=True)


# with open("scripts/kaggle_train.py", "w") as f:
#     f.write(kaggle_train_script)

# print("Created scripts/kaggle_train.py")

# Create config files
def create_config(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)

deepseek_config = f"""
model_name: \"deepseek-ai/deepseek-llm-7b-base\"
prompt_format_type: \"default\"
train_path: \"/kaggle/input/{DATASET_NAME}/v13_sft_data.jsonl\"
lora_r: 16
num_train_epochs: 3
output_dir: \"/kaggle/working/deepseek_7b_base_adapter\"
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
"""
create_config("configs/kaggle_deepseek_7b_base.yaml", deepseek_config)
print("Created configs/kaggle_deepseek_7b_base.yaml")

olmoe_config = f"""
model_name: "allenai/OLMoE-7B-Instruct"
prompt_format_type: "default"
train_path: "/kaggle/input/{DATASET_NAME}/v13_sft_data.jsonl"
lora_r: 16
num_train_epochs: 3
output_dir: "/kaggle/working/olmoe_7b_instruct_v13_adapter"
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
"""
create_config("configs/kaggle_olmoe_7b.yaml", olmoe_config)
print("Created configs/kaggle_olmoe_7b.yaml")

openhermes_config = f"""
model_name: "teknium/OpenHermes-2.5-Mistral-7B"
prompt_format_type: "openhermes"
train_path: "/kaggle/input/{DATASET_NAME}/v13_sft_data.jsonl"
lora_r: 16
num_train_epochs: 3
output_dir: "/kaggle/working/openhermes_7b_v13_adapter"
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
"""
create_config("configs/kaggle_openhermes_7b.yaml", openhermes_config)
print("Created configs/kaggle_openhermes_7b.yaml")

print("\n✅ All files created successfully.")

# Cell 8
import os

# Đọc lại các biến từ ô cấu hình (để đảm bảo chúng được đặt)
config_map = {
    "deepseek": "configs/kaggle_deepseek_7b_base.yaml",
    "olmoe": "configs/kaggle_olmoe_7b.yaml",
    "openhermes": "configs/kaggle_openhermes_7b.yaml"
}

config_file = config_map.get(MODEL_TO_TRAIN)

if config_file:
    print(f"--- Starting training for {MODEL_TO_TRAIN} ---")
    # Sử dụng os.system để chạy lệnh shell
    exit_code = os.system(f"python scripts/kaggle_train.py --config {config_file}")
    if exit_code != 0:
        raise RuntimeError(f"Training failed with exit code {exit_code}")
else:
    print(f"ERROR: Invalid model choice '{MODEL_TO_TRAIN}'. Please choose from {list(config_map.keys())}")

# Cell 10
import os

adapter_dir_map = {
    "deepseek": "deepseek_7b_base_adapter",
    "olmoe": "olmoe_7b_instruct_v13_adapter",
    "openhermes": "openhermes_7b_v13_adapter"
}

adapter_folder_name = adapter_dir_map.get(MODEL_TO_TRAIN)
output_path = f"/kaggle/working/{adapter_folder_name}"
archive_name = f"{adapter_folder_name}.tar.gz"

if adapter_folder_name and os.path.exists(output_path):
    print(f"--- Packaging adapter from {output_path} ---")
    # Lệnh tar: -c: tạo archive, -z: nén với gzip, -f: chỉ định tên tệp, -C: thay đổi thư mục trước khi nén
    os.system(f"tar -czf {archive_name} -C {output_path} .")
    print(f"✅ Adapter packaged to /kaggle/working/{archive_name}")
    print("You can now find this file in the 'Output' section and download it.")
else:
    print(f"ERROR: Could not find adapter directory '{output_path}' to package. Did training complete successfully?")