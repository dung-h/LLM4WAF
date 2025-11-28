import argparse
import json
import os
import time
import logging
from dataclasses import dataclass

import yaml
from typing import Dict, Any, List

# Parse args first to set GPU before importing torch
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (0-3)")
    parser.add_argument("--log-suffix", type=str, default="", help="Log file suffix")
    return parser.parse_args()

args = parse_args()
# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

# --- Setup Logging ---
# Ensure logs are written to the file and flushed, preventing cache issues.
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# File handler with suffix support
def setup_logging(suffix=""):
    log_file = f"SFT{suffix}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    return file_handler

# Will be setup in main with suffix
file_handler = None

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


def require_env_token(var: str = "HF_TOKEN") -> None:
    """Check for HF token, use cached token if available"""
    if not os.environ.get(var):
        # Try to load from cache
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        logger.debug(f"Checking token path: {token_path}")
        logger.debug(f"Path exists: {os.path.exists(token_path)}")
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                token = f.read().strip()
                os.environ[var] = token
                logger.info(f"✅ Loaded HF_TOKEN from cache (len={len(token)})")
                return
        logger.debug(f"Token not found, aborting")
        raise SystemExit(f"Environment variable {var} not set and no cached token found. Aborting for safety.")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "float16")),
    )


def build_lora_config(cfg: Dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=cfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        # use_dora=bool(cfg.get("use_dora", True)),  # Not supported in peft==0.6.0
    )


def main() -> None:
    # Setup logging with suffix
    global file_handler
    log_suffix = f"_gpu{args.gpu}{args.log_suffix}" if args.log_suffix else f"_gpu{args.gpu}"
    file_handler = setup_logging(log_suffix)
    logger.addHandler(file_handler)
    
    require_env_token("HF_TOKEN")

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    auth_env = cfg.get("use_auth_token_env", "HF_TOKEN")
    auth_token = os.environ.get(auth_env)

    bnb_cfg = build_bnb_config(cfg)

    # GPU is already set via CUDA_VISIBLE_DEVICES at import time
    torch.cuda.set_device(0)  # Only visible GPU is now GPU 0
    logger.info(f"🎯 Using GPU {args.gpu} (visible as GPU 0) for {model_name}")
    
    logger.info("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
    tok.padding_side = cfg.get("padding_side", "left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    logger.info("Loading 4-bit model… (this may take a while)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        device_map={"": 0},  # Use GPU 0 (which is the only visible GPU)
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    logger.info("Preparing k-bit training and applying DoRA adapters…")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)

    # Load datasets
    train_path = cfg["train_path"]
    eval_path = cfg.get("eval_path")

    data_files = {"train": train_path}
    if eval_path and os.path.exists(eval_path):
        data_files["validation"] = eval_path

    ds = load_dataset("json", data_files=data_files)

    out_dir = cfg.get("output_dir", f"experiments/red_{int(time.time())}")

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(cfg.get("num_train_epochs", 2)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        fp16=bool(cfg.get("fp16", True)),
        bf16=bool(cfg.get("bf16", False)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        eval_steps=int(cfg.get("eval_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        report_to=["tensorboard"],
        logging_dir=f"{out_dir}/logs",
        logging_first_step=True,
        disable_tqdm=False,
        log_level="info",
        log_on_each_node=False,
        # Force single GPU training
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # Disable DataParallel
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )

    # Optional max_steps override for smoke tests
    if "max_steps" in cfg and int(cfg["max_steps"]) > 0:
        targs.max_steps = int(cfg["max_steps"])  # type: ignore

    # Ensure model is not wrapped with DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    logger.info("Initializing SFTTrainer…")
    
    # Auto-detect format
    train_cols = ds["train"].column_names
    sft_kwargs = {}
    if "messages" in train_cols:
        logger.info("Dataset has 'messages' column. Using automatic chat template application.")
        # Do NOT set dataset_text_field, SFTTrainer handles 'messages' by default if tokenizer has chat_template
    elif "prompt" in train_cols:
        logger.info("Dataset has 'prompt' column. Using 'prompt' as text field.")
        sft_kwargs["dataset_text_field"] = "prompt"
    else:
        logger.warning(f"Unknown dataset format. Columns: {train_cols}. Assuming 'text' field or letting SFTTrainer decide.")
        # sft_kwargs["dataset_text_field"] = "text" # Let defaults handle it

    trainer = SFTTrainer(
        model=model,
        # tokenizer=tok,  # Not supported in some trl versions
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        max_seq_length=int(cfg.get("seq_length", 2048)),  # Use config value, default 2048
        packing=False,
        **sft_kwargs  # Auto-detected dataset_text_field or none for messages format
    )

    logger.info("Starting training… (ctrl+c to stop)")
    trainer.train()
    logger.info("Saving model using trainer.save_model()…")
    trainer.save_model(out_dir) # save to the output_dir directly
    logger.info("Done saving model.")


if __name__ == "__main__":
    main()