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
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID(s) to use: single '0' or multi '0,1'")
    parser.add_argument("--log-suffix", type=str, default="", help="Log file suffix")
    return parser.parse_args()

args = parse_args()
# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# Determine if multi-GPU
USE_MULTI_GPU = "," in args.gpu
NUM_GPUS = len(args.gpu.split(",")) if USE_MULTI_GPU else 1

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel # Keep PeftModel import for compatibility, though not directly used in this version for initial loading
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TrainerCallback
import sys

# Custom callback to log training metrics with flush
class DetailedLoggingCallback(TrainerCallback):
    """Log detailed training metrics and force flush to file"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens"""
        if logs is not None:
            # Format loss info
            step = state.global_step
            epoch = logs.get('epoch', 0)
            loss = logs.get('loss', None)
            lr = logs.get('learning_rate', None)
            
            log_msg = f"Step {step}"
            if epoch:
                log_msg += f" | Epoch {epoch:.2f}"
            if loss is not None:
                log_msg += f" | Loss: {loss:.4f}"
            if lr is not None:
                log_msg += f" | LR: {lr:.2e}"
            
            # Log with INFO level so it appears in file
            logger.info(log_msg)
            
            # Force flush all handlers
            for handler in logger.handlers:
                handler.flush()
            sys.stdout.flush()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch completion"""
        logger.info(f"✅ Epoch {state.epoch:.0f} completed | Global step: {state.global_step}")
        for handler in logger.handlers:
            handler.flush()
    
    def on_save(self, args, state, control, **kwargs):
        """Log checkpoint saves"""
        logger.info(f"💾 Checkpoint saved at step {state.global_step}")
        for handler in logger.handlers:
            handler.flush()


# --- Setup Logging ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def setup_logging(model_name="", suffix=""):
    """Create unique log file with timestamp and model name"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract short model name (e.g., gemma-2-2b, phi-3-mini)
    model_short = model_name.split('/')[-1].replace('-it', '').replace('-instruct', '') if model_name else "model"
    log_file = f"SFT_{model_short}_{timestamp}{suffix}.log"
    # Force unbuffered mode for real-time logging
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # Set to INFO to capture training metrics
    file_handler.setFormatter(log_formatter)
    logger.info(f"📝 Logging to: {log_file}")
    return file_handler

file_handler = None
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

def require_env_token(var: str = "HF_TOKEN") -> None:
    if not os.environ.get(var):
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                token = f.read().strip()
                os.environ[var] = token
                logger.info(f"✅ Loaded HF_TOKEN from cache")
                return
        raise SystemExit(f"Environment variable {var} not set and no cached token found.")

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
    )

def main() -> None:
    global file_handler
    
    require_env_token("HF_TOKEN")

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    
    # Setup logging with model name and timestamp
    log_suffix = f"_gpu{args.gpu}{args.log_suffix}" if args.log_suffix else f"_gpu{args.gpu}"
    file_handler = setup_logging(model_name, log_suffix)
    logger.addHandler(file_handler)
    auth_env = cfg.get("use_auth_token_env", "HF_TOKEN")
    auth_token = os.environ.get(auth_env)

    bnb_cfg = build_bnb_config(cfg)

    if USE_MULTI_GPU:
        logger.info(f"🎯 Using Multi-GPU: {NUM_GPUS} GPUs ({args.gpu}) for {model_name}")
        device_map = "auto"  # Let accelerate handle distribution
    else:
        torch.cuda.set_device(0)
        logger.info(f"🎯 Using Single GPU {args.gpu} (visible as GPU 0) for {model_name}")
        device_map = {"": 0}
    
    logger.info("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
    tok.padding_side = cfg.get("padding_side", "right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    logger.info("Loading 4-bit model…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        device_map=device_map,  # Use auto for multi-GPU, {"": 0} for single
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    logger.info("Preparing k-bit training and applying DoRA adapters…")
    model = prepare_model_for_kbit_training(model)
    
    # CHECK FOR EXISTING ADAPTER TO RESUME/CONTINUE TRAINING
    if "adapter_path" in cfg and cfg["adapter_path"] and os.path.exists(cfg["adapter_path"]):
        logger.info(f"🔄 Resuming training from existing adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"], is_trainable=True)
    else:
        logger.info("🆕 Creating new LoRA adapter...")
        lora_cfg = build_lora_config(cfg)
        model = get_peft_model(model, lora_cfg)

    # Load datasets
    train_path = cfg["train_path"]
    eval_path = cfg.get("eval_path")
    ds = load_dataset("json", data_files={"train": train_path})
    if eval_path and os.path.exists(eval_path):
        ds["validation"] = load_dataset("json", data_files={"validation": eval_path})["validation"]

    out_dir = cfg.get("output_dir", f"experiments/red_{int(time.time())}")

    # Pre-process dataset to create a single 'text' column
    def format_example(example):
        if "messages" in example:
            # Handle chat format directly
            text = tok.apply_chat_template(example["messages"], tokenize=False)
        elif hasattr(tok, "chat_template") and tok.chat_template and "instruction" in example:
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["payload"]},
            ]
            # Apply chat template but don't tokenize yet
            text = tok.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback for simple instruction/payload without chat template
            text = f"User: {example.get('instruction', '')}\nAssistant: {example.get('payload', '')}{tok.eos_token}"
        return {"text": text}

    logger.info(" formatting dataset to 'text' column...")
    # We map the formatting function to the dataset
    # Support both 'messages' and 'instruction'/'payload' formats
    if "messages" in ds["train"].column_names or ("instruction" in ds["train"].column_names and "payload" in ds["train"].column_names):
        ds = ds.map(format_example)
    
    # SFTConfig setup
    sft_config = SFTConfig(
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
        dataloader_pin_memory=bool(cfg.get("dataloader_pin_memory", True)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 4)),
        dataloader_prefetch_factor=int(cfg.get("dataloader_prefetch_factor", 2)),
        ddp_find_unused_parameters=False if not USE_MULTI_GPU else None,  # Auto for DDP
        max_length=int(cfg.get("max_length", 2048)),
        packing=False,
        group_by_length=bool(cfg.get("group_by_length", True)),
        dataset_text_field="text",
    )
    
    # Multi-GPU specific settings
    if USE_MULTI_GPU:
        logger.info(f"⚡ Enabling DataParallel training on {NUM_GPUS} GPUs")
        sft_config.ddp_backend = "nccl"  # Best for multi-GPU
        sft_config.local_rank = int(os.environ.get("LOCAL_RANK", -1))  # For torchrun

    if "max_steps" in cfg and int(cfg["max_steps"]) > 0:
        sft_config.max_steps = int(cfg["max_steps"])

    logger.info("Initializing SFTTrainer…")
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        callbacks=[DetailedLoggingCallback()],  # Add custom logging callback
    )

    logger.info("Starting training… (ctrl+c to stop)")
    logger.info(f"📊 Training config: {len(ds['train'])} samples × {sft_config.num_train_epochs} epochs")
    effective_batch = sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps * NUM_GPUS
    logger.info(f"📊 Effective batch size: {effective_batch} (per_device={sft_config.per_device_train_batch_size} × grad_accum={sft_config.gradient_accumulation_steps} × {NUM_GPUS} GPUs)")
    logger.info(f"📊 Total steps: ~{len(ds['train']) // effective_batch * sft_config.num_train_epochs}")
    
    for handler in logger.handlers:
        handler.flush()
    
    trainer.train()
    
    logger.info("Saving model…")
    trainer.save_model(out_dir)
    logger.info(f"✅ Training complete! Model saved to {out_dir}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
