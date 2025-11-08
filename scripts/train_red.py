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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments


def require_env_token(var: str = "HF_TOKEN") -> None:
    """Check for HF token, use cached token if available"""
    if not os.environ.get(var):
        # Try to load from cache
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        print(f"[DEBUG] Checking token path: {token_path}")
        print(f"[DEBUG] Path exists: {os.path.exists(token_path)}")
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                token = f.read().strip()
                os.environ[var] = token
                print(f"✅ Loaded HF_TOKEN from cache (len={len(token)})")
                return
        print(f"[DEBUG] Token not found, aborting")
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
        use_dora=bool(cfg.get("use_dora", True)),
    )


def format_example(example: Dict[str, Any], fields: Dict[str, str]) -> str:
    """
    Support 3 formats:
    1. Full: instruction, context, constraints, payload, reasoning
    2. v5_simple: instruction, payload, reasoning (no context/constraints)
    3. Simple: payload, reasoning only
    """
    payload = example.get(fields["payload"], "")
    reasoning = example.get(fields["reasoning"], "")
    
    # Safe strip (handle None)
    payload = payload.strip() if payload else ""
    reasoning = reasoning.strip() if reasoning else ""
    
    # Check if full format (has instruction field)
    if "instruction" in fields and fields["instruction"] in example:
        instr = example.get(fields["instruction"], "")
        instr = instr.strip() if instr else ""
        
        # Check if we have context and constraints fields
        if "context" in fields and "constraints" in fields:
            # Full format with all 5 fields
            ctx = example.get(fields["context"], "")
            cons = example.get(fields["constraints"], "")
            
            ctx = ctx.strip() if ctx else ""
            cons = cons.strip() if cons else ""
            
            user = f"Instruction: {instr}\nContext: {ctx}\nConstraints: {cons}\n".strip()
        else:
            # v5_simple format - instruction only (no context/constraints)
            user = f"Instruction: {instr}"
        
        assistant = f"Payload: {payload}\nReasoning: {reasoning}".strip()
        return user + "\n\n" + assistant
    else:
        # Simple format (v2 dataset)
        return f"Payload: {payload}\nReasoning: {reasoning}".strip()


def formatting_func(examples: Dict[str, List[str]], fields) -> List[str]:
    """
    fields can be:
    - list: [payload, reasoning] → assume default mapping
    - dict: {instruction, context, constraints, payload, reasoning}
    """
    # If fields is a list, create default mapping
    if isinstance(fields, list):
        # Default: first field = payload, second field = reasoning
        fields_dict = {
            "instruction": "instruction",
            "context": "context", 
            "constraints": "constraints",
            "payload": fields[0] if len(fields) > 0 else "payload",
            "reasoning": fields[1] if len(fields) > 1 else "reasoning"
        }
    else:
        fields_dict = fields
    
    out = []
    # Use payload field as length reference (all fields should have same length)
    ref_field = fields_dict["payload"]
    for i in range(len(examples[ref_field])):
        ex = {k: examples[k][i] for k in examples}
        out.append(format_example(ex, fields_dict))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    require_env_token("HF_TOKEN")

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    auth_env = cfg.get("use_auth_token_env", "HF_TOKEN")
    auth_token = os.environ.get(auth_env)

    bnb_cfg = build_bnb_config(cfg)

    print("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    tok.padding_side = cfg.get("padding_side", "left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading 4-bit model… (this may take a while)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=auth_token,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print("Preparing k-bit training and applying DoRA adapters…")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)

    # Load datasets
    train_path = cfg["train_path"]
    eval_path = cfg.get("eval_path")
    text_fields = cfg["text_fields"]

    data_files = {"train": train_path}
    if eval_path and os.path.exists(eval_path):
        data_files["validation"] = eval_path

    ds = load_dataset("json", data_files=data_files)

    def _fmt_func(batch):
        return {"text": formatting_func(batch, text_fields)}

    # Fix: text_fields can be list or dict
    if isinstance(text_fields, dict):
        cols = list(text_fields.values())
    else:
        cols = text_fields if isinstance(text_fields, list) else [text_fields]
    
    ds_proc = {}
    for split in ds:
        ds_proc[split] = ds[split].map(_fmt_func, batched=True, remove_columns=[c for c in ds[split].column_names if c not in cols])

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
        report_to=[],
    )

    # Optional max_steps override for smoke tests
    if "max_steps" in cfg and int(cfg["max_steps"]) > 0:
        targs.max_steps = int(cfg["max_steps"])  # type: ignore

    print("Initializing SFTTrainer…")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=targs,
        train_dataset=ds_proc.get("train"),
        eval_dataset=ds_proc.get("validation"),
        max_seq_length=int(cfg.get("seq_length", 2048)),
        packing=False,
        dataset_text_field="text",
    )

    print("Starting training… (ctrl+c to stop)")
    trainer.train()
    print("Saving adapter…")
    trainer.model.save_pretrained(os.path.join(out_dir, "adapter"))
    print("Done.")


if __name__ == "__main__":
    main()
