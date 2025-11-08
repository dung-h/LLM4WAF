import argparse
import os
import time
from typing import Dict, Any, List

import yaml
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def require_env_token(var: str = "HF_TOKEN") -> None:
    if not os.environ.get(var):
        raise SystemExit(f"Environment variable {var} not set. Aborting for safety.")


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


def format_example(example: Dict[str, Any]) -> str:
    # Input: payload + context/evidence → Output: patch (CRS rule/regex/normalizer) + rationale
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()
    rationale = example.get("rationale", "").strip()
    return f"Input:\n{inp}\n\nPatch:\n{out}\n\nRationale:\n{rationale}"


def formatting_func(examples):
    res = []
    for i in range(len(examples["input"])):
        ex = {k: examples[k][i] for k in examples}
        res.append(format_example(ex))
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    require_env_token("HF_TOKEN")
    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    token = os.environ.get(cfg.get("use_auth_token_env", "HF_TOKEN"))

    bnb_cfg = build_bnb_config(cfg)

    tok = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    tok.padding_side = cfg.get("padding_side", "left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=token,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = get_peft_model(model, build_lora_config(cfg))

    data_files = {"train": cfg["train_path"]}
    if os.path.exists(cfg.get("eval_path", "")):
        data_files["validation"] = cfg["eval_path"]
    ds = load_dataset("json", data_files=data_files)

    out_dir = cfg.get("output_dir", f"experiments/blue_{int(time.time())}")
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(cfg.get("num_train_epochs", 2)),
        learning_rate=float(cfg.get("learning_rate", 1.5e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        fp16=bool(cfg.get("fp16", True)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        eval_steps=int(cfg.get("eval_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=targs,
        train_dataset=ds.get("train"),
        eval_dataset=ds.get("validation"),
        max_seq_length=int(cfg.get("seq_length", 2048)),
        packing=False,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.model.save_pretrained(os.path.join(out_dir, "adapter"))
    print("Blue adapter saved.")


if __name__ == "__main__":
    main()

