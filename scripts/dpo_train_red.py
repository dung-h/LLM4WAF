import os
import argparse
from typing import Dict, Any

import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import TrainingArguments


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_bnb(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    b = cfg.get('bnb', {})
    return BitsAndBytesConfig(
        load_in_4bit=bool(b.get('load_in_4bit', True)),
        bnb_4bit_quant_type=b.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_use_double_quant=bool(b.get('bnb_4bit_use_double_quant', True)),
        bnb_4bit_compute_dtype=getattr(torch, b.get('bnb_4bit_compute_dtype', 'float16')),
    )


def build_lora(cfg: Dict[str, Any]) -> LoraConfig:
    l = cfg.get('lora', {})
    return LoraConfig(
        r=int(l.get('r', 16)),
        lora_alpha=int(l.get('alpha', 32)),
        target_modules=l.get('target_modules', ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        lora_dropout=float(l.get('dropout', 0.05)),
        bias='none',
        task_type='CAUSAL_LM',
        use_dora=bool(l.get('use_dora', False)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to DPO yaml config')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg.get('model_name')
    out_dir = cfg.get('output_dir', 'experiments/dpo_phi3_lora')
    pairs_path = cfg.get('pairs_path')

    if not pairs_path or not os.path.exists(pairs_path):
        raise SystemExit(f"DPO pairs file not found: {pairs_path}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    # Model (policy) with value head + LoRA
    bnb = build_bnb(cfg)
    peft_cfg = build_lora(cfg)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=bnb,
        torch_dtype=torch.float16,
        peft_config=peft_cfg,
    )
    if hasattr(model, 'gradient_checkpointing_enable') and bool(cfg.get('train', {}).get('gradient_checkpointing', False)):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

    # Dataset: expects columns 'prompt', 'chosen', 'rejected'
    ds = load_dataset('json', data_files={'train': pairs_path})

    # Training arguments
    t = cfg.get('train', {})
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(t.get('per_device_train_batch_size', 1)),
        gradient_accumulation_steps=int(t.get('gradient_accumulation_steps', 8)),
        num_train_epochs=float(t.get('num_train_epochs', 1)),
        learning_rate=float(t.get('learning_rate', 5e-6)),
        lr_scheduler_type=t.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=float(t.get('warmup_ratio', 0.03)),
        weight_decay=float(t.get('weight_decay', 0.0)),
        logging_steps=int(t.get('logging_steps', 10)),
        save_steps=int(t.get('save_steps', 200)),
        save_total_limit=int(t.get('save_total_limit', 2)),
        optim=t.get('optim', 'paged_adamw_8bit'),
        fp16=bool(t.get('fp16', True)),
        report_to=[],
    )

    # DPO config
    d = cfg.get('dpo', {})

    trainer = DPOTrainer(
        model,
        ref_model=None,  # default: copy of model without LoRA grads; TRL handles it
        args=targs,
        beta=float(d.get('beta', 0.1)),
        train_dataset=ds['train'],
        tokenizer=tok,
        max_length=int(d.get('max_length', 512)),
        max_prompt_length=int(d.get('max_prompt_length', 256)),
        formatting_func=None,
    )

    print('[DPO] Starting training (ctrl+c to stop)')
    trainer.train()
    print('[DPO] Saving adapter...')
    trainer.model.save_pretrained(os.path.join(out_dir, 'adapter'))
    tok.save_pretrained(os.path.join(out_dir, 'adapter'))
    print('[DPO] Done.')


if __name__ == '__main__':
    main()

