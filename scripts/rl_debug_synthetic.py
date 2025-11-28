import os
import json
import argparse
from typing import Dict, Any, List

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from pathlib import Path
from util.debug_instrumentation import (
    text_hash, preview, measure_time, last_elapsed_ms, grad_global_norm, cuda_mem_stats
)


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
        target_modules=l.get('target_modules', [
            "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
        ]),
        lora_dropout=float(l.get('dropout', 0.05)),
        bias='none',
        task_type='CAUSAL_LM',
        use_dora=bool(l.get('use_dora', False)),
    )


def extract_json_fields(text: str) -> Dict[str, Any] | None:
    """Attempt to parse a JSON object from the text and return it.
    This is deliberately generic and safe (no domain-specific content).
    """
    import re

    s = text.strip()
    # Try to find first JSON object in the text
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        return None


def synthetic_reward(obj: Dict[str, Any] | None) -> float:
    """Safe, content-agnostic reward:
    - If valid JSON and has keys {"answer", "reason"}:
        * +1.0 if answer matches a simple target style (contains a digit and parentheses)
        * +0.4 if JSON is valid but style is partial (has a digit only)
        * 0.0 otherwise
    - If not valid JSON: -0.5
    """
    if obj is None:
        return -0.5
    ans = str(obj.get("answer", ""))
    has_digit = any(c.isdigit() for c in ans)
    has_paren = ("(" in ans and ")" in ans)
    if ("answer" in obj) and ("reason" in obj):
        if has_digit and has_paren:
            return 1.0
        if has_digit:
            return 0.4
        return 0.0
    return -0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=False, default='configs/rl_debug_synthetic.yaml')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg.get('model_name', 'microsoft/Phi-3-mini-4k-instruct')
    out_dir = cfg.get('output_dir', 'experiments/rl_debug_synthetic')
    os.makedirs(out_dir, exist_ok=True)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    # Model + LoRA (4-bit)
    bnb = build_bnb(cfg)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    base = prepare_model_for_kbit_training(base)
    peft_cfg = build_lora(cfg)
    model = get_peft_model(base, peft_cfg)
    init_adapter = cfg.get('init_adapter_path') or ''
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    model.train()

    # Optimizer
    lr = float(cfg.get('train', {}).get('lr', 5e-6))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # Prompt template (safe)
    base_intro = (
        "You are a helpful assistant.\n"
        "Output a single-line JSON object with keys 'answer' and 'reason'.\n"
        "Example style only (not to be copied): {\"answer\": \"(A) 42\", \"reason\": \"brief\"}.\n"
        "Respond ONLY with JSON.\n"
    )

    steps = int(cfg.get('train', {}).get('steps', 20))
    candidates = int(cfg.get('train', {}).get('candidates', 2))
    max_new_tokens = int(cfg.get('train', {}).get('max_new_tokens', 64))
    clip_grad = float(cfg.get('train', {}).get('clip_grad_norm', 1.0))
    log_every = int(cfg.get('train', {}).get('log_every', 5))

    samples_log = os.path.join(out_dir, 'debug_samples.jsonl')
    seen_texts: Dict[str, int] = {}

    for step in range(1, steps + 1):
        # Encode prompt
        prompt_text = base_intro
        enc = tok(prompt_text, return_tensors='pt').to(model.device)

        # Generate candidates
        with torch.no_grad():
            with measure_time():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    num_return_sequences=candidates,
                    pad_token_id=tok.eos_token_id,
                )
            gen_ms = last_elapsed_ms(0.0) or 0.0
        texts: List[str] = tok.batch_decode(out, skip_special_tokens=True)

        losses: List[torch.Tensor] = []
        rewards: List[float] = []

        for idx, t in enumerate(texts):
            obj = extract_json_fields(t)
            rwd = synthetic_reward(obj)
            simplified = t.strip()
            prev = seen_texts.get(simplified, 0)
            seen_texts[simplified] = prev + 1
            duplicate_penalty = 0.25 * prev
            rwd = max(-1.0, rwd - duplicate_penalty)

            # Compute CE on generated portion only
            full_ids = tok(t, return_tensors='pt').input_ids.to(model.device)
            prompt_len = enc.input_ids.size(1)
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100
            out_logits = model(input_ids=full_ids, labels=labels)
            ce = out_logits.loss
            loss = torch.tensor(rwd, device=model.device) * ce
            losses.append(loss)
            rewards.append(float(rwd))

            # Debug print per candidate
            print(f"[DBG] step={step} cand={idx} prompt_hash={text_hash(prompt_text)} gen_ms={gen_ms:.1f}")
            print(f"  text: {preview(t, 160)}")
            reason_preview = ''
            if obj and isinstance(obj.get('reason'), str):
                reason_preview = preview(obj.get('reason'), 200)
            print(f"  parsed: {obj is not None} reward={rwd:.3f} ce={ce.item():.4f}")
            print(f"  reason: {reason_preview or '<none>'}")

            # Persist sample
            sample = {
                'step': step,
                'candidate': idx,
                'prompt_hash': text_hash(prompt_text),
                'text_preview': preview(t, 200),
                'parsed': obj is not None,
                'reward': float(rwd),
                'ce_loss': float(ce.item()),
                'gen_ms': float(gen_ms),
                **cuda_mem_stats(),
            }
            with open(samples_log, 'a', encoding='utf-8') as sf:
                sf.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Backprop
        total_loss = sum(losses) / max(1, len(losses))
        with measure_time():
            total_loss.backward()
        bwd_ms = last_elapsed_ms(0.0) or 0.0

        gnorm = grad_global_norm(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        with measure_time():
            optim.step()
        opt_ms = last_elapsed_ms(0.0) or 0.0
        optim.zero_grad()

        if step % log_every == 0:
            avg_r = sum(rewards) / max(1, len(rewards))
            print(
                f"[RL-DEBUG] step={step} loss={total_loss.item():.4f} avg_reward={avg_r:.3f} "
                f"gen_ms={gen_ms:.1f} bwd_ms={bwd_ms:.1f} opt_ms={opt_ms:.1f} gnorm={gnorm:.3f} "
                f"mem={cuda_mem_stats()}"
            )

    # Save adapter
    save_path = os.path.join(out_dir, 'adapter')
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"Saved adapter to {save_path}")


if __name__ == '__main__':
    main()
