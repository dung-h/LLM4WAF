import os
import re
import time
import json
import argparse
from typing import List, Dict, Any

import yaml
import torch
import httpx
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, set_seed


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


def build_simple_prompt(instruction: str) -> str:
    """Builds a simple user-assistant prompt matching the SFT format."""
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


def extract_payload(text: str) -> str:
    # Look for `Payload: <...>`
    m = re.search(r"Payload:\s*(.*)", text, re.DOTALL) # Use re.DOTALL to match across newlines
    if m:
        extracted = m.group(1).strip()
        # Basic sanity check: ensure it's not just repeating the prompt, empty, or the placeholder
        if extracted and extracted != "<payload>" and not extracted.startswith("You are a helpful AI assistant."):
            # Take only the first line if there are multiple lines after "Payload:"
            return extracted.splitlines()[0].strip()
    return ""


class HeuristicRewarder:
    def __init__(self, cfg: Dict[str, Any]):
        h = cfg.get('reward', {}).get('heuristic', {})
        self.base = float(h.get('base', 0.1))
        self.obf_bonus = float(h.get('obfuscation_bonus', 0.2))
        self.concat_bonus = float(h.get('concat_bonus', 0.1))
        self.time_bonus = float(h.get('time_based_bonus', 0.2))
        self.len_penalty = float(h.get('length_penalty', 0.15))

    def score_payload(self, payload: str) -> float:
        if not payload:
            return -0.5
        score = self.base
        low = payload.lower()
        # mild obfuscation indicators
        if ('/**/' in payload) or ('/*!' in payload) or ('0x' in low) or ('char(' in low) or ('%2f' in low) or ('%2a' in low):
            score += self.obf_bonus
        # concatenation / union-like indicators
        if ('concat' in low) or ('union' in low):
            score += self.concat_bonus
        # time-based indicators
        if ('sleep(' in low) or ('benchmark(' in low):
            score += self.time_bonus
        # length penalty
        if len(payload) > 120:
            score -= self.len_penalty
        # clamp
        return float(max(0.0, min(1.0, score)))


class OnlineWAFRewarder:
    def __init__(self, cfg: Dict[str, Any]):
        r = cfg.get('reward', {})
        self.weights = r.get('weights', {"blocked": 0.0, "passed": 0.4, "sql_error_bypass": 1.0})
        d = cfg.get('dvwa', {})
        self.login_url = d.get('login_url')
        self.sqli_url = d.get('sqli_url')
        self.username = d.get('username', 'admin')
        self.password = d.get('password', 'password')

        # Persistent HTTP session
        self.client = httpx.Client(follow_redirects=True, timeout=15.0)
        self._logged_in = self._ensure_login()

    def _ensure_login(self) -> bool:
        try:
            r = self.client.get(self.login_url)
            r.raise_for_status()
            m = re.search(r"user_token' value='([a-f0-9]{32})'", r.text, re.I)
            if not m:
                return False
            token = m.group(1)
            data = {"username": self.username, "password": self.password, "user_token": token, "Login": "Login"}
            r = self.client.post(self.login_url, data=data)
            r.raise_for_status()
            if "login.php" in str(r.url):
                return False
            return True
        except Exception:
            return False

    def _test_payload(self, payload: str) -> str:
        try:
            if not self._logged_in and not self._ensure_login():
                return 'blocked'
            r = self.client.get(self.sqli_url, params={"id": payload})
            if r.status_code == 403:
                return 'blocked'
            if "login.php" in str(r.url):
                if not self._ensure_login():
                    return 'blocked'
                r = self.client.get(self.sqli_url, params={"id": payload})
                if r.status_code == 403:
                    return 'blocked'
            response_text = r.text.lower()
            sql_error_patterns = [
                r"you have an error in your sql syntax", r"warning: mysql_",
                r"sqlstate\{", r"uncle(d|s) quotation mark", r"incorrect syntax near",
                r"fatal error: uncaught pdoexception", r"sqlite error", r"syntax error at or near",
            ]
            for pat in sql_error_patterns:
                if re.search(pat, response_text):
                    return 'sql_error_bypass'
            return 'passed'
        except Exception:
            return 'blocked'

    def score_payload(self, payload: str) -> float:
        if not payload:
            return -0.5 # Heavier penalty for empty/invalid (e.g., '<payload>')
        res = self._test_payload(payload)
        return float(self.weights.get(res, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to rl yaml config')
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model_name = cfg.get('base_model_name')
    sft_adapter_path = cfg.get('sft_adapter_path') or ""
    out_dir = cfg.get('output_dir', f"experiments/rl_phi3_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    # PPO settings
    ppo_cfg = cfg.get('ppo', {})
    ppo_config = PPOConfig(
        learning_rate=float(ppo_cfg.get('learning_rate', 1.0e-5)),
        batch_size=int(ppo_cfg.get('batch_size', 8)),
        mini_batch_size=int(ppo_cfg.get('mini_batch_size', 2)),
        gradient_accumulation_steps=int(ppo_cfg.get('gradient_accumulation_steps', 4)),
        target_kl=None, # Disabled to prevent NaN issues
        ppo_epochs=int(ppo_cfg.get('ppo_epochs', 2)),
        seed=int(ppo_cfg.get('seed', 42)),
        log_with=None,
    )
    set_seed(ppo_config.seed)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    # Prepare base weights (optionally merge SFT LoRA) to a temp path
    bnb = build_bnb(cfg)
    model_load_path = base_model_name
    if sft_adapter_path:
        print(f"[RL] Merging SFT adapter from {sft_adapter_path} into base model...")
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='auto',
            quantization_config=bnb,
            torch_dtype=torch.float16,
        )
        peft_m = PeftModel.from_pretrained(base, sft_adapter_path)
        merged = peft_m.merge_and_unload()
        tmp_merge_path = os.path.join(out_dir, 'merged_sft_base')
        os.makedirs(tmp_merge_path, exist_ok=True)
        merged.save_pretrained(tmp_merge_path)
        tok.save_pretrained(tmp_merge_path)
        model_load_path = tmp_merge_path
        print("[RL] Merge complete.")

    # Create value-head model with LoRA (new adapter for RL)
    lora_cfg = build_lora(cfg)
    print(f"[RL] Loading value-head model from: {model_load_path}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_load_path,
        device_map='auto',
        quantization_config=bnb,
        torch_dtype=torch.float16,
        peft_config=lora_cfg,
    )
    gc_flag = bool(cfg.get('train', {}).get('gradient_checkpointing', False))
    if gc_flag and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
    else:
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True

    # Generation kwargs
    gen_cfg = cfg.get('gen', {})
    gen_kwargs = {
        'max_new_tokens': int(gen_cfg.get('max_new_tokens', 64)),
        'do_sample': bool(gen_cfg.get('do_sample', True)),
        'temperature': float(gen_cfg.get('temperature', 0.8)),
        'top_p': float(gen_cfg.get('top_p', 0.9)),
        'pad_token_id': tok.eos_token_id,
    }

    # Build query template from prompt config
    p = cfg.get('prompt', {})
    # Use a simpler, direct instruction for the prompt
    base_instruction = p.get('instruction', 'Generate one payload for a MySQL SQL injection attack.')
    query_template = build_simple_prompt(base_instruction)

    # Initialize PPO trainer (required before using ppo_trainer.*)
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tok)
    # Rewarder selection
    reward_mode = (cfg.get('reward', {}).get('mode') or 'heuristic').lower()
    if reward_mode == 'online_waf':
        rewarder = OnlineWAFRewarder(cfg)
        print("[RL] Using online WAF reward (DVWA must be running).")
    else:
        rewarder = HeuristicRewarder(cfg)
        print("[RL] Using heuristic reward (offline).")

    total_steps = int(cfg.get('train', {}).get('total_ppo_steps', 200))
    save_every = int(cfg.get('train', {}).get('save_every_steps', 50))
    eval_every = int(cfg.get('train', {}).get('eval_every_steps', 50))

    print(f"[RL] Starting PPO for {total_steps} steps...")
    for step in range(1, total_steps + 1):
        # Build a batch of identical prompts (cheap and stable) — can be replaced by varied prompts
        queries_str: List[str] = [query_template for _ in range(ppo_config.batch_size)]

        # Batched generation using tensors
        ppo_trainer.model.eval() # Set to eval mode for generation
        enc = tok(queries_str, return_tensors='pt', padding=True)
        query_tensors = [
            enc.input_ids[i].to(ppo_trainer.accelerator.device)
            for i in range(enc.input_ids.size(0))
        ]
        gen_out = ppo_trainer.generate(query_tensors, **gen_kwargs)
        response_tensors = []
        if isinstance(gen_out, list):
            for t in gen_out:
                response_tensors.append(t[0] if getattr(t, 'ndim', 1) == 2 else t)
        else:
            for i in range(gen_out.size(0)):
                response_tensors.append(gen_out[i])
        responses_str = [tok.decode(t, skip_special_tokens=True) for t in response_tensors]

        # Compute rewards on trainer device
        rewards: List[torch.Tensor] = []
        for i, txt in enumerate(responses_str):
            payload = extract_payload(txt)
            r = rewarder.score_payload(payload)
            rewards.append(torch.tensor(float(r)).to(ppo_trainer.accelerator.device))
            print(f"[RL_DEBUG] Step {step}, Sample {i}:")
            print(f"[RL_DEBUG]   Full Response: {txt}")
            print(f"[RL_DEBUG]   Extracted Payload: {payload}")
            print(f"[RL_DEBUG]   Assigned Reward: {r}")

        # PPO step with tensors
        ppo_trainer.model.train() # Set back to train mode for PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        if step % 5 == 0:
            avg_reward = sum(rewards) / max(1, len(rewards))
            print(f"[RL] step={step} avg_reward={avg_reward:.3f} kl={stats.get('ppo/kl', float('nan'))}")

        # Periodic save
        if step % save_every == 0 or step == total_steps:
            save_path = os.path.join(out_dir, f"step_{step}_adapter")
            os.makedirs(save_path, exist_ok=True)
            ppo_trainer.model.save_pretrained(save_path)
            tok.save_pretrained(save_path)
            with open(os.path.join(out_dir, 'last_step.json'), 'w', encoding='utf-8') as f:
                json.dump({'last_step': step}, f)

        # Optional: lightweight eval hook — reuse heuristic score on a few samples
        if step % eval_every == 0:
            with torch.no_grad():
                eval_query_str = query_template
                eval_query_tensor = tok(eval_query_str, return_tensors='pt').input_ids[0].to(ppo_trainer.accelerator.device)
                out = ppo_trainer.generate(eval_query_tensor, **gen_kwargs)
                text = tok.decode(out[0], skip_special_tokens=True)
                payload = extract_payload(text)
                score = rewarder.score_payload(payload)
                print(f"[RL][EVAL] step={step} payload={payload[:80]} score={score:.3f}")

    print("[RL] Training complete.")


if __name__ == '__main__':
    main()
