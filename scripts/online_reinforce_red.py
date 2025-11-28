import os
import re
import json
import time
import argparse
from typing import Dict, Any, List
from pathlib import Path
import sys
import logging
from collections import Counter

# Add project root to sys.path for module imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import yaml
import torch
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteriaList, StopStringCriteria
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from rag.retrievers import TFIDFRetriever

# --- Setup Logging ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set to INFO by default, use DEBUG for more verbosity

# File handler
file_handler = logging.FileHandler("RL.log", mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


SQL_ERROR_PATTERNS = [
    r"you have an error in your sql syntax",
    r"warning: mysql_",
    r"sqlstate{",
    r"uncle(d|s) quotation mark",
    r"incorrect syntax near",
    r"fatal error: uncaught pdoexception",
    r"sqlite error",
    r"syntax error at or near",
]


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


def ensure_login(client: httpx.Client, login_url: str, username: str, password: str) -> bool:
    logger.info(f"Attempting to log in to DVWA at {login_url}...")
    try:
        r = client.get(login_url)
        r.raise_for_status()
        m = re.search(r"user_token' value='([a-f0-9]{32})'", r.text, re.I)
        if not m:
            logger.error("Failed to find user_token on login page. Is DVWA running?")
            return False
        token = m.group(1)
        data = {"username": username, "password": password, "user_token": token, "Login": "Login"}
        r = client.post(login_url, data=data)
        r.raise_for_status()
        if 'login.php' in str(r.url):
            logger.error("DVWA login failed. Check credentials.")
            return False
        logger.info("DVWA login successful.")
        return True
    except Exception as e:
        logger.error(f"An error occurred during DVWA login: {e}")
        return False


def eval_payload(client: httpx.Client, sqli_url: str, payload: str):
    """Return (label, status_code, resp_len). Label in {blocked, passed, sql_error_bypass}.
    Also detect DVWA data leak markers as bypass (not only SQL errors).
    This version uses a non-redirecting client to correctly handle failures.
    """
    try:
        # Use a temporary client that does NOT follow redirects for evaluation
        with httpx.Client(follow_redirects=False, timeout=15.0, cookies=client.cookies) as eval_client:
            r = eval_client.get(sqli_url, params={"id": payload})
        
        logger.debug(f"Payload: '{payload[:50]}...', Status: {r.status_code}, Response Length: {len(r.text)}")

        if r.status_code == 403:
            return 'blocked', r.status_code, 0
        
        # If we get redirected (e.g., to login.php), it's a failure.
        if r.status_code in (301, 302, 307, 308):
            return 'redirect_failed', r.status_code, 0

        if r.status_code == 200:
            txt = r.text.lower()
            dvwa_terms = ['first name', 'surname', 'user id', 'first name:', 'surname:']
            if any(term in txt for term in dvwa_terms):
                # Check for SQL errors even in successful data leak pages
                for pat in SQL_ERROR_PATTERNS:
                    if re.search(pat, txt):
                        return 'sql_error_bypass', r.status_code, len(r.text)
                return 'passed', r.status_code, len(r.text) # It's a pass if it shows data without SQL errors
            
            # If it's a 200 OK but doesn't contain DVWA terms, it might be a different page.
            # We still check for SQL errors as a primary bypass indicator.
            for pat in SQL_ERROR_PATTERNS:
                if re.search(pat, txt):
                    return 'sql_error_bypass', r.status_code, len(r.text)
            
            # If no data and no error, it's a neutral/failed attempt.
            return 'passed_but_no_data', r.status_code, len(r.text)

        # Handle other unexpected status codes
        return f'unexpected_status_{r.status_code}', r.status_code, len(r.text)

    except httpx.RequestError as e:
        logger.warning(f"Request error for payload: '{payload[:50]}...': {e}")
        return 'request_error', None, 0


def extract_payload(text: str) -> str:
    """
    Cleans the generated text to extract only the payload.
    Removes common model artifacts and descriptive text.
    """
    # Remove any leading/trailing whitespace and special tokens
    cleaned_text = text.strip()
    # Remove any trailing <|end|> or <|endoftext|> tokens that might be generated
    if cleaned_text.endswith("<|end|>"):
        cleaned_text = cleaned_text[:-len("<|end|>")].strip()
    if cleaned_text.endswith("<|endoftext|>"):
        cleaned_text = cleaned_text[:-len("<|endoftext|>")].strip()
    
    # Remove any leading "Assistant:" or similar conversational markers if they somehow appear
    if cleaned_text.lower().startswith("assistant:"):
        cleaned_text = cleaned_text[len("assistant:"):].strip()

    # Remove markdown code blocks if present
    if cleaned_text.startswith("```") and cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[3:-3].strip()
        # Remove language specifier if present (e.g., ```sql)
        if cleaned_text.startswith("sql\n"):
            cleaned_text = cleaned_text[4:].strip()
        elif cleaned_text.startswith("html\n"):
            cleaned_text = cleaned_text[5:].strip()

    # Fallback: take the first non-empty line if multiple lines are generated
    for line in cleaned_text.splitlines():
        s = line.strip()
        if s:
            return s
    return ''


def load_retriever():
    """Lazy-load TFIDFRetriever if available by adding project root to sys.path."""
    try:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from rag.retrievers import TFIDFRetriever  # type: ignore
        idx_path = root / 'rag' / 'indexes' / 'payloads_tfidf.joblib'
        if idx_path.exists():
            logger.info(f"Loading TFIDFRetriever from {idx_path}")
            return TFIDFRetriever.load(str(idx_path))
    except Exception as e:
        logger.warning(f"Could not load TFIDFRetriever: {e}")
    return None


# Known invalid markers that are not real SQLi payloads
INVALID_MARKERS = {"sqlvuln", "example_payload", "testpayload", "payload", "user"}


SQL_KEYWORDS = [
    "select",
    "union",
    "insert",
    "update",
    "delete",
    "drop",
    "sleep",
    "benchmark",
    "waitfor",
    "where",
    "from",
    "having",
    "group by",
    "order by",
]


def is_repetitive_gibberish(payload: str) -> bool:
    """
    Heuristic detector for collapse-style outputs like 'Gener Gener Gener ...'.
    Treat as invalid if:
      - The text has no obvious SQL keyword, and
      - A single alphabetic token dominates the sequence.
    """
    text = payload.strip().lower()
    if not text:
        return False

    # If there is any SQL keyword, we do NOT treat it as gibberish here.
    if any(kw in text for kw in SQL_KEYWORDS):
        return False

    tokens = re.findall(r"[a-zA-Z]+", text)
    if not tokens:
        return False

    counts = Counter(tokens)
    total = len(tokens)
    most_common_token, freq = counts.most_common(1)[0]

    # If one token accounts for >=70% of the words and
    # we have enough words, it's almost certainly collapse.
    if total >= 10 and freq / total >= 0.7:
        return True
    return False


def is_invalid_marker(payload: str) -> bool:
    # First, detect repetitive gibberish patterns (mode collapse artefacts)
    if is_repetitive_gibberish(payload):
        return True

    s = payload.strip().strip('"\'').lower()
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s in INVALID_MARKERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--hf_token', type=str, default=None, help='Hugging Face API token for private models.')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg.get('model_name')
    out_dir = cfg.get('output_dir', f"experiments/online_reinforce_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- 1.5 Load RAG Retriever ---
    retriever = None
    retriever_path = Path(__file__).resolve().parents[1] / "rag" / "indexes" / "payloads_tfidf.joblib"
    if cfg.get('use_rag', False):
        if retriever_path.exists():
            try:
                logger.info(f"Loading TF-IDF retriever from {retriever_path}...")
                retriever = TFIDFRetriever.load(str(retriever_path))
                logger.info("Retriever loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load retriever: {e}. RAG will be disabled.")
        else:
            logger.warning(f"Retriever index not found at {retriever_path}. RAG will be disabled.")

    # Tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    logger.info("Tokenizer loaded.")

    # Model + LoRA in 4-bit
    bnb = build_bnb(cfg)
    logger.info(f"Loading base model: {model_name} with 4-bit quantization.")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=bnb,
        torch_dtype=torch.float16,
        token=args.hf_token,
    )
    logger.info("Base model loaded.")

    # Prepare model for kbit training
    base_model = prepare_model_for_kbit_training(base_model)
    
    # Load SFT adapter directly onto the base model
    init_adapter = cfg.get('init_adapter_path') or ''
    if init_adapter:
        logger.info(f"Loading initial SFT adapter from: {init_adapter}")
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
        logger.info("Initial SFT adapter loaded.")
    else:
        logger.info("No initial SFT adapter path provided. Initializing new LoRA adapter.")
        peft_cfg = build_lora(cfg)
        model = get_peft_model(base_model, peft_cfg)
    
    # Wrap model with ValueHead for PPO
    model = AutoModelForCausalLMWithValueHead(model)
    # HACK: AutoModelForCausalLMWithValueHead does not delegate all attributes, so we manually set them
    model.is_peft_model = True # We know it's a peft model
    model.active_peft_config = model.pretrained_model.active_peft_config
    logger.info("Model wrapped with AutoModelForCausalLMWithValueHead.")

    # DVWA session
    d = cfg.get('dvwa', {})
    client = httpx.Client(follow_redirects=True, timeout=15.0)
    if not ensure_login(client, d.get('login_url'), d.get('username', 'admin'), d.get('password', 'password')):
        logger.critical("Failed to log in to DVWA. Exiting.")
        raise SystemExit('DVWA login failed. Open /setup.php and initialize tables, then retry.')
    logger.info("DVWA login confirmed.")

    # PPO Config
    train_cfg = cfg.get('train', {})
    ppo_cfg = PPOConfig(
        steps=train_cfg.get('steps', 200),
        learning_rate=float(train_cfg.get('lr', 5e-6)),
        # Enable KL regularization to keep the policy close to the SFT model
        adap_kl_ctrl=bool(train_cfg.get('adap_kl_ctrl', True)),
        init_kl_coef=float(train_cfg.get('init_kl_coef', 0.02)),
        target=float(train_cfg.get('target_kl', 0.1)),
        batch_size=train_cfg.get('candidates', 4),  # Batch size is number of candidates
        mini_batch_size=train_cfg.get('candidates', 4),  # Set mini_batch_size equal to batch_size
        gradient_accumulation_steps=train_cfg.get('grad_accum', 1),
        max_grad_norm=float(train_cfg.get('clip_grad_norm', 1.0)),
        whiten_rewards=bool(train_cfg.get('whiten_rewards', False)),
        # Make PPO very conservative around the SFT policy
        ppo_epochs=int(train_cfg.get('ppo_epochs', 1)),
        cliprange=float(train_cfg.get('cliprange', 0.1)),
        score_clip=float(train_cfg.get('score_clip', 2.0)),
        log_with=None,  # No external logger for now
    )
    logger.info(f"PPOConfig initialized with steps={ppo_cfg.steps}, learning_rate={ppo_cfg.learning_rate}, batch_size={ppo_cfg.batch_size}")

    # PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_cfg,
        model=model,
        ref_model=None, # No reference model for now
        tokenizer=tok,
    )
    logger.info("PPOTrainer initialized.")

    # --- PPO Training Loop ---
    invalid_streak = 0
    samples_log = os.path.join(out_dir, 'online_samples.jsonl')
    logger.info(f"Generated samples will be logged to: {samples_log}")
    generated_payload_history = set() # To store unique payloads generated so far

    for step in range(ppo_cfg.steps):
        logger.info(f"--- RL Step {step+1}/{ppo_cfg.steps} ---")

        # --- Generate Prompts ---
        prompts_text = []
        for _ in range(ppo_cfg.batch_size): # Generate prompts for each candidate in the batch
            instruction = "Generate a single, effective MySQL SQL injection payload. Output only the payload itself, with no other text or explanation."
            
            # RAG integration
            if retriever and cfg.get('use_rag', False):
                try:
                    rag_query = "sql injection"
                    examples = retriever.query(rag_query, top_k=cfg.get('rag_top_k', 3), filters={"category": "SQLi"})
                    if examples:
                        example_payloads = [ex['meta']['payload'] for ex in examples]
                        example_str = "\n".join([f"- `{p}`" for p in example_payloads])
                        instruction = (
                            "Here are some examples of effective SQL injection payloads:\n"
                            f"{example_str}\n\n"
                            "Now, based on these examples, generate a new, single, effective MySQL SQL injection payload. "
                            "Output only the payload itself, with no other text or explanation."
                        )
                        logger.debug("Using RAG-augmented instruction.")
                    else:
                        logger.warning("RAG returned no results. Falling back to base instruction.")
                except Exception as e:
                    logger.error(f"Error during RAG retrieval: {e}. Falling back to base instruction.")
            
            # Apply Phi-3 chat template
            messages = [{"role": "user", "content": f"{instruction}\n"}]
            formatted_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts_text.append(formatted_prompt)
            logger.debug(f"Generated prompt: {formatted_prompt}")

        # Encode prompts
        device = ppo_trainer.model.parameters().__next__().device
        query_tensors = [tok(p, return_tensors='pt').input_ids.squeeze(0).to(device) for p in prompts_text]
        
        # --- Generate Responses ---
        # Define stop sequences
        stop_strings = cfg.get('generation', {}).get('stop_strings', ["<|end|>", "\n```", "\n<|user|>"])
        stop_criteria = StoppingCriteriaList([StopStringCriteria(tok, stop_strings)])

        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=cfg.get('train', {}).get('max_new_tokens', 70),
            do_sample=True,
            temperature=cfg.get('temperature', 1.3),
            top_p=cfg.get('top_p', 0.9),
            top_k=cfg.get('top_k', 50),
            pad_token_id=tok.eos_token_id,
            stopping_criteria=stop_criteria, # Apply stop criteria
        )
        
        # Decode responses
        # We need to decode the full response and then extract the generated part
        responses = []
        for i, response_tensor in enumerate(response_tensors):
            full_decoded_output = tok.decode(response_tensor, skip_special_tokens=False)
            # Find the start of the assistant's response
            assistant_start_tag = "<|assistant|>"
            assistant_start_index = full_decoded_output.rfind(assistant_start_tag) # Use rfind for robustness

            if assistant_start_index != -1:
                generated_part = full_decoded_output[assistant_start_index + len(assistant_start_tag):].strip()
                responses.append(generated_part)
            else:
                logger.warning(f"Could not find assistant tag in generated output for prompt {i}. Full output: {full_decoded_output}")
                responses.append("") # Append empty string if tag not found
        
        # --- Calculate Rewards ---
        def sql_heuristic_bonus(payload: str) -> float:
            """
            Lightweight shaping reward to prefer SQL-looking payloads
            even when DVWA blocks them (403). This helps prevent the
            policy from drifting to pure gibberish strings.
            """
            text = payload.lower()
            bonus = 0.0

            # Core SQL keywords
            hits = 0
            for kw in ("select", "union", "insert", "update", "delete", "drop", "sleep", "benchmark", "waitfor"):
                if kw in text:
                    hits += 1
            if hits > 0:
                bonus += 0.2 + 0.05 * min(hits - 1, 3)  # up to +0.35

            # Typical SQLi syntax features
            if "'" in payload or '"' in payload:
                bonus += 0.05
            if "--" in payload or "#" in payload:
                bonus += 0.05
            if " or " in text or " and " in text:
                bonus += 0.05

            return min(bonus, 0.5)

        rewards = []
        step_labels: List[str] = []
        for idx, response in enumerate(responses):
            payload = extract_payload(response)
            rwd = 0.0 # Default reward

            # Novelty reward: penalize repeated payloads (exact repeats)
            novelty_penalty = 0.0
            is_repeated = False
            if payload in generated_payload_history:
                is_repeated = True
                novelty_penalty = -0.1 # Small penalty for repetition
                logger.debug(f"Candidate {idx+1}: Payload '{payload[:50]}...' is a repeat. Applying novelty penalty: {novelty_penalty:.2f}")
            else:
                generated_payload_history.add(payload)

            if not payload or is_invalid_marker(payload):
                label = 'invalid_marker' if payload else 'empty'
                rwd = -1.5 # Heavily penalize invalid / collapse-style payloads
                sc, rlen = None, None
                invalid_streak += 1
                logger.info(f"Candidate {idx+1} (Invalid): Payload='{payload[:50]}...', Label='{label}', Reward={rwd:.2f}, Invalid Streak={invalid_streak}")
            else:
                logger.info(f"Candidate {idx+1} (Raw): {payload}")
                label, sc, rlen = eval_payload(client, d.get('sqli_url'), payload)
                if label == 'blocked':
                    rwd = -0.2 # Slight penalty for being blocked by the WAF.
                elif label in ('passed', 'sql_error_bypass'):
                    rwd = 1.0 # Reward successful bypasses.
                elif label == 'redirect_failed':
                    rwd = -1.0 # Strongly penalize payloads that cause redirects (e.g., garbage input).
                else: # Covers 'passed_but_no_data', 'request_error', and other unexpected cases.
                    rwd = -0.5 # Penalize other non-successful outcomes.

                # Add a heuristic bonus for SQL-looking payloads so the
                # policy prefers them over non-SQL gibberish.
                bonus = sql_heuristic_bonus(payload)
                if bonus > 0.0:
                    rwd += bonus
                    logger.debug(f"Candidate {idx+1}: Applied SQL heuristic bonus {bonus:.2f}, raw reward now {rwd:.2f}")

                # Reset invalid streak if the payload was at least valid enough to be tested.
                if label not in ('invalid_marker', 'empty'):
                    invalid_streak = 0

                logger.info(f"Candidate {idx+1} (Valid): Payload='{payload[:50]}...', Label='{label}', Reward={rwd:.2f}, Status Code={sc}, Response Length={rlen}")

            step_labels.append(label)

            # Apply novelty penalty and cumulative penalty, then clip
            rwd = rwd + novelty_penalty
            cum_pen = float(cfg.get('train', {}).get('invalid_cum_penalty', 0.1))
            rwd = max(-2.0, min(1.0, rwd - cum_pen * invalid_streak))
            rewards.append(rwd)
            logger.debug(f"Candidate {idx+1}: Final reward after novelty, cumulative penalty and clipping: {rwd:.2f}")

            sample = {
                'step': step, 'candidate': idx, 'payload': payload, 'label': label,
                'reward': float(rwd), 'status_code': sc, 'resp_len': rlen,
            }
            with open(samples_log, 'a', encoding='utf-8') as sf:
                sf.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.debug(f"Candidate {idx+1} sample logged to {samples_log}")

        # --- PPO Step ---
        # If all candidates are invalid/empty (e.g. collapse-style gibberish),
        # skip the PPO update for this step to avoid reinforcing bad regions.
        if not any(lab not in ('invalid_marker', 'empty') for lab in step_labels):
            logger.warning("All candidates this step are invalid/empty; skipping PPO update to avoid reinforcing collapse.")
            continue

        rewards_tensors = [torch.tensor([r], device=device) for r in rewards] # Wrap each reward in a 1D tensor for ppo_trainer.step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensors)
        # ppo_trainer.log_stats(stats, query_tensors, response_tensors, rewards) # Temporarily commented out due to TypeError
        
        # Remove the noisy objective/kl_dist from stats before logging
        if 'objective/kl_dist' in stats:
            del stats['objective/kl_dist']

        logger.info(f"PPO Step {step+1} completed. Stats: {stats}")

        if (step + 1) % cfg.get('train', {}).get('log_every', 5) == 0:
            avg_r = sum(rewards) / max(1, len(rewards))
            logger.info(f"[ONLINE-REINFORCE] step={step+1} avg_reward={avg_r:.3f}")
            try:
                # Log a sample payload for quick check
                valid_payloads = [p for p, r in zip(responses, rewards) if extract_payload(p) and not is_invalid_marker(extract_payload(p))]
                if valid_payloads:
                    p_info = extract_payload(valid_payloads[0])
                    logger.info(f"  Sample Payload from this step: '{p_info[:100]}...'\n")
            except Exception as e:
                logger.warning(f"Error logging sample payload: {e}")

    # Save adapter
    save_path = os.path.join(out_dir, 'adapter')
    os.makedirs(save_path, exist_ok=True)
    ppo_trainer.model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    logger.info(f"Saved final adapter to {save_path}")
    logger.info("RL Training complete.")


if __name__ == '__main__':
    main()
