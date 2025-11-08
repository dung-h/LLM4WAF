import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.retrievers import TFIDFRetriever
from rag.utils import canonicalize, sim_ratio, simple_mutations
from rag.red_prompt import build_prompt


def pick_examples(retr_res: List[Dict], k: int = 5) -> List[Dict]:
    ex = []
    for r in retr_res[:k]:
        ex.append({
            "payload": r.get("meta", {}).get("payload", ""),
            "reason": r.get("meta", {}).get("reason", ""),
            "source": r.get("meta", {}).get("source", ""),
        })
    return ex


def fallback_generate(retr_res: List[Dict], n: int = 3, sim_threshold: float = 0.92) -> List[Dict]:
    outs = []
    seen: List[str] = []
    for r in retr_res:
        base_any = r.get("meta", {}).get("payload") or r.get("id") or "' OR 1=1 --"
        base = str(base_any)
        candidates = simple_mutations(base)
        random.shuffle(candidates)
        for c in candidates:
            if all(sim_ratio(c, s) < sim_threshold for s in seen):
                seen.append(c)
                outs.append({
                    "payload": c,
                    "reasoning": "Mutated from retrieved exemplar to evade simple pattern and transformations.",
                    "source": r.get("id"),
                })
                if len(outs) >= n:
                    return outs
    # If not enough, fall back to variations of a simple payload
    base = "' OR 1=1 --"
    while len(outs) < n:
        v = random.choice(simple_mutations(base))
        if all(sim_ratio(v, s) < sim_threshold for s in seen):
            seen.append(v)
            outs.append({"payload": v, "reasoning": "Heuristic variant for novelty.", "source": "fallback"})
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to payload TF-IDF index")
    parser.add_argument("--instruction", default="Generate SQLi payload for testing")
    parser.add_argument("--context", default="DB=MySQL; sink=query; method=GET; CRS=PL2")
    parser.add_argument("--constraints", default="lab-only; avoid destructive ops")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n", type=int, default=3, help="Number of outputs (fallback mode only)")
    parser.add_argument("--use_model", action="store_true", help="Attempt to use local LLM (4-bit)")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--adapter", default="", help="Path to PEFT adapter (optional)")
    # New gen controls
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate when using model")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens per sample when using model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--attn_impl", type=str, default="", help="Attention impl (e.g., flash_attention_2)")
    args = parser.parse_args()

    retr = TFIDFRetriever.load(args.index)
    # For retrieval, we use the context as a simple query
    res = retr.query(args.context, top_k=args.top_k)

    # Prioritize training data (has both payload AND reasoning) over seed data (payload only)
    # Sort results: training data first, then seed data
    def has_reasoning(r):
        return bool(r.get("meta", {}).get("reason", "").strip())
    
    res_with_reason = [r for r in res if has_reasoning(r)]
    res_no_reason = [r for r in res if not has_reasoning(r)]
    res = res_with_reason + res_no_reason  # Prioritize items with reasoning

    examples = pick_examples(res, k=min(5, len(res)))
    prompt = build_prompt(args.instruction, args.context, args.constraints, examples)

    if not args.use_model:
        outs = fallback_generate(res, n=args.n)
        for o in outs:
            print(json.dumps({
                "instruction": args.instruction,
                "context": args.context,
                "constraints": args.constraints,
                "payload": o["payload"],
                "reasoning": o["reasoning"],
                "source": o.get("source", "retrieval"),
            }, ensure_ascii=False))
        return

    # Optional model path (heavy) — guarded
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
    except Exception as e:
        print("Model libraries not available; falling back to heuristic generation.")
        outs = fallback_generate(res, n=args.n)
        for o in outs:
            print(json.dumps({
                "instruction": args.instruction,
                "context": args.context,
                "constraints": args.constraints,
                "payload": o["payload"],
                "reasoning": o["reasoning"],
                "source": o.get("source", "retrieval"),
            }, ensure_ascii=False))
        return

    token = os.environ.get("HF_TOKEN")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=token)
    # Right padding is generally safer for half precision training/generation
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=token, device_map="auto", quantization_config=bnb, torch_dtype=torch.float16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    # Optional perf tweaks
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # speed on Ampere+
    except Exception:
        pass
    if args.attn_impl:
        try:
            model.config.attn_implementation = args.attn_impl
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        for _ in range(max(1, int(args.samples))):
            # Apply chat template if available (for chat models like TinyLlama)
            if hasattr(tok, 'apply_chat_template') and 'chat' in args.model_name.lower():
                messages = [
                    {"role": "system", "content": "You are a cybersecurity expert specializing in SQL injection testing for educational purposes."},
                    {"role": "user", "content": prompt + "\n\nGenerate only the payload and reasoning, nothing else."}
                ]
                try:
                    formatted_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    formatted_prompt = prompt  # Fallback to raw prompt
            else:
                formatted_prompt = prompt
            
            enc = tok(formatted_prompt, return_tensors="pt").to(model.device)
            gen = model.generate(
                **enc,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            out = tok.decode(gen[0], skip_special_tokens=True)
            
            # DEBUG: Print raw output
            import sys
            print(f"\n[DEBUG] Raw model output:\n{out}\n", file=sys.stderr)
            
            # Try to extract JSON from output
            payload = ""
            reasoning = ""
            
            # Extract only the response part (after formatted_prompt)
            response_text = out[len(formatted_prompt):].strip() if len(out) > len(formatted_prompt) else out
            
            print(f"[DEBUG] Response after prompt removal:\n{response_text}\n", file=sys.stderr)
            
            # Try to parse as JSON first
            try:
                # Look for JSON object in response
                for line in response_text.splitlines():
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        obj = json.loads(line)
                        payload = obj.get("payload", "")
                        reasoning = obj.get("reasoning", "")
                        break
            except json.JSONDecodeError:
                pass
            
            # Fallback: parse key-value format
            if not payload:
                for line in response_text.splitlines():
                    l = line.strip()
                    if l.lower().startswith("payload:") and not payload:
                        payload = l.split(":", 1)[1].strip().strip('"').strip("'")
                    if l.lower().startswith("reasoning:") and not reasoning:
                        reasoning = l.split(":", 1)[1].strip().strip('"').strip("'")
            
            print(json.dumps({
                "instruction": args.instruction,
                "context": args.context,
                "constraints": args.constraints,
                "payload": payload,
                "reasoning": reasoning,
                "source": "model",
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
