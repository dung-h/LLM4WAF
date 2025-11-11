#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs" / "loop"


def run(cmd: List[str], cwd: Path | None = None) -> Tuple[int, str]:
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    return p.returncode, out


def generate(adapter_dir: Path, limit: int) -> Tuple[Path, Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(ROOT / "scripts" / "infer" / "v6_langchain_generate.py"),
        "--limit",
        str(limit),
        "--adapter-dir",
        str(adapter_dir),
    ]
    code, out = run(cmd, cwd=ROOT)
    if code != 0:
        raise SystemExit(f"Generation failed:\n{out}")
    sqli = RESULTS / "v6_sqli_langchain.txt"
    xss = RESULTS / "v6_xss_langchain.txt"
    return sqli, xss


def harness(file: Path, base_url: str, param: str, out_name: str) -> Tuple[Path, List[str]]:
    out_jsonl = RESULTS / out_name
    harness_py = ROOT / ("replay/harness.py" if "sqli" in out_name else "replay/harness_xss.py")
    cmd = [
        "python",
        str(harness_py),
        str(file),
        "--output",
        str(out_jsonl),
        "--base-url",
        base_url,
        "--param-name",
        param,
    ]
    code, out = run(cmd, cwd=ROOT)
    if code != 0:
        raise SystemExit(f"Harness failed:\n{out}")

    passed: List[str] = []
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not row.get("blocked", True):
                p = str(row.get("payload", "")).strip()
                if p:
                    passed.append(p)
    return out_jsonl, passed


def build_rows(payloads: List[str], attack_type: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in payloads:
        if not p:
            continue
        if attack_type == "SQLi":
            instr = "Generate an extremely difficult-to-bypass WAF SQL injection payload for DVWA SQLi endpoint."
            ctx = "Target: /vulnerabilities/sqli/ param 'id'. Environment: ModSecurity CRS PL1."
            cons = "Keep real SQL syntax; avoid placeholders; concise boolean/union variants. Focus on advanced obfuscation, encoding, and evasion techniques. Think about the techniques and steps to apply to enhance the payload for WAF bypass."
        else:
            instr = "Generate an extremely difficult-to-bypass WAF XSS payload for DVWA reflected XSS endpoint."
            ctx = "Target: /vulnerabilities/xss_r/ param 'name'. Environment: ModSecurity CRS PL1."
            cons = "Prefer advanced obfuscation (e.g., HTML entities, Unicode, bracket access, atob, srcdoc, polyglot); short (<120 chars); avoid simple tags like <script> or <onerror>. Think about the techniques and steps to apply to enhance the payload for WAF bypass."
        rows.append({
            "instruction": instr,
            "context": ctx,
            "constraints": cons,
            "payload": p,
            "reasoning": "<start-thought>from replay pass</end-thought>",
            "attack_type": attack_type,
        })
    return rows


def append_unique(rows: List[Dict[str, str]], store: Path) -> int:
    seen = set()
    if store.exists():
        with open(store, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                seen.add((d.get("attack_type", ""), d.get("payload", "")))
    added = 0
    with open(store, "a", encoding="utf-8") as f:
        for r in rows:
            key = (r.get("attack_type", ""), r.get("payload", ""))
            if key in seen:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            seen.add(key)
            added += 1
    return added


def make_train_yaml(template_cfg: Path, train_path: Path, out_dir: Path, out_yaml: Path) -> None:
    with open(template_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["train_path"] = str(train_path)
    cfg["output_dir"] = str(out_dir)
    # keep other fields
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def combine_train(base_path: Path, loop_store: Path, out_path: Path, max_loop: int | None = None) -> None:
    base_lines: List[str] = []
    if base_path.exists():
        base_lines = [l for l in base_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    loop_lines: List[str] = []
    if loop_store.exists():
        loop_lines = [l for l in loop_store.read_text(encoding="utf-8").splitlines() if l.strip()]
    if max_loop is not None:
        loop_lines = loop_lines[-max_loop:]
    all_lines = base_lines + loop_lines
    random.Random(42).shuffle(all_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay-in-the-loop trainer")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--limit", type=int, default=200, help="Total samples to generate per cycle (split half/half)")
    ap.add_argument("--base-train", type=str, default=str(PROCESSED / "red_train_v6_small.jsonl"))
    ap.add_argument("--adapter-dir", type=str, default=str(ROOT / "experiments" / "red_gemma2_v6_small_quick" / "adapter"))
    ap.add_argument("--template-cfg", type=str, default=str(ROOT / "configs" / "red_v6_small_quick.yaml"))
    ap.add_argument("--max-loop-append", type=int, default=500)
    ap.add_argument("--sqli-url", type=str, default="http://localhost:8080/vulnerabilities/sqli/")
    ap.add_argument("--sqli-param", type=str, default="id")
    ap.add_argument("--xss-url", type=str, default="http://localhost:8080/vulnerabilities/xss_r/")
    ap.add_argument("--xss-param", type=str, default="name")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    loop_store = PROCESSED / "red_train_v6_loop_additions.jsonl"
    current_adapter = Path(args.adapter_dir)

    for cycle in range(1, args.cycles + 1):
        print(f"=== CYCLE {cycle}/{args.cycles} ===")
        # 1) Generate
        sqli_file, xss_file = generate(current_adapter, args.limit)
        print(f"Generated: {sqli_file.name}, {xss_file.name}")

        # 2) Test
        sqli_jsonl, sqli_pass = harness(sqli_file, args.sqli_url, args.sqli_param, "v6_sqli_loop_replay.jsonl")
        xss_jsonl, xss_pass = harness(xss_file, args.xss_url, args.xss_param, "v6_xss_loop_replay.jsonl")
        print(f"Pass: SQLi={len(sqli_pass)} XSS={len(xss_pass)}")

        # 3) Append to loop dataset (unique)
        rows = build_rows(sqli_pass, "SQLi") + build_rows(xss_pass, "XSS")
        added = append_unique(rows, loop_store)
        print(f"Added {added} new rows to {loop_store}")

        # 4) Build new train file = base + loop additions (capped)
        train_out = PROCESSED / f"red_train_v6_small_loop_c{cycle}.jsonl"
        combine_train(Path(args.base_train), loop_store, train_out, max_loop=args.max_loop_append)
        print(f"Train file: {train_out}")

        # 5) Make a per-cycle yaml and train
        out_dir = ROOT / "experiments" / f"red_gemma2_v6_loop_c{cycle}"
        yaml_out = ROOT / "configs" / f"red_v6_loop_c{cycle}.yaml"
        make_train_yaml(Path(args.template_cfg), train_out, out_dir, yaml_out)
        code, out = run(["python", str(ROOT / "scripts" / "train_red.py"), "--config", str(yaml_out)], cwd=ROOT)
        (LOGS / f"train_c{cycle}.log").write_text(out, encoding="utf-8")
        if code != 0:
            raise SystemExit(f"Training failed in cycle {cycle}. See logs: {LOGS / f'train_c{cycle}.log'}")

        # 6) Update adapter path
        current_adapter = out_dir / "adapter"
        print(f"Cycle {cycle} complete. New adapter: {current_adapter}")

    print("All cycles completed.")


if __name__ == "__main__":
    main()
