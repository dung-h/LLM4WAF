import csv
import yaml
from pathlib import Path


def load_csv(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["total"] = int(r["total"])
            r["passed"] = int(r["passed"])
            r["blocked"] = int(r["blocked"])
            r["invalid"] = int(r["invalid"])
            r["pass_rate_pct"] = float(r["pass_rate_pct"])
            r["invalid_rate_pct"] = float(r["invalid_rate_pct"])
            rows.append(r)
    return rows


def stage_from_model(model: str) -> str:
    return model.split("_")[-1]


def find_row(rows, model, waf, dataset_phase):
    for r in rows:
        if (
            r["model"] == model
            and r["waf_level"] == waf
            and r["dataset_phase"] == dataset_phase
        ):
            return r
    return None


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def sft_row(model, phase_label, cfg):
    per_batch = cfg.get("per_device_train_batch_size")
    grad_acc = cfg.get("gradient_accumulation_steps")
    eff_batch = per_batch * grad_acc if per_batch is not None and grad_acc is not None else None
    return {
        "model": model,
        "phase": phase_label,
        "lr": cfg.get("learning_rate"),
        "per_device_batch": per_batch,
        "grad_accum": grad_acc,
        "effective_batch": eff_batch,
        "epochs": cfg.get("num_train_epochs"),
        "max_seq_len": cfg.get("max_seq_length") or cfg.get("max_context_length"),
        "lora_r": cfg.get("lora_r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "quant_4bit": cfg.get("load_in_4bit"),
    }


def rl_row(model, cfg):
    return {
        "model": model,
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epochs"),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "context_len": cfg.get("max_context_length"),
        "adapter_from": cfg.get("adapter_path"),
    }


def main():
    modsec_rows = load_csv(Path("reports/eval_modsec_pass_rates.csv"))
    coraza_rows = load_csv(Path("reports/eval_coraza_pass_rates.csv"))

    phase_map = {"Pretrained": "phase1", "Phase1": "phase1", "Phase2": "phase2", "RL": "phase3"}

    modsec_wafs = ["PL1", "PL4"]
    model_order = [
        "Gemma_2B_Phase1",
        "Gemma_2B_Phase2",
        "Gemma_2B_RL",
        "Qwen_3B_Phase1",
        "Qwen_3B_Phase2",
        "Qwen_3B_RL",
        "Phi3_Mini_Phase1",
        "Phi3_Mini_Phase2",
        "Phi3_Mini_RL",
        "Gemma_2B_Pretrained",
        "Qwen_3B_Pretrained",
        "Phi3_Mini_Pretrained",
    ]

    modsec_lines = [
        "| Model | PL1 Pass % | PL1 Invalid % | PL4 Pass % | PL4 Invalid % |",
        "|---|---|---|---|---|",
    ]
    for model in model_order:
        ds = phase_map.get(stage_from_model(model), "phase1")
        row_parts = [model]
        for waf in modsec_wafs:
            r = find_row(modsec_rows, model, waf, ds)
            if r:
                row_parts.append(f"{r['pass_rate_pct']:.1f}")
                row_parts.append(f"{r['invalid_rate_pct']:.1f}")
            else:
                row_parts.extend(["", ""])
        modsec_lines.append("| " + " | ".join(row_parts) + " |")

    coraza_lines = ["| Model | CORAZA Pass % | CORAZA Invalid % |", "|---|---|---|"]
    for model in model_order:
        ds = phase_map.get(stage_from_model(model), "phase1")
        r = find_row(coraza_rows, model, "CORAZA", ds)
        if r:
            coraza_lines.append(f"| {model} | {r['pass_rate_pct']:.1f} | {r['invalid_rate_pct']:.1f} |")

    sft_configs = {
        "Gemma_2B_Phase1": ("Phase 1", Path("configs/remote_gemma2_2b_phase1.yaml")),
        "Gemma_2B_Phase2": ("Phase 2", Path("configs/remote_gemma2_2b_phase2.yaml")),
        "Phi3_Mini_Phase1": ("Phase 1", Path("configs/remote_phi3_mini_phase1.yaml")),
        "Phi3_Mini_Phase2": ("Phase 2", Path("configs/remote_phi3_mini_phase2.yaml")),
        "Qwen_3B_Phase1": ("Phase 1", Path("configs/remote_qwen_3b_phase1.yaml")),
        "Qwen_3B_Phase2": ("Phase 2", Path("configs/remote_qwen_3b_phase2.yaml")),
    }
    rl_configs = {
        "Gemma_2B_RL": Path("configs/remote_gemma2_2b_phase3_rl.yaml"),
        "Phi3_Mini_RL": Path("configs/remote_phi3_mini_phase3_rl.yaml"),
        "Qwen_3B_RL": Path("configs/remote_qwen_3b_phase3_rl.yaml"),
    }

    sft_rows = [sft_row(m, p, load_yaml(path)) for m, (p, path) in sft_configs.items()]
    rl_rows = [rl_row(m, load_yaml(path)) for m, path in rl_configs.items()]

    sft_lines = [
        "| Model | Phase | LR | Per-device batch | Grad accum | Effective batch | Epochs | Max seq len | LoRA r | LoRA alpha | 4-bit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sft_rows:
        sft_lines.append(
            "| {model} | {phase} | {lr} | {per_device_batch} | {grad_accum} | {effective_batch} | {epochs} | {max_seq_len} | {lora_r} | {lora_alpha} | {quant_4bit} |".format(
                **{k: (v if v is not None else "") for k, v in r.items()}
            )
        )

    rl_lines = ["| Model | LR | Batch | Epochs | Max new tokens | Context len | Adapter from |", "|---|---|---|---|---|---|---|"]
    for r in rl_rows:
        rl_lines.append(
            "| {model} | {lr} | {batch_size} | {epochs} | {max_new_tokens} | {context_len} | {adapter_from} |".format(
                **{k: (v if v is not None else "") for k, v in r.items()}
            )
        )

    out_path = Path("reports/training_eval_tables.md")
    chunks = [
        "# Section 4 Tables (auto-generated)",
        "",
        "## Hyperparameters - SFT (Phase 1 & Phase 2)",
        *sft_lines,
        "",
        "## Hyperparameters - RL (Phase 3)",
        *rl_lines,
        "",
        "## Pass Rate - ModSecurity (matched dataset per model)",
        *modsec_lines,
        "",
        "## Pass Rate - Coraza (matched dataset per model)",
        *coraza_lines,
        "",
        "## Raw CSV references",
        "- reports/eval_modsec_pass_rates.csv",
        "- reports/eval_coraza_pass_rates.csv",
    ]
    out_path.write_text("\n".join(chunks), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
