import json
from pathlib import Path
from typing import Iterable, Dict, List


def collect(path: Path, waf_label: str, targets: Iterable[str], samples: List[Dict]):
    data = json.loads(path.read_bytes().decode("utf-8"))
    for entry in data:
        model = entry["model"]
        if model not in targets:
            continue
        for phase_key in ["phase1", "phase2", "phase3"]:
            recs = entry.get(phase_key, [])
            by_type = {"SQLI": None, "XSS": None}
            for r in recs:
                if r.get("test_result", {}).get("status") == "passed":
                    atype = r.get("attack_type")
                    if atype in by_type and by_type[atype] is None:
                        by_type[atype] = r
                if all(by_type.values()):
                    break
            for atype, val in by_type.items():
                if val:
                    samples.append(
                        {
                            "model": model,
                            "phase": phase_key,
                            "waf": waf_label,
                            "attack_type": atype,
                            "technique": val.get("technique"),
                            "payload": val.get("payload"),
                            "reason": val.get("test_result", {}).get("reason"),
                        }
                    )


def main():
    modsec_path = Path("eval/rl_validation_20251210_094937_modsec/all_results.json")
    coraza_path = Path("eval/rl_validation_20251212_004519_coraza/all_results.json")

    targets = {
        "Gemma_2B_Phase2",
        "Gemma_2B_RL",
        "Qwen_3B_RL",
        "Phi3_Mini_RL",
    }

    samples: List[Dict] = []
    collect(modsec_path, "ModSecurity", targets, samples)
    collect(coraza_path, "Coraza", targets, samples)

    out_lines = []
    for s in samples:
        out_lines.append(
            f"### {s['model']} ({s['phase']}) - {s['waf']} - {s['attack_type']}\n"
            f"- Technique: {s['technique']}\n"
            f"- Reason: {s['reason']}\n"
            f"- Payload:\n```\n{s['payload']}\n```\n"
        )
    Path("reports/payload_samples.md").write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote reports/payload_samples.md with {len(samples)} samples")


if __name__ == "__main__":
    main()
