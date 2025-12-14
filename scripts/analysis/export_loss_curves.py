import argparse
from pathlib import Path
from typing import List, Dict

from tensorboard.backend.event_processing import event_accumulator

PRIORITY_TAGS = ["train/loss", "loss", "train_loss", "eval/loss"]


def pick_tag(ea: event_accumulator.EventAccumulator) -> str:
    tags = ea.Tags().get("scalars", [])
    for t in PRIORITY_TAGS:
        if t in tags:
            return t
    return tags[0] if tags else None


def load_scalars(path: Path, tag: str) -> List[Dict]:
    ea = event_accumulator.EventAccumulator(str(path))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    scalars = ea.Scalars(tag)
    return [
        {
            "wall_time": s.wall_time,
            "step": s.step,
            "value": s.value,
            "tag": tag,
            "source_file": path.name,
        }
        for s in scalars
    ]


def export_run(name: str, event_files: List[Path], out_dir: Path, summary: List[str]):
    rows: List[Dict] = []
    tag_used = None
    for ef in sorted(event_files):
        ea = event_accumulator.EventAccumulator(str(ef))
        ea.Reload()
        tag = pick_tag(ea)
        if not tag:
            continue
        if tag_used is None:
            tag_used = tag
        rows.extend(load_scalars(ef, tag))
    if not rows:
        summary.append(f"- {name}: no scalar found")
        return
    rows.sort(key=lambda r: (r["step"], r["wall_time"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{name}.csv"
    out_csv.write_text(
        "step,value,wall_time,tag,source_file\n"
        + "\n".join(
            f"{r['step']},{r['value']},{r['wall_time']},{r['tag']},{r['source_file']}" for r in rows
        ),
        encoding="utf-8",
    )
    summary.append(f"- {name}: {len(rows)} points, tag='{tag_used}' -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/loss_curves", help="Output directory for CSVs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    runs = {
        "Gemma_2B_Phase1": list(Path("experiments/remote_gemma2_2b_phase1/logs").glob("events.out.tfevents.*")),
        "Gemma_2B_Phase2": list(Path("experiments/remote_gemma2_2b_phase2/logs").glob("events.out.tfevents.*")),
        "Phi3_Mini_Phase1": list(Path("experiments/remote_phi3_mini_phase1/logs").glob("events.out.tfevents.*")),
        "Phi3_Mini_Phase2": list(Path("experiments/remote_phi3_mini_phase2/logs").glob("events.out.tfevents.*")),
        "Qwen_3B_Phase1": list(Path("experiments/remote_qwen_3b_phase1/logs").glob("events.out.tfevents.*")),
        "Qwen_3B_Phase2": list(Path("experiments/remote_qwen_3b_phase2/logs").glob("events.out.tfevents.*")),
    }

    summary_lines: List[str] = []
    for name, files in runs.items():
        if not files:
            summary_lines.append(f"- {name}: no event files")
            continue
        export_run(name, files, out_dir, summary_lines)

    summary_path = Path(out_dir) / "SUMMARY.md"
    summary_path.write_text("# Loss curves export\n" + "\n".join(summary_lines), encoding="utf-8")
    print(f"Written summary to {summary_path}")


if __name__ == "__main__":
    main()
