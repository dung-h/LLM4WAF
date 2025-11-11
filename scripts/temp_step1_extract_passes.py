import json
from pathlib import Path

# Use relative paths, which will work correctly inside WSL
# after the 'cd' command.
root = Path('.')
add_file = root / "data" / "processed" / "red_train_v6_loop_additions.jsonl"

# Ensure the destination directory exists
add_file.parent.mkdir(parents=True, exist_ok=True)

def extract(p):
  out = []
  if not p.exists():
    print(f"File not found, skipping: {p}")
    return out
  try:
    for line in p.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      d = json.loads(line)
      if d.get("blocked") is False:
        out.append(d.get("payload", ""))
  except Exception as e:
    print(f"Error processing file {p}: {e}")
  return out

passed = []
files_to_process = [
  root / "results" / "sqli_batch_encoded_replay.jsonl",
  root / "results" / "sqli_batch_manual_replay.jsonl",
]

print(f"Processing files: {files_to_process}")

for f in files_to_process:
  passed.extend(extract(f))

seen = set()
# Pre-read existing entries to avoid duplicates if script is run multiple times
if add_file.exists():
    with add_file.open("r", encoding="utf-8") as r:
        for line in r:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                seen.add((d.get("attack_type"), d.get("payload","" ).strip()))
            except json.JSONDecodeError:
                continue # Ignore malformed lines

appended_count = 0
with add_file.open("a", encoding="utf-8") as w:
  for p in passed:
    p = p.strip()
    if not p or ("SQLi", p) in seen:
      continue
    seen.add(("SQLi", p))
    row = {
      "instruction": "Generate an SQL injection payload to test DVWA SQLi endpoint.",
      "context": "Target: /vulnerabilities/sqli/ param 'id'. Environment: ModSecurity CRS PL1.",
      "constraints": "Keep real SQL syntax; avoid placeholders; concise boolean/union variants.",
      "payload": p,
      "reasoning": "from replay pass",
      "attack_type": "SQLi"
    }
    w.write(json.dumps(row, ensure_ascii=False) + "\n")
    appended_count += 1

print(f"Appended {appended_count} new unique SQLi passes to {add_file}")
print(f"Total payloads extracted from source files: {len(passed)}")