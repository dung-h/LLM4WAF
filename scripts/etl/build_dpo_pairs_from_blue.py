import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


PROMPT_TEMPLATE = (
    "You are an AI Red Team assistant operating in a lab.\n"
    "Generate a single MySQL SQL injection test payload and a concise reasoning.\n\n"
    "Respond exactly as:\nPayload: <payload>\nReasoning: <one or two sentences>\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to JSONL with fields: payload, waf_blocked or label')
    ap.add_argument('--out', required=True, help='Output DPO pairs JSONL path')
    ap.add_argument('--n_pairs', type=int, default=2000)
    args = ap.parse_args()

    good = []  # chosen candidates (bypass/passed)
    bad = []   # rejected candidates (blocked)

    for row in load_jsonl(Path(args.input)):
        payload = (row.get('payload') or '').strip()
        if not payload:
            continue
        # Heuristics from fields
        waf_blocked = row.get('waf_blocked')
        label = (row.get('label') or '').lower()
        if waf_blocked is None:
            # Fallback: label false_negative -> bypass (good); true_positive -> blocked (bad)
            if 'false_negative' in label:
                good.append(payload)
            elif 'true_positive' in label:
                bad.append(payload)
        else:
            if int(waf_blocked) == 0:
                good.append(payload)
            else:
                bad.append(payload)

    random.shuffle(good)
    random.shuffle(bad)
    n = min(args.n_pairs, len(good), len(bad))
    if n == 0:
        raise SystemExit('No pairs could be formed from the input. Check the fields.')

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        for i in range(n):
            g = good[i % len(good)]
            b = bad[i % len(bad)]
            # Build response strings in the same output schema
            chosen = f"Payload: {g}\nReasoning: Attempt with obfuscation/compactness."
            rejected = f"Payload: {b}\nReasoning: Likely blocked by WAF."
            ex = {
                'prompt': PROMPT_TEMPLATE,
                'chosen': chosen,
                'rejected': rejected,
            }
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f'Wrote {n} pairs to {outp}')


if __name__ == '__main__':
    main()

