import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
import argparse

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def analyze_distribution(input_file: str, output_dir: str, output_filename: str):
    attack_type_counts = Counter()
    technique_counts = Counter()
    attack_type_technique_counts = Counter()
    source_counts = Counter()
    
    total_samples = 0

    print(f"Analyzing distribution for {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                total_samples += 1

                attack_type = sample.get("attack_type", "UNKNOWN_ATTACK_TYPE")
                technique = sample.get("technique", "UNKNOWN_TECHNIQUE")
                source = sample.get("source", "UNKNOWN_SOURCE")

                attack_type_counts[attack_type] += 1
                technique_counts[technique] += 1
                attack_type_technique_counts[(attack_type, technique)] += 1
                source_counts[source] += 1

            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line: {e}")
            except Exception as e:
                print(f"An error occurred processing line: {e}")

    # Prepare results for output
    results = {
        "total_samples": total_samples,
        "attack_type_distribution": {k: v for k, v in attack_type_counts.most_common()},
        "technique_distribution": {k: v for k, v in technique_counts.most_common()},
        "attack_type_technique_distribution": {f"{k[0]} - {k[1]}": v for k, v in attack_type_technique_counts.most_common()},
        "source_distribution": {k: v for k, v in source_counts.most_common()},
        "minority_techniques": {}, # Techniques with count <= 20
        "majority_techniques": {}  # Techniques with count > 500
    }

    # Identify minority and majority techniques
    for tech, count in technique_counts.items():
        if count <= 20:
            results["minority_techniques"][tech] = count
        elif count > 500: # Example threshold for majority
            results["majority_techniques"][tech] = count

    # Print summary to stdout
    print(f"\n--- Distribution Analysis Summary for {os.path.basename(input_file)} ---")
    print(f"Total Samples: {total_samples}")

    print("\nAttack Type Distribution:")
    for atype, count in attack_type_counts.most_common():
        print(f"- {atype}: {count} ({count/total_samples:.2%})")

    print("\nTop 10 Techniques:")
    for (atype, tech), count in attack_type_technique_counts.most_common(10):
        print(f"- {atype} - {tech}: {count} ({count/total_samples:.2%})")

    print("\nMinority Techniques (<= 20 samples):")
    if not results["minority_techniques"]:
        print("- None found.")
    else:
        for tech, count in results["minority_techniques"].items():
            print(f"- {tech}: {count}")

    print("\nMajority Techniques (> 500 samples):")
    if not results["majority_techniques"]:
        print("- None found.")
    else:
        for tech, count in results["majority_techniques"].items():
            print(f"- {tech}: {count}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results to JSON file
    output_filepath = os.path.join(output_dir, output_filename)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed distribution report saved to {output_filepath}")
    
    return output_filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the distribution of attack types and techniques in a RED Phase 1 dataset.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the main RED Phase 1 dataset (JSONL format).")
    parser.add_argument("--output_dir", type=str, default="data/red",
                        help="Directory to save the distribution report.")
    parser.add_argument("--output_filename", type=str, default="red_phase1_distribution.json",
                        help="Filename for the distribution report JSON file.")
    args = parser.parse_args()

    cmd_str = f"python scripts/red_phase1_analyze_distribution.py --input_file {args.input_file} --output_dir {args.output_dir} --output_filename {args.output_filename}"
    
    try:
        output_file = analyze_distribution(args.input_file, args.output_dir, args.output_filename)
        log_message(cmd_str, "OK", output_file)
    except Exception as e:
        print(f"Error running analysis: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))