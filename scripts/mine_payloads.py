import argparse
import csv
import json
import subprocess
from pathlib import Path
import os
import sys
import tempfile
from datetime import datetime, timedelta

print("DEBUG: Before Path import")
from pathlib import Path
print(f"DEBUG: Path imported: {Path}")

def convert_txt_to_jsonl(txt_path: Path, jsonl_path: Path, attack_type: str):
    """Reads a .txt file line-by-line and converts it to a .jsonl file."""
    print(f"    [CONVERT] Converting {txt_path.name} to {jsonl_path.name}...")
    count = 0
    with txt_path.open("r", encoding="utf-8", errors="ignore") as in_f, \
         jsonl_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            payload = line.strip()
            if payload:
                record = {"payload": payload, "attack_type": attack_type}
                out_f.write(json.dumps(record) + "\n")
                count += 1
    print(f"    [CONVERT] Converted {count} payloads.")
    return count > 0

def run_command(command: str):
    """Runs a command directly in the current shell and checks for errors."""
    print(f"  [EXEC] {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True, executable='/bin/bash')
    if process.returncode != 0:
        print(f"  [ERROR] Command failed with exit code {process.returncode}")
        print(f"  [STDERR] {process.stderr.strip()}")
        return False
    
    stdout_preview = (process.stdout.strip()[:200] + '...') if len(process.stdout.strip()) > 200 else process.stdout.strip()
    if stdout_preview:
        print(f"  [STDOUT] {stdout_preview}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Find, convert, and screen payloads from new vendor repositories.")
    args = parser.parse_args()

    vendor_dir = Path("vendor")
    processed_dir = Path("data/processed")
    
    # Use absolute path to venv python interpreter
    python_executable = "/mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber/.venv/bin/python"
    print(f"DEBUG: python_executable is {python_executable}")
    
    # Identify new repos based on creation date (cloned today)
    today = datetime.now().date()
    new_repos = []
    for d in vendor_dir.iterdir():
        if d.is_dir():
            try:
                # Use st_mtime as a proxy for clone time if birthtime is not directly available/reliable
                # We've already established mtime is a good enough proxy for creation time in this context
                mod_time = datetime.fromtimestamp(d.stat().st_mtime)
                if mod_time.date() == today:
                    new_repos.append(d)
            except FileNotFoundError:
                continue

    if not new_repos:
        print(f"No new vendor repositories found (cloned today, {today}).")
        return
        
    print(f"Found {len(new_repos)} new vendor repositories to process: {[r.name for r in new_repos]}")

    for repo_dir in new_repos:
        print(f"\n--- Processing repository: {repo_dir.name} ---")
        
        payload_files = list(repo_dir.rglob("*.txt"))
        
        if not payload_files:
            print("  No payload files (.txt) found in this repository.")
        else:
            print(f"  Found {len(payload_files)} potential payload files.")
            for payload_file in payload_files:
                if "xss" in payload_file.name.lower():
                    attack_type = "XSS"
                else:
                    attack_type = "SQLI"
                
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl", dir=".") as temp_jsonl_file:
                    temp_jsonl_path = Path(temp_jsonl_file.name)
                
                print(f"  - Processing file: {payload_file} (guessed type: {attack_type})")

                if not convert_txt_to_jsonl(payload_file, temp_jsonl_path, attack_type):
                    print(f"    [WARN] No payloads found in {payload_file}. Skipping.")
                    os.remove(temp_jsonl_path)
                    continue

                output_filename = f"seeds_from_{repo_dir.name}_{payload_file.stem}_screened.jsonl"
                output_path = processed_dir / output_filename
                
                command = f"{python_executable} scripts/screen_seeds_against_waf.py --input {temp_jsonl_path.as_posix()} --output {output_path.as_posix()}"
                
                success = run_command(command)
                if not success:
                    print(f"    [WARN] Failed to screen {payload_file}. Check errors above.")
                
                os.remove(temp_jsonl_path)
        
        print(f"--- Finished processing {repo_dir.name}. ---")

if __name__ == "__main__":
    main()