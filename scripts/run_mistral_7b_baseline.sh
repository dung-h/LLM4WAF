#!/usr/bin/env bash

set -euo pipefail

# Simple baseline script for testing the Mistral 7B adapter on v14 SFT data.
# Steps:
#   1. Activate WSL virtualenv
#   2. Extract mistral_7b_adapter.tar.gz if needed
#   3. Run the generic test script to sample a few payloads

REPO="/mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber"
cd "$REPO"

if [ ! -d ".venv" ]; then
  echo "[-] .venv not found in $REPO; please create it before running." >&2
  exit 1
fi

source .venv/bin/activate

ADAPTER_DIR="experiments/mistral_7b_adapter"
TARBALL="mistral_7b_adapter.tar.gz"

if [ ! -d "$ADAPTER_DIR" ]; then
  if [ ! -f "$TARBALL" ]; then
    echo "[-] $TARBALL not found in $REPO; cannot unpack adapter." >&2
    exit 1
  fi
  echo "[*] Extracting Mistral 7B adapter from $TARBALL into $ADAPTER_DIR ..."
  mkdir -p "$ADAPTER_DIR"
  tar -xzf "$TARBALL" -C "$ADAPTER_DIR"
fi

echo "[*] Testing Mistral 7B adapter on v14 SFT data..."

python scripts/test_mistral_v14_sft.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter_dir "$ADAPTER_DIR" \
  --dataset_path data/processed/v14_sft_data.jsonl \
  --num_samples 5

echo "[*] Done. Review the generated payloads above for a quick sanity check."
