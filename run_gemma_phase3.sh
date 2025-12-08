#!/bin/bash

# Active environment
source .venv/bin/activate

# Export Token (Set your own HF_TOKEN in environment)
# export HF_TOKEN=your_token_here

# Config file
CONFIG="configs/red_phase3_adaptive_gemma.yaml"
LOG_FILE="train_phase3_gemma.log"

echo "Starting training for Gemma 2 2B Phase 3..."
echo "Config: $CONFIG"
echo "Log: $LOG_FILE"

# Run with nohup
nohup python scripts/train_red.py --config $CONFIG > $LOG_FILE 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Monitor logs with: tail -f $LOG_FILE"
