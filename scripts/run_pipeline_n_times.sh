#!/bin/bash

NUM_RUNS=20
# The python script itself logs to logs/attack_pipeline_results.jsonl
# This variable here is just for convenience if we wanted a separate wrapper log.
WRAPPER_LOG_FILE="attack_pipeline_multi_run_wrapper_results.log" 

echo "Starting Attack Pipeline for Phi-3 Mini Phase 2, {NUM_RUNS} times."
echo "Results will be appended to logs/attack_pipeline_results.jsonl by the Python script."
echo "---"

# Clear previous multi-run wrapper log if it exists (optional, as main logs are in .jsonl)
if [ -f "$WRAPPER_LOG_FILE" ]; then
    rm "$WRAPPER_LOG_FILE"
fi

# Ensure the main attack log is clear if this is meant to be a fresh set of 20 runs
# This will clear the cumulative logs/attack_pipeline_results.jsonl from previous runs
# It's important to confirm if the user wants to append or overwrite for this 20-run sequence.
# For now, I'll assume a fresh set is desired. If appending is required, this line should be removed.
if [ -f "logs/attack_pipeline_results.jsonl" ]; then
    rm "logs/attack_pipeline_results.jsonl"
fi


for i in $(seq 1 $NUM_RUNS); do
    echo "=== Running pipeline iteration $i/$NUM_RUNS ===" >> "$WRAPPER_LOG_FILE"
    # It's important to reactivate venv inside the loop or ensure it's active outside
    # and not deactivating itself, which source does. Let's make it robust.
    # Ensure HF_TOKEN is set in your environment before running this script.
    wsl bash -c "source .venv/bin/activate && python scripts/run_attack_pipeline.py" >> "$WRAPPER_LOG_FILE" 2>&1
    echo "--- Iteration $i complete ---" >> "$WRAPPER_LOG_FILE"
    sleep 5 # Small delay between runs to avoid overwhelming WAF/DVWA
done

echo "All $NUM_RUNS pipeline iterations completed." >> "$WRAPPER_LOG_FILE"
echo "Check logs/attack_pipeline_results.jsonl for detailed outcomes of each attack."
echo "Check $WRAPPER_LOG_FILE for execution details of the wrapper script."
