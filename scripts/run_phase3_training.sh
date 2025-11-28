#!/bin/bash
# Phase 3 Training Execution Script
# Automated training pipeline for 4x RTX 3090 setup

set -e

# Configuration
WORKSPACE_DIR="/workspace/LLM_in_Cyber"
RESULTS_DIR="${WORKSPACE_DIR}/results/phase3_training"
LOG_DIR="${WORKSPACE_DIR}/logs/phase3"

# Create directories
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Model priorities (based on Phase 2 performance)
declare -a MODELS=("qwen_7b" "deepseek_7b" "llama_8b" "phi3_14b")
declare -A CONFIGS=(
    ["qwen_7b"]="configs/phase3_qwen_7b_4x3090.yaml"
    ["deepseek_7b"]="configs/phase3_deepseek_7b_4x3090.yaml" 
    ["llama_8b"]="configs/phase3_llama_8b_4x3090.yaml"
    ["phi3_14b"]="configs/phase3_phi3_14b_4x3090.yaml"
)

# GPU monitoring function
monitor_gpu() {
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
}

# Training function
train_model() {
    local model_name=$1
    local config_path=$2
    local start_time=$(date +%s)
    
    echo ""
    echo "🚀 Starting training: $model_name"
    echo "📋 Config: $config_path"
    echo "⏰ Started: $(date)"
    
    # Log file
    local log_file="$LOG_DIR/${model_name}_training.log"
    
    # GPU check before training
    monitor_gpu
    
    # Run training with full logging
    python scripts/run_sft.py \
        --config "$config_path" \
        --output_dir "experiments/${model_name}_4gpu_phase3" \
        --logging_dir "$LOG_DIR/${model_name}_tensorboard" \
        --run_name "${model_name}_phase3_4x3090" \
        --report_to "tensorboard" \
        --deepspeed_config_file "configs/deepspeed_stage2.json" \
        2>&1 | tee "$log_file"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    echo "✅ $model_name training complete!"
    echo "⏱️  Duration: ${hours}h ${minutes}m"
    
    # Log training completion
    echo "{\"model\": \"$model_name\", \"duration_hours\": $((duration / 3600)), \"status\": \"completed\", \"timestamp\": \"$(date -Iseconds)\"}" >> "$RESULTS_DIR/training_log.jsonl"
}

# Evaluation function
evaluate_model() {
    local model_name=$1
    
    echo "📊 Evaluating $model_name..."
    
    # Quick quality check
    python scripts/evaluate_model.py \
        --model_path "experiments/${model_name}_4gpu_phase3" \
        --test_data "data/splits/sft_experiment/test_200_${model_name/_*/}.jsonl" \
        --output_file "$RESULTS_DIR/${model_name}_quick_eval.jsonl" \
        --max_length 512 \
        --batch_size 8
    
    # Quality scoring
    python scripts/evaluate_quality.py \
        --input_file "$RESULTS_DIR/${model_name}_quick_eval.jsonl" \
        --output_file "$RESULTS_DIR/${model_name}_quality.json"
    
    echo "✅ $model_name evaluation complete"
}

# Main execution pipeline
main() {
    echo "🎯 Phase 3 Training Pipeline Started"
    echo "🖥️  Target: 4x RTX 3090 GPUs"
    echo "📊 Models: ${MODELS[*]}"
    echo ""
    
    # Initial GPU check
    echo "Initial GPU Status:"
    monitor_gpu
    echo ""
    
    # Sequential training (avoid resource conflicts)
    for model in "${MODELS[@]}"; do
        config="${CONFIGS[$model]}"
        
        if [ -f "$WORKSPACE_DIR/$config" ]; then
            # Train model
            train_model "$model" "$config"
            
            # Quick evaluation
            evaluate_model "$model"
            
            echo "⏸️  Brief cooldown..."
            sleep 30
        else
            echo "❌ Config not found: $config"
        fi
    done
    
    echo ""
    echo "🏁 All models training complete!"
    echo "📈 Running comprehensive evaluation..."
    
    # Comprehensive evaluation
    python scripts/phase3_auto_evaluator.py
    
    echo "🔍 Generating model comparison..."
    python scripts/phase3_model_comparison.py
    
    echo ""
    echo "🎯 Phase 3 Pipeline Complete!"
    echo "📁 Results: $RESULTS_DIR"
    echo "📊 Logs: $LOG_DIR"
}

# Error handling
handle_error() {
    echo "❌ Error occurred on line $1"
    echo "🚨 Training pipeline failed"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Execute main pipeline
main "$@"