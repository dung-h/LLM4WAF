# PowerShell script to run Qwen RL training
Write-Host "Starting Qwen 2.5 3B RL Training (200 epochs)..." -ForegroundColor Green
Write-Host "Config: configs/qwen_3b_phase4_rl_200epochs.yaml"
Write-Host "Output: experiments/qwen_3b_phase4_rl_200epochs/"
Write-Host ""

wsl bash -c "cd /mnt/d/AI_in_cyber/LLM_in_Cyber && source .venv/bin/activate && python scripts/train_rl_reinforce.py --config configs/qwen_3b_phase4_rl_200epochs.yaml 2>&1 | tee training_qwen_rl_200epochs.log"

Write-Host ""
Write-Host "✅ Qwen 2.5 3B RL training complete!" -ForegroundColor Green
Write-Host "Log: training_qwen_rl_200epochs.log"
