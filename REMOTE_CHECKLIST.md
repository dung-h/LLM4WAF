# Pre-Upload Checklist for Remote Training

## ✅ Before Upload

### 1. Prepare Datasets
```bash
bash prepare_datasets.sh
```
- [ ] Phase 1 dataset created (10k samples)
- [ ] Phase 3 dataset verified

### 2. Cleanup Workspace
```bash
bash cleanup_for_remote.sh
```
- [ ] Removed eval results
- [ ] Removed logs
- [ ] Removed Python cache
- [ ] Removed temp files
- [ ] (Optional) Removed old experiments

### 3. Verify Configs
```bash
ls -l configs/remote_*.yaml
```
- [ ] remote_gemma2_2b_phase1.yaml
- [ ] remote_phi3_mini_phase1.yaml
- [ ] remote_qwen_7b_phase1.yaml

### 4. Make Scripts Executable
```bash
chmod +x *.sh scripts/*.sh
```

### 5. Compress Workspace
```bash
tar -czf llm4waf.tar.gz \
  --exclude='.git' \
  --exclude='experiments/*' \
  --exclude='eval/*' \
  --exclude='*.log' \
  .

# Check size
ls -lh llm4waf.tar.gz
```
**Target size**: <2GB without experiments, <20GB with experiments

---

## 📤 Upload to Remote

```bash
# Upload tarball
scp llm4waf.tar.gz user@remote-server:~/

# SSH to remote
ssh user@remote-server

# Extract
cd ~
tar -xzf llm4waf.tar.gz
cd LLM_in_Cyber
```

---

## 🚀 On Remote Server

### 1. Setup Environment
```bash
bash remote_quickstart.sh
```

### 2. Test GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

### 3. Verify HuggingFace Token
```bash
export HF_TOKEN="your_token_here"
# Or create ~/.cache/huggingface/token
```

### 4. Start Training
```bash
# Activate environment
source .venv/bin/activate

# Train Gemma Phase 1 (recommended first)
nohup python scripts/train_red.py \
  --config configs/remote_gemma2_2b_phase1.yaml \
  2>&1 | tee training_gemma_phase1.log &

# Monitor
tail -f training_gemma_phase1.log
nvidia-smi -l 1  # In another terminal
```

---

## 📊 Monitor Training

### Check Progress
```bash
# Loss trend
grep "Loss:" training_gemma_phase1.log | tail -20

# Epoch completion
grep "Epoch" training_gemma_phase1.log

# GPU usage
nvidia-smi

# Tensorboard
tensorboard --logdir experiments/remote_gemma2_2b_phase1/logs --port 6006
```

### Expected Metrics
- **GPU Memory**: 18-22GB for Gemma/Phi-3, 22-23GB for Qwen 7B
- **Training speed**: ~30-50 steps/hour
- **Loss**: Should decrease from ~2-3 to ~0.5-1.0

---

## ⚠️ Common Issues

### Issue: CUDA Out of Memory
**Solution**:
```yaml
# Reduce batch size in config
per_device_train_batch_size: 1
gradient_accumulation_steps: 32  # Double to keep effective batch = 16
```

### Issue: Phi-3 DynamicCache Error
**Solution**: Already fixed in train_red.py with `use_cache=False`

### Issue: HuggingFace Token Error
**Solution**:
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
# Or
huggingface-cli login
```

### Issue: Slow Data Loading
**Solution**:
```yaml
# Reduce workers in config
dataloader_num_workers: 4
dataloader_prefetch_factor: 2
```

---

## 📥 Download Results

### After Training Completes
```bash
# On remote server
cd experiments/
tar -czf remote_gemma2_2b_phase1.tar.gz remote_gemma2_2b_phase1/

# On local machine
scp user@remote-server:~/LLM_in_Cyber/experiments/remote_gemma2_2b_phase1.tar.gz .
```

### What to Download
- Checkpoints: `checkpoint-{step}/` folders
- Logs: `*.log` files
- Tensorboard: `logs/` directory
- Final adapter: `adapter_model.safetensors`, `adapter_config.json`

---

## 🎯 Training Schedule

**Day 1**: Gemma 2 2B
- Morning: Phase 1 (2-3 hours)
- Afternoon: Phase 2 (2-3 hours)  
- Evening: Phase 3 RL (3-4 hours)

**Day 2**: Phi-3 Mini
- Same schedule as Gemma

**Day 3**: Qwen 2.5 7B (if time allows)
- Phase 1 only (4-5 hours)

**Total**: 2-3 days for complete thesis training

---

## ✅ Success Criteria

### Phase 1
- [ ] Loss converges to <1.0
- [ ] Model generates valid payloads (not "Do not provide explanations")
- [ ] Manual test: 5 generated payloads look reasonable

### Phase 2
- [ ] Loss converges to <0.8
- [ ] Model uses observations in responses
- [ ] No significant degradation of Phase 1 knowledge

### Phase 3 RL
- [ ] Episode rewards increase over time
- [ ] Baseline reward stabilizes
- [ ] Model generates bypassing payloads (>50% success rate)

---

## 📝 Final Notes

1. **Always use `nohup` and `&`** for long training runs
2. **Monitor first 100 steps** to ensure no errors
3. **Save checkpoints frequently** (every 500 steps)
4. **Test adapters immediately** after each phase
5. **Keep logs** for thesis documentation
