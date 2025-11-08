# v6 Multi-Attack Quick Start

**Goal**: Expand from SQLi-only (v5) → SQLi + XSS (v6)

---

## ⚡ TL;DR Commands

```bash
# 1. Download XSS data (5 min)
cd data/raw && git clone https://github.com/swisskyrepo/PayloadsAllTheThings.git

# 2. Create seed_xss.csv (manual - 20 payloads)
# See template in AGENT_INSTRUCTION.md

# 3. Enrich with DeepSeek (10 min, costs ~$0.10)
export DEEPSEEK_API_KEY="sk-..."
python scripts/etl/enrich_xss_deepseek.py

# 4. Combine datasets (1 min)
python scripts/etl/combine_sqli_xss.py

# 5. Train v6 (60-90 min)
nohup python scripts/train_red.py --config configs/red_v6_multi_attack.yaml > logs/train_v6_multi.log 2>&1 &

# 6. Test
python scripts/simple_gen_v6_multi.py
```

**Total Time**: ~2 hours  
**Total Cost**: ~$0.10 (DeepSeek API)

---

## 📁 Files to Create

### 1. `data/raw/seed_xss.csv`
```csv
payload,type,description
<script>alert('XSS')</script>,reflected,Basic script tag
<img src=x onerror=alert(1)>,reflected,Image error handler
<svg onload=alert(1)>,reflected,SVG onload event
... (20 total)
```

### 2. `scripts/etl/enrich_xss_deepseek.py`
See AGENT_INSTRUCTION.md → Phase 2 for complete code.

### 3. `scripts/etl/combine_sqli_xss.py`
See AGENT_INSTRUCTION.md → Phase 3 for complete code.

### 4. `configs/red_v6_multi_attack.yaml`
See AGENT_INSTRUCTION.md → Phase 4 for complete config.

### 5. `scripts/simple_gen_v6_multi.py`
See AGENT_INSTRUCTION.md → Phase 6 for complete code.

### 6. `replay/harness_xss.py`
See AGENT_INSTRUCTION.md → Phase 7 for complete code.

---

## 📊 Expected Results

### Dataset Sizes
- SQLi: 1,136 samples (from v5)
- XSS: 20 seeds → ~20-50 enriched samples (DeepSeek)
- **Total: ~1,156-1,186 samples**
- Split: 80% train / 10% val / 10% test

### Training Metrics
- Runtime: 60-90 minutes
- Final loss: < 1.0
- Model size: ~82MB (adapter only)
- GPU memory: ~3-4GB VRAM

### Generation Quality
- SQLi payloads: Should maintain 83.3% bypass rate
- XSS payloads: Target 70-80% bypass rate (new attack type)
- Combined: Target 80%+ average bypass rate

---

## 🔍 Validation Checklist

**Before Training:**
- [ ] XSS dataset downloaded (PayloadsAllTheThings)
- [ ] seed_xss.csv created (20+ payloads)
- [ ] DeepSeek API key set (`DEEPSEEK_API_KEY`)
- [ ] Enrichment completed (xss_enriched_deepseek.jsonl exists)
- [ ] Combined dataset created (red_train_v6_multi.jsonl)
- [ ] Dataset sizes verified (wc -l data/processed/red_*_v6_multi.jsonl)
- [ ] Config file created (configs/red_v6_multi_attack.yaml)
- [ ] Field mapping uses DICT not LIST (critical!)

**During Training:**
- [ ] GPU memory monitored (nvidia-smi)
- [ ] Loss decreasing (tail -f logs/train_v6_multi.log)
- [ ] No CUDA OOM errors
- [ ] Checkpoints saving every 50 steps

**After Training:**
- [ ] Model saved (experiments/red_gemma2_v6_multi/adapter/)
- [ ] adapter_model.safetensors exists (~82MB)
- [ ] Final loss < 1.0
- [ ] Generated payloads look valid (syntax check)
- [ ] WAF testing shows bypass rate (both SQLi and XSS)

---

## 🚨 Common Issues

### Issue 1: DeepSeek API Fails
```bash
# Check API key
echo $DEEPSEEK_API_KEY

# Test API
curl https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"test"}]}'
```

### Issue 2: Dataset Combination Fails
```bash
# Check file sizes
ls -lh data/processed/red_train.jsonl
ls -lh data/processed/xss_enriched_deepseek.jsonl

# Verify JSON format
head -1 data/processed/red_train.jsonl | jq .
head -1 data/processed/xss_enriched_deepseek.jsonl | jq .
```

### Issue 3: Training Hangs at PEFT Loading
```python
# Add to train_red.py before PEFT load:
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

### Issue 4: Low XSS Bypass Rate
```bash
# Check XSS payload syntax
head -10 results/v6_xss_payloads.txt

# Manually test one payload
curl "http://localhost:8080/vulnerabilities/xss_r/?name=<script>alert(1)</script>"

# Check ModSecurity logs
docker logs waf_nginx_modsec | grep XSS
```

---

## 📈 Performance Targets

| Metric | v5 (SQLi Only) | v6 (Multi-Attack) Target |
|--------|----------------|--------------------------|
| Dataset Size | 348 samples | 1,156+ samples |
| Attack Types | 1 (SQLi) | 2 (SQLi + XSS) |
| SQLi Bypass | 83.3% | ≥80% (maintain) |
| XSS Bypass | N/A | ≥70% (new) |
| Combined Bypass | 83.3% | ≥80% |
| Training Time | 38 min | 60-90 min |
| Model Size | 82MB | ~82MB (same) |

---

## 📝 Next Steps After v6

1. **Add more attack types**:
   - Command Injection
   - Path Traversal
   - LDAP Injection
   - XML Injection

2. **Expand XSS dataset**:
   - Scrape PortSwigger XSS cheat sheet (500+ payloads)
   - Add OWASP XSS examples
   - Collect from bug bounty reports

3. **Improve WAF bypass**:
   - Analyze failed payloads
   - Generate obfuscation variations
   - Test against higher paranoia levels (PL2, PL3)

4. **Deploy v6 to Hugging Face**:
   ```bash
   huggingface-cli repo create gemma2-v6-multi-attack
   # Upload adapter to HF
   ```

---

**Status**: 🚧 Ready for Implementation  
**Priority**: High (expands attack coverage from 1 → 2 types)  
**Risk**: Low (same architecture as v5, proven to work)
