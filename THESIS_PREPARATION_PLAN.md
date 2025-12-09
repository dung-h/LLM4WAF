# Thesis Preparation Plan

**Created:** December 9, 2025  
**Purpose:** Complete plan for thesis report writing and repository preparation

---

## 📋 PHASE 1: ARCHIVE LEGACY & CLEANUP

### 1.1 Archive Old Phase 2 Datasets (Deprecated)

**Reason:** Old Phase 2 datasets with reasoning format lacked PASSED payloads, superseded by Phase 3 lightweight (now called Phase 2 with replay).

**Files to archive:**

```bash
mkdir data/processed/archive_old_phase2
mv data/processed/red_v40_phase2_reasoning.jsonl data/processed/archive_old_phase2/
mv data/processed/red_v40_phase2_reasoning_ready.jsonl data/processed/archive_old_phase2/
mv data/processed/red_v40_phase2_phi3_1500.jsonl data/processed/archive_old_phase2/
mv data/processed/red_v40_phase2_eval_100.jsonl data/processed/archive_old_phase2/
mv data/processed/red_v40_phase2_eval_test.jsonl data/processed/archive_old_phase2/
mv data/processed/phase2_rag_old.jsonl data/processed/archive_old_phase2/
```

### 1.2 Archive Legacy Configs

**Files to archive:**

```bash
mkdir configs/archive_legacy
mv configs/red_*.yaml configs/archive_legacy/
mv configs/phase2_*.yaml configs/archive_legacy/
mv configs/phase3_*.yaml configs/archive_legacy/
mv configs/phi3_mini_phase3_rl.yaml configs/archive_legacy/
mv configs/gemma2_2b_phase3_rl.yaml configs/archive_legacy/
mv configs/phi3_mini_phase4_rl_200epochs.yaml configs/archive_legacy/
mv configs/qwen_3b_phase4_rl_200epochs.yaml configs/archive_legacy/
mv configs/gemma2_2b_phase1_quick_test_1k.yaml configs/archive_legacy/
```

### 1.3 Update ARCHIVE_INDEX.md

Document all archived files with reasons.

---

## 📊 PHASE 2: DATASET ANALYSIS & DOCUMENTATION

### 2.1 Dataset Creation Process Analysis

**Objective:** Document the complete synthetic data generation pipeline using LLM APIs.

#### **Phase 1 Dataset Creation Analysis**

**Script to create:** `scripts/report/analyze_phase1_creation_process.py`

**Analysis points:**

1. **Source:** LLM API (Gemini/Deepseek) → Payload generation
2. **Prompt template reconstruction:**
   - Analyze existing Phase 1 samples
   - Infer prompt structure from `technique` and `attack_type` fields
   - Document system prompt, user prompt format
3. **Quality validation:**
   - DVWA + ModSecurity testing
   - Classification: PASSED vs BLOCKED
4. **Statistics:**
   - Total samples: 39,155 PASSED
   - Techniques distribution: 509 unique
   - Attack types: SQLI, XSS breakdown
   - Payload characteristics: length, encoding, obfuscation

#### **Phase 2 Dataset Creation Analysis**

**Script to create:** `scripts/report/analyze_phase2_creation_process.py`

**Analysis points:**

1. **Source:** Phase 1 model inference + Phase 1 replay buffer
2. **Format evolution:**
   - Phase 1: Simple technique-based prompts
   - Phase 2: Structured prompts with context, history, constraints
3. **Composition:**
   - 20,000 new observations (from model inference)
   - 2,000 Phase 1 replay (20% rate for knowledge retention)
4. **Quality metrics:**
   - Technique coverage: 517 unique
   - Payload complexity increase
   - Success rate against WAF

### 2.2 Technique Distribution Analysis

**Script to create:** `scripts/report/analyze_technique_distribution.py`

**Outputs:**

1. **Phase 1 technique distribution:**

   - Count per technique
   - Percentage distribution
   - Top 20 most common techniques
   - Long-tail analysis

2. **Phase 2 technique distribution:**

   - Comparison with Phase 1
   - New techniques introduced: 517 - 509 = 8 new
   - Technique evolution/refinement

3. **Visualizations:**
   - Bar chart: Top 20 techniques
   - Pie chart: Attack type distribution (SQLI vs XSS)
   - Histogram: Payload length distribution
   - Heatmap: Technique correlation matrix

### 2.3 Synthetic Data Quality Analysis

**Script to create:** `scripts/report/analyze_synthetic_data_quality.py`

**Critical concerns for LLM-generated data:**

1. **Diversity Metrics:**

   - Unique payload ratio
   - N-gram diversity (2-gram, 3-gram)
   - Character-level entropy
   - Technique coverage completeness

2. **Validity Metrics:**

   - Syntax correctness (SQL/JavaScript validation)
   - WAF bypass effectiveness
   - Injection point compatibility

3. **Bias Detection:**

   - Over-representation of certain techniques
   - Prompt leakage (common LLM artifacts)
   - Hallucination detection (invalid SQL syntax, etc.)

4. **Comparison with Baseline:**
   - Compare with PayloadsAllTheThings (if available for reference)
   - Manual expert validation (sample 100 payloads)

### 2.4 Data Generation Reconstruction Script

**Script to create:** `scripts/report/reconstruct_dataset_generation.py`

**Purpose:** Demonstrate how Phase 1 dataset was created (for thesis documentation).

**Features:**

- Support Gemini API and Deepseek API
- Configurable prompts (reconstructed from dataset analysis)
- DVWA + ModSecurity validation
- Output format matching Phase 1 dataset
- Quality filtering (only PASSED payloads)

**Prompt reconstruction strategy:**

```python
# Inferred from Phase 1 dataset structure
SYSTEM_PROMPT = """You are an expert penetration tester specializing in WAF bypass techniques.
Generate SQL injection and XSS payloads using various obfuscation and evasion techniques."""

USER_PROMPT_TEMPLATE = """Generate a {attack_type} payload using the technique: {technique}.
Requirements:
- Must be a valid {attack_type} payload
- Use the specified technique for WAF evasion
- Keep it concise and effective
- Output only the payload, no explanations

Technique: {technique}
Attack Type: {attack_type}
"""
```

---

## 📝 PHASE 3: THESIS REPORT OUTLINE

### 3.1 Dataset Section Structure

```markdown
## Chapter X: Dataset Construction

### X.1 Synthetic Data Generation Pipeline

#### X.1.1 Phase 1: Foundation Dataset Creation

- LLM-based payload generation (Gemini/Deepseek API)
- Prompt engineering strategy
- Quality validation via DVWA + ModSecurity
- Dataset statistics and characteristics

#### X.1.2 Phase 2: Adaptive Dataset Construction

- Model-driven inference from Phase 1 adapter
- Structured prompt format design
- Replay buffer integration (20% rate)
- Dataset evolution analysis

### X.2 Dataset Quality Analysis

#### X.2.1 Technique Distribution

- 509 unique techniques in Phase 1
- 517 unique techniques in Phase 2
- Distribution analysis and long-tail handling
- Coverage completeness validation

#### X.2.2 Synthetic Data Quality Assurance

- Diversity metrics (n-gram, entropy)
- Validity metrics (syntax, effectiveness)
- Bias detection and mitigation
- Comparison with expert-curated datasets

#### X.2.3 Stratified Sampling Strategy

- Problem: Simple head sampling only captured 3/509 techniques
- Solution: Proportional stratified sampling
- Validation: All 509 techniques represented

### X.3 Dataset Artifacts

#### Table X.1: Dataset Statistics

| Phase                  | Samples | Techniques | Attack Types | Avg Length | Size (MB) |
| ---------------------- | ------- | ---------- | ------------ | ---------- | --------- |
| Phase 1 (Full)         | 39,155  | 509        | SQLI, XSS    | ~200 chars | 17.74     |
| Phase 1 (Balanced)     | 10,000  | 509        | SQLI, XSS    | ~200 chars | 4.51      |
| Phase 2 (Observations) | 20,000  | 517        | SQLI, XSS    | ~300 chars | 20.91     |
| Phase 2 (Final)        | 22,000  | 517        | SQLI, XSS    | ~290 chars | 22.71     |

#### Table X.2: Top 20 Techniques (Phase 1)

[Generated from analyze_technique_distribution.py]

#### Figure X.1: Technique Distribution

[Bar chart showing distribution across techniques]

#### Figure X.2: Payload Length Distribution

[Histogram showing length characteristics]

#### Figure X.3: Attack Type Breakdown

[Pie chart: SQLI vs XSS ratio]

### X.4 Ethical Considerations

- All payloads tested only on controlled environments (DVWA)
- No real-world systems targeted
- Data used solely for research purposes
- Synthetic generation reduces reliance on leaked/stolen data
```

---

## 🛠️ PHASE 4: CLEAN REPOSITORY PREPARATION

### 4.1 Repository Structure for Submission

```
LLM4WAF-Final/
├── README.md                          # Main documentation
├── METHODOLOGY.md                     # Detailed methodology
├── REPRODUCTION.md                    # Reproduction instructions
├── requirements.txt                   # Dependencies
│
├── data/
│   ├── processed/
│   │   ├── phase1_balanced_10k.jsonl
│   │   ├── phase2_with_replay_22k.jsonl
│   │   └── phase1_passed_only_39k.jsonl
│   └── README.md                      # Data documentation
│
├── configs/
│   ├── phase1/
│   │   ├── gemma_2b.yaml
│   │   ├── phi3_mini.yaml
│   │   └── qwen_7b.yaml
│   ├── phase2/
│   │   └── (same structure)
│   ├── phase3_rl/
│   │   └── (same structure)
│   └── README.md                      # Config documentation
│
├── scripts/
│   ├── dataset_creation/
│   │   ├── create_phase1_balanced.py
│   │   ├── create_phase2_with_replay.py
│   │   └── reconstruct_generation.py  # NEW
│   ├── training/
│   │   ├── train_sft.py
│   │   └── train_rl.py
│   ├── evaluation/
│   │   ├── evaluate_adapters.py
│   │   └── generate_report.py
│   └── analysis/                      # NEW - For thesis
│       ├── analyze_phase1_process.py
│       ├── analyze_phase2_process.py
│       ├── analyze_technique_distribution.py
│       └── analyze_synthetic_quality.py
│
├── reports/                           # NEW
│   ├── dataset_analysis/
│   │   ├── technique_distribution.csv
│   │   ├── quality_metrics.json
│   │   └── figures/
│   │       ├── technique_dist_phase1.png
│   │       ├── technique_dist_phase2.png
│   │       ├── payload_length_hist.png
│   │       └── attack_type_pie.png
│   └── training_results/
│       └── (evaluation results)
│
└── docs/
    ├── DATASET_CREATION.md            # Dataset creation process
    ├── TRAINING_GUIDE.md              # Training guide
    └── API_REFERENCE.md               # Code API reference
```

### 4.2 Scripts to Clean/Refactor

**Current scripts to refactor for submission:**

1. **`train_red.py` → `scripts/training/train_sft.py`**

   - Remove temporary fixes
   - Clean up comments
   - Add comprehensive docstrings
   - Remove debug prints

2. **`train_rl_reinforce.py` → `scripts/training/train_rl.py`**

   - Clean VRAM optimization code
   - Document memory management strategy
   - Remove experimental code paths

3. **Dataset creation scripts:**

   - Keep as-is, already clean
   - Add detailed docstrings

4. **New analysis scripts (for thesis):**
   - Create 4 new scripts in `scripts/analysis/`

### 4.3 Documentation to Create

**NEW documentation files:**

1. **`METHODOLOGY.md`**

   - Dataset creation methodology
   - Training pipeline description
   - Evaluation protocol

2. **`REPRODUCTION.md`**

   - Step-by-step reproduction instructions
   - Hardware requirements
   - Expected results

3. **`data/README.md`**

   - Dataset documentation
   - Field descriptions
   - Statistics summary

4. **`configs/README.md`**

   - Config file structure
   - Hyperparameter choices
   - Usage examples

5. **`docs/DATASET_CREATION.md`**
   - Detailed dataset creation process
   - LLM API usage
   - Quality validation pipeline

---

## 📊 PHASE 5: ANALYSIS SCRIPTS IMPLEMENTATION

### 5.1 Script 1: `analyze_phase1_creation_process.py`

**Purpose:** Analyze Phase 1 dataset creation characteristics

**Outputs:**

- `reports/dataset_analysis/phase1_creation_summary.txt`
- `reports/dataset_analysis/phase1_statistics.json`
- Prompt template reconstruction

**Key metrics:**

- Total samples, techniques, attack types
- Payload length statistics (mean, median, P95, P99)
- Character distribution
- Technique frequency distribution

### 5.2 Script 2: `analyze_phase2_creation_process.py`

**Purpose:** Analyze Phase 2 dataset construction

**Outputs:**

- `reports/dataset_analysis/phase2_creation_summary.txt`
- `reports/dataset_analysis/phase2_statistics.json`
- Comparison with Phase 1

**Key metrics:**

- Replay buffer composition (20k + 2k)
- New techniques introduced (8 new)
- Payload complexity evolution
- Format structure analysis

### 5.3 Script 3: `analyze_technique_distribution.py`

**Purpose:** Comprehensive technique distribution analysis

**Outputs:**

- `reports/dataset_analysis/technique_distribution.csv`
- `reports/dataset_analysis/figures/technique_dist_phase1.png`
- `reports/dataset_analysis/figures/technique_dist_phase2.png`
- `reports/dataset_analysis/figures/technique_comparison.png`

**Visualizations:**

- Bar charts (top 20 techniques)
- Full distribution histograms
- Phase 1 vs Phase 2 comparison
- Attack type breakdown (pie chart)

### 5.4 Script 4: `analyze_synthetic_quality.py`

**Purpose:** Synthetic data quality assessment

**Outputs:**

- `reports/dataset_analysis/quality_metrics.json`
- `reports/dataset_analysis/quality_report.txt`

**Metrics:**

1. **Diversity:**

   - Unique payload ratio
   - 2-gram diversity score
   - 3-gram diversity score
   - Shannon entropy

2. **Validity:**

   - SQL syntax validation (for SQLI)
   - JavaScript syntax validation (for XSS)
   - Encoding correctness

3. **Bias Detection:**
   - Technique over-representation (top 10% vs bottom 10%)
   - Prompt leakage patterns (common LLM artifacts)
   - Hallucination rate (invalid syntax)

### 5.5 Script 5: `reconstruct_dataset_generation.py`

**Purpose:** Demonstrate dataset generation process (for thesis documentation)

**Features:**

- **API Support:** Gemini, Deepseek
- **Configurable prompts** (inferred from dataset analysis)
- **WAF validation** (optional DVWA + ModSecurity testing)
- **Output format:** Matches Phase 1 dataset structure

**Usage:**

```bash
python scripts/dataset_creation/reconstruct_generation.py \
  --api gemini \
  --api-key YOUR_KEY \
  --technique "Double URL Encode" \
  --attack-type SQLI \
  --num-samples 10 \
  --output sample_generation.jsonl \
  --validate-waf  # Optional: test against DVWA
```

---

## 🎯 PHASE 6: EXECUTION PLAN

### Week 1: Cleanup & Archive (Dec 9-15)

- [x] Archive old log files (DONE)
- [x] Archive old documentation (DONE)
- [x] Archive deprecated scripts (DONE)
- [ ] Archive old Phase 2 datasets
- [ ] Archive legacy configs
- [ ] Update ARCHIVE_INDEX.md
- [ ] Update WORKSPACE_REPORT.md

### Week 2: Analysis Scripts (Dec 16-22)

- [ ] Implement `analyze_phase1_creation_process.py`
- [ ] Implement `analyze_phase2_creation_process.py`
- [ ] Implement `analyze_technique_distribution.py`
- [ ] Implement `analyze_synthetic_quality.py`
- [ ] Implement `reconstruct_dataset_generation.py`

### Week 3: Data Analysis & Visualization (Dec 23-29)

- [ ] Run all analysis scripts
- [ ] Generate visualizations (charts, graphs)
- [ ] Collect statistics for thesis tables
- [ ] Review and validate findings

### Week 4: Report Writing (Dec 30 - Jan 5)

- [ ] Write Dataset Section (Chapter X)
- [ ] Create tables and figures
- [ ] Document methodology
- [ ] Write quality analysis

### Week 5: Repository Cleanup (Jan 6-12)

- [ ] Refactor core scripts
- [ ] Create clean repository structure
- [ ] Write all documentation files
- [ ] Test reproduction instructions

### Week 6: Final Review (Jan 13-19)

- [ ] Review thesis chapter
- [ ] Review repository
- [ ] Get advisor feedback
- [ ] Finalize submission

---

## 📌 IMMEDIATE NEXT STEPS

### Step 1: Archive Old Phase 2 Data (TODAY)

```bash
mkdir data/processed/archive_old_phase2
# Move 6 old phase2 files
```

### Step 2: Archive Legacy Configs (TODAY)

```bash
mkdir configs/archive_legacy
# Move 25 legacy configs
```

### Step 3: Create Analysis Scripts Skeleton (THIS WEEK)

```bash
mkdir scripts/analysis
mkdir reports/dataset_analysis
mkdir reports/dataset_analysis/figures
# Create 5 analysis scripts with basic structure
```

### Step 4: Implement First Analysis Script (THIS WEEK)

```bash
# Start with analyze_technique_distribution.py
# Easiest to implement, provides immediate value
```

---

## 🔍 KEY QUESTIONS TO RESOLVE

1. **Prompt Reconstruction:**

   - Do you have original prompts used for Phase 1 generation?
   - If not, can we infer from dataset structure?

2. **API Keys:**

   - Gemini API key available?
   - Deepseek API key available?

3. **WAF Validation:**

   - Should `reconstruct_generation.py` validate against DVWA?
   - Or just demonstrate generation without validation?

4. **Visualization Style:**

   - Preferred chart library: matplotlib, seaborn, plotly?
   - Color scheme for thesis consistency?

5. **Thesis Chapter:**
   - Which chapter will contain dataset section?
   - Page limit for dataset analysis?

---

## 📊 SUCCESS METRICS

**Repository Quality:**

- [ ] All legacy files archived with documentation
- [ ] Only production-ready code in main directories
- [ ] Clean commit history (squash temp commits)
- [ ] Comprehensive README and documentation
- [ ] Reproduction instructions verified

**Thesis Quality:**

- [ ] Dataset creation process fully documented
- [ ] Technique distribution analyzed with visualizations
- [ ] Synthetic data quality assessed with metrics
- [ ] All tables and figures generated programmatically
- [ ] Methodology clearly explained

**Reproducibility:**

- [ ] Someone can recreate Phase 1 dataset using `reconstruct_generation.py`
- [ ] Someone can retrain models using provided configs
- [ ] Someone can regenerate all analysis reports
- [ ] All scripts run without errors

---

**Status:** 🟡 Planning Phase  
**Next Action:** Archive old Phase 2 data and legacy configs  
**Priority:** HIGH - Thesis deadline approaching
