# Writeup Payload Extraction Pipeline

**Customized version inspired by CyberLLMInstruct for WAF payload generation.**

## 🎯 Purpose

Extract WAF bypass payloads from CTF writeups using LLM-based extraction, NOT chatbot training.

## 🏗️ Architecture

```
writeup_pipeline/
├── crawlers/           # Data collection from multiple sources
│   ├── ctftime.py      # CTFtime API + scraping
│   ├── github.py       # GitHub search + RAW files
│   ├── medium.py       # Medium RSS feeds
│   └── base.py         # Base crawler class
├── extractors/         # LLM-based payload extraction
│   ├── llm_extractor.py   # Gemma 2 2B extraction
│   ├── prompts.py         # Prompt templates
│   └── parser.py          # JSON parsing & cleanup
├── validators/         # Quality control
│   ├── content.py      # Content validation
│   ├── format.py       # Format validation
│   └── waf.py          # WAF testing
├── data/              # Pipeline data storage
│   ├── raw/           # Crawled writeups
│   ├── extracted/     # LLM extracted payloads
│   ├── validated/     # Validated payloads
│   └── passed/        # WAF passed payloads
├── config.yaml        # Configuration
├── pipeline.py        # Main orchestrator
└── README.md         # This file
```

## 🔄 Pipeline Flow

```
1. CRAWL (500-1000 writeups)
   ├─ CTFtime (200 writeups)
   ├─ GitHub (300 writeups)
   └─ Medium (100 writeups)
          ↓
2. EXTRACT (3-5 payloads/writeup)
   ├─ Preprocessing (chunk, clean)
   ├─ LLM extraction (Gemma 2 2B)
   └─ JSON parsing
          ↓
3. VALIDATE (60-80% pass)
   ├─ Format validation
   ├─ Content validation
   └─ Deduplication
          ↓
4. WAF TEST (5-15% pass)
   ├─ ModSecurity testing
   ├─ Result logging
   └─ Final dataset
          ↓
5. MERGE
   └─ v29 (1414) + new (100-300) = v30 (1500-1700)
```

## 📊 Expected Metrics

| Stage    | Input              | Output             | Pass Rate       |
| -------- | ------------------ | ------------------ | --------------- |
| Crawl    | N/A                | 500-1000 writeups  | N/A             |
| Extract  | 500-1000 writeups  | 2000-5000 payloads | 3-5 per writeup |
| Validate | 2000-5000 payloads | 1500-2500 payloads | 60-80%          |
| WAF Test | 1500-2500 payloads | 100-300 passed     | 5-15%           |
| Merge    | v29 + new          | v30 dataset        | N/A             |

**Final Target: v30 = 1,500-1,700 passed payloads**

## 🆚 Differences from CyberLLMInstruct

| Aspect         | CyberLLMInstruct                                         | Our Pipeline                                    |
| -------------- | -------------------------------------------------------- | ----------------------------------------------- |
| **Goal**       | Generate instruction-response pairs for chatbot training | Extract WAF bypass payloads for model training  |
| **LLM Usage**  | Process/structure writeups                               | Extract specific payloads with context          |
| **Output**     | Question-answer dataset                                  | Executable payloads with bypass reasons         |
| **Validation** | Format/coherence                                         | WAF testing + format + content                  |
| **Scale**      | 1000+ writeups → instruction pairs                       | 500-1000 writeups → 100-300 WAF-passed payloads |
| **Focus**      | Educational chatbot                                      | Offensive security model                        |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml`:

```yaml
crawlers:
  ctftime:
    limit: 200
    rate_limit: 2.0
  github:
    limit: 300
  medium:
    limit: 100

extractors:
  model: "google/gemma-2-2b-it"
  batch_size: 10
  max_tokens: 500

validators:
  waf_url: "http://localhost:8080"
  workers: 40
```

### 3. Run Pipeline

```bash
# Full pipeline
python pipeline.py --mode full

# Individual stages
python pipeline.py --mode crawl
python pipeline.py --mode extract
python pipeline.py --mode validate
python pipeline.py --mode waf-test
```

## 📝 Usage Examples

### Crawl Only

```python
from crawlers.ctftime import CTFtimeCrawler

crawler = CTFtimeCrawler()
writeups = crawler.crawl(limit=200)
```

### Extract Only

```python
from extractors.llm_extractor import PayloadExtractor

extractor = PayloadExtractor(model="google/gemma-2-2b-it")
payloads = extractor.extract_from_writeup(writeup_content)
```

### Validate Only

```python
from validators.content import ContentValidator

validator = ContentValidator()
valid, reason = validator.validate(payload_obj)
```

## 🔧 Configuration

### Crawler Settings

- `rate_limit`: Delay between requests (seconds)
- `limit`: Maximum writeups per source
- `timeout`: Request timeout (seconds)

### Extractor Settings

- `model`: HuggingFace model ID
- `batch_size`: Parallel processing batch size
- `chunk_size`: Text chunk size for LLM (chars)
- `max_tokens`: Max output tokens

### Validator Settings

- `min_length`: Minimum payload length
- `max_length`: Maximum payload length
- `allowed_types`: ['xss', 'sqli']

## 📦 Output Format

### Extracted Payload

```json
{
  "payload": "<svg/onload=alert(1)>",
  "attack_type": "xss",
  "bypass_technique": "self-closing tag",
  "waf_bypassed": "ModSecurity",
  "context": "Used in stored XSS challenge",
  "source_url": "https://ctftime.org/writeup/12345",
  "source_title": "Example CTF Writeup",
  "extracted_at": "2025-11-27T10:00:00"
}
```

### WAF Test Result

```json
{
  "payload": "<svg/onload=alert(1)>",
  "attack_type": "xss",
  "waf_status": "passed",
  "response_code": 200,
  "response_time": 0.123,
  "tested_at": "2025-11-27T10:05:00"
}
```

## 📈 Monitoring

### Progress Tracking

```bash
# Check crawl progress
python pipeline.py --status crawl

# Check extraction progress
python pipeline.py --status extract

# Check WAF testing progress
python pipeline.py --status waf-test
```

### Statistics

```bash
# Show pipeline statistics
python pipeline.py --stats
```

## 🐛 Troubleshooting

### Low Extraction Rate

- Check LLM prompt effectiveness
- Adjust `chunk_size` in config
- Review writeup quality

### Low WAF Pass Rate

- Validate payload format
- Check WAF configuration
- Review bypass techniques

### Slow Crawling

- Increase `rate_limit`
- Use parallel crawling
- Check network connection

## 📚 References

- **CyberLLMInstruct**: [GitHub](https://github.com/imomoe233/CyberLLMInstruct) - Original inspiration
- **Our differences**: Focused on payload extraction, not chatbot training
- **Strategy**: See [WRITEUP_STRATEGY.md](../WRITEUP_STRATEGY.md)

## 📄 License

MIT License - Customized for WAF bypass payload generation
