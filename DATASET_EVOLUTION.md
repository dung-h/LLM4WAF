# Dataset Evolution Summary

## Version History

### v18-v25 (Legacy)

- **v18-v23**: Various unified datasets with reasoning subsets
- **v24**: 12MB unified dataset (MySQL + XSS)
- **v25**: 12MB Phi-3 formatted dataset
- **Status**: Archived, superseded by v29

### v26 (Vietnamese Mixed - DEPRECATED)

- **Records**: 139,588
- **Issue**: Mixed Vietnamese/English text, 58,231 unknown results
- **Action**: Cleaned and tested → v27

### v27 (English Cleaned)

- **Records**: 139,588 (52% cleaned from Vietnamese)
- **Passed**: 770
- **Blocked**: 138,678
- **Unknown**: 140
- **Method**: WAF tested all unknown records (40 workers, ~2 min)
- **File**: Archived to `archive/red_v27_cleaned_waf_tested.jsonl`

### v28 (Merged + Deduped)

- **Records**: 11,171 (92.6% reduction from v27 due to duplicates)
- **Passed**: 893
- **Source**: v27 + cleaned seeds (filtered 99.5% garbage from 215k seeds)
- **File**: Archived to `archive/red_v28_merged_normalized.jsonl`

### v29 (CURRENT - Final Enriched) ✅

- **Records**: 11,692
- **Passed**: 1,414 (12.1%)
- **Blocked**: 10,258 (87.7%)
- **Unknown**: 20 (0.2%)
- **Sources**: v28 + extracted from v19-v23 historical datasets
- **File**: `red_v29_enriched.jsonl`
- **Quality**: 100% English, deduplicated, WAF validated

### v30 (Mutation Attempt - FAILED)

- **Method**: Local mutation (URL encoding, Unicode, comment injection, etc.)
- **Generated**: 2,290 mutations from 800 blocked payloads
- **WAF Test**: 0/2,290 passed (100% blocked)
- **Conclusion**: WAF too strict for simple mutations
- **File**: Archived to `archive/testing/red_v30_final.jsonl` (identical to v29)

---

## Repository Sources (23 Total)

### Round 1 - Legacy (7 repos)

1. `swisskyrepo/PayloadsAllTheThings`
2. `fuzzdb-project/fuzzdb`
3. `danielmiessler/SecLists`
4. `1N3/IntruderPayloads`
5. `payloadbox/sql-injection-payload-list`
6. `payloadbox/xss-payload-list`
7. `humblelad/Awesome-XSS-Payloads`

**Result**: Extracted 487 files → Contributed to v26-v29

### Round 2 - First Expansion (8 repos)

8. `OWASP/WebGoat`
9. `PortSwigger/xss-cheatsheet-data` (302 payloads extracted)
10. `sqlmapproject/sqlmap` (773 SQLi payloads)
11. `beefproject/beef` (28 payloads)
12. `s0md3v/XSStrike` (definitions.json - 0 valid)
13. `commixproject/commix` (command injection - 0 valid)
14. `khalilbijjou/WAFNinja` (SQLite db - 125 payloads)
15. `mandatoryprogrammer/xsshunter-express` (0 payloads)

**Extracted**: 1,228 total → **0 passed WAF** (100% blocked)

### Round 3 - Second Expansion (8 repos)

16. `OWASP/Nettacker`
17. `ron190/jsql-injection`
18. `mandatoryprogrammer/xsshunter`
19. `projectdiscovery/nuclei-templates`
20. `Bo0oM/fuzz.txt` (14 payloads)
21. `TakSec/Injectx`
22. `assetnote/commonspeak2-wordlists`
23. `nixawk/pentest-wiki`

**Extracted**: 1,161 unique → **0 passed WAF** (100% blocked)

---

## WAF Testing Statistics

### ModSecurity WAF Configuration

- **URL**: http://localhost:8080 (DVWA)
- **Rules**: OWASP CRS
- **Workers**: 40 parallel
- **Speed**: ~500-550 payloads/second
- **Detection**: Payload reflected in response = passed

### Testing Results Summary

| Source          | Payloads Tested | Passed    | Pass Rate |
| --------------- | --------------- | --------- | --------- |
| v26 unknown     | 58,231          | 770       | 1.3%      |
| v27-v29 merged  | 11,692          | 1,414     | 12.1%     |
| Local mutations | 2,290           | 0         | 0.0%      |
| New repos R2    | 1,228           | 0         | 0.0%      |
| New repos R3    | 1,161           | 0         | 0.0%      |
| **TOTAL**       | **74,602**      | **1,414** | **1.9%**  |

### Key Findings

- **WAF Strictness**: Only 1.9% overall pass rate across all sources
- **Legacy repos** (R1) provided all 1,414 passed payloads
- **New repos** (R2+R3, 16 repos) contributed 0 passed payloads
- **Mutations** failed completely (0% success)
- **Seed quality**: 99.5% of "passed" seeds were plain text garbage

---

## File Organization

### Active Files (`data/processed/`)

- `red_v29_enriched.jsonl` - **Current production dataset** (11,692 records)
- `red_v19_unified_mysql_xss.jsonl` - Historical v19 (456 passed)
- `red_v20_unified_mysql_xss.jsonl` - Historical v20 (888 passed)
- `red_v21-v23_unified_mysql_xss_balanced.jsonl` - Historical balanced sets
- `used_payload_repos.txt` - List of processed repositories

### Archived Files (`data/processed/archive/`)

- `red_v26_*.jsonl` - Vietnamese mixed versions
- `red_v27_cleaned_waf_tested.jsonl` - 56MB cleaned version
- `red_v28_merged_normalized.jsonl` - 3.8MB merged version
- `seeds/` - 537 seed files archived (most were garbage)
- `testing/` - v30 mutation attempts, new repo test results

---

## Lessons Learned

1. **Seed Quality**: Always validate "passed" labels - 99.5% of seeds were plain text
2. **Deduplication Critical**: v26 139k → v28 11k (92.6% duplicates)
3. **WAF Strictness**: ModSecurity OWASP CRS extremely effective (98% block rate)
4. **New Repos Ineffective**: 16 new repos, 2,389 payloads tested → 0 passed
5. **Mutation Failures**: Simple encoding/obfuscation 100% blocked by modern WAF
6. **Language Cleaning**: 52% of v26 had Vietnamese text requiring translation
7. **Historical Value**: v19-v23 contributed 521 additional passed payloads

---

## Next Steps Options

1. **Accept 1,414 passed** - Train with current high-quality dataset
2. **Relax WAF** - Lower ModSecurity ParanoidLevel to allow ~600 more passed
3. **LLM Mutation Retry** - Fix DeepSeek API and regenerate advanced mutations
4. **Manual Crafting** - Security experts create 586 novel bypasses
5. **Hybrid Approach** - Slight WAF relaxation (10% → 15%) + LLM mutations

**Recommended**: Option 1 (proceed with 1,414) or Option 5 (hybrid approach)
