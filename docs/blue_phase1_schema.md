
# Blue Episode Schema v1 - Phase 1 Data & KB Preparation

This document defines the unified JSON schema for a `blue_episode` object, which represents a single observed attack attempt and its outcome against a WAF-protected application. This schema is designed to store normalized attack logs and WAF/application observations, making them ready for RAG (Retrieval-Augmented Generation) and fine-tuning a BLUE LLM.

## Schema Definition

A `blue_episode` is a JSON object with the following structure:

```jsonc
{
  "waf_env": {
    "engine": "modsecurity",               // "modsecurity" | "coraza" | "naxsi" | "safeline" ...
    "profile": "owasp_crs",
    "crs_version": "3.3.5",
    "paranoia_level": 1,
    "anomaly_threshold": 5,
    "extra_config": {
      "custom_rule_files": ["custom/my_rules.conf"]
    }
  },

  "app_context": {
    "app_name": "dvwa",                    // Name of the target application: "dvwa", "juice_shop", "bwapp", etc.
    "endpoint": "/vulnerabilities/sqli/",  // Specific endpoint attacked (e.g., path in URL)
    "http_method": "GET",                  // HTTP method used (GET, POST)
    "injection_point": "query_param",      // Where the payload was injected: "query_param", "body_param", "header", etc.
    "tech_stack": "php_mysql"              // Technology stack of the application: "php_mysql", "node_mongodb", "java", etc.
  },

  "attack": {
    "attack_type": "SQLI",                 // General attack category: "SQLI", "XSS", "OS_INJECTION", "attack-disclosure", "RCE", etc.
    "technique": "obf_double_url_encode+case_random", // Specific obfuscation or attack technique used
    "payload": "%2527%2520AnD%2520...",    // The actual payload string sent
    "red_model": "gemma2_2b_phase3_rl",    // The RED model that generated this payload
    "phase": "rl_phase3",                  // Phase of RED model training (sft_phase1, sft_phase2, rl_phase3)
    "timestamp": "2025-12-02T12:34:56Z"    // Timestamp of the attack attempt
  },

  "waf_observation": {
    "blocked": false,                      // true if WAF blocked the request (e.g., HTTP 403)
    "http_status": 200,                    // HTTP status code returned by the WAF/backend
    "anomaly_score": 0,                    // ModSecurity anomaly score (0 if not available/applicable)
    "matched_rules": [],                   // List of WAF rules that were triggered (e.g., ["942100", "941100"])
    "raw_waf_log": null                    // Optional: Raw snippet of WAF audit log for deep analysis (can be null/"" if not captured)
  },

  "app_observation": {
    "http_status": 500,                    // HTTP status code returned by the *application* (after passing WAF)
    "resp_body_snippet": "SyntaxError (/path/to/your/app/controllers/reflects_controller.rb:10: ...)", // Snippet of the response body, especially useful for error messages or reflected content
    "error_class": "Ruby::SyntaxError",    // Type of application error detected
    "stack_trace_lang": "ruby"             // Language of the stack trace if detected
  },

  "blue_label": {
    "vuln_effect": "error-disclosure",     // "sql-error", "xss-reflected", "xss-stored", "no-effect", ...
    "is_false_negative": true,            // true nếu WAF nên chặn nhưng không chặn
    "expected_crs_rules": ["956100"],     // có thể trống nếu chưa map được
    "severity": "high"                     // Severity of the attack effect: "low", "medium", "high", "critical"
  }
}
```

## Field Explanations:

### `waf_env` (WAF Environment Context)
Describes the WAF environment the attack was tested against.
- `engine`: The WAF engine in use.
- `profile`: The ruleset profile used (e.g., OWASP CRS).
- `crs_version`: Version of CRS.
- `paranoia_level`: ModSecurity Paranoia Level (PL).
- `anomaly_threshold`: Anomaly score threshold for blocking (ModSecurity).
- `extra_config`: Any additional WAF-specific configurations.

### `app_context` (Application Context)
Details about the target application.
- `app_name`: Name of the web application.
- `endpoint`: The specific vulnerable path.
- `http_method`: HTTP method (GET/POST).
- `injection_point`: Location of payload injection (query param, body param, etc.).
- `tech_stack`: Technology stack of the app (e.g., `php_mysql` for DVWA).

### `attack` (Attack Details)
Information about the payload and its origin.
- `attack_type`: General category of the attack.
- `technique`: Specific method or obfuscation technique.
- `payload`: The generated payload.
- `red_model`: The RED model that generated this payload.
- `phase`: The training phase of the RED model.
- `timestamp`: When the attack occurred.

### `waf_observation` (WAF's Reaction)
What the WAF observed or did.
- `blocked`: `true` if the WAF actively blocked the request.
- `http_status`: HTTP status code from the WAF or the proxied response.
- `anomaly_score`: ModSecurity's calculated anomaly score.
- `matched_rules`: List of WAF rules that were triggered (e.g., ["942100", "941100"]).
- `raw_waf_log`: Snippet from the WAF audit log (if available).

### `app_observation` (Application's Reaction)
What happened on the application side after the request passed the WAF.
- `http_status`: HTTP status code from the application (e.g., 200, 500).
- `resp_body_snippet`: A snippet of the application's response body, especially useful for error messages or reflected content.
- `error_class`: Type of application error detected.
- `stack_trace_lang`: Language of the stack trace if present.

### `blue_label` (Annotated Label for BLUE LLM)
The "ground truth" labels for the BLUE LLM to learn from.
- `vuln_effect`: Categorization of the vulnerability's impact.
- `is_false_negative`: `true` if the WAF *should* have blocked but allowed the attack to succeed/cause an effect.
- `expected_crs_rules`: Recommended CRS rules to block this type of attack.
- `severity`: Severity rating of the effect.

## Example Blue Episodes:

### Example 1: SQL Error Disclosure (False Negative)
```json
{
  "waf_env": {
    "engine": "modsecurity",
    "profile": "owasp_crs",
    "crs_version": "3.3.5",
    "paranoia_level": 1,
    "anomaly_threshold": 5,
    "extra_config": {}
  },
  "app_context": {
    "app_name": "dvwa",
    "endpoint": "/vulnerabilities/sqli/",
    "http_method": "GET",
    "injection_point": "query_param",
    "tech_stack": "php_mysql"
  },
  "attack": {
    "attack_type": "SQLI",
    "technique": "obf_union_select",
    "payload": "1' UNION SELECT @@version, USER(), DATABASE() --",
    "red_model": "gemma2_2b_phase3_rl",
    "phase": "rl_phase3",
    "timestamp": "2025-12-02T12:34:56Z"
  },
  "waf_observation": {
    "blocked": false,
    "http_status": 200,
    "anomaly_score": 0,
    "matched_rules": [],
    "raw_waf_log": null
  },
  "app_observation": {
    "http_status": 200,
    "resp_body_snippet": "Version: 8.0.26, User: root@localhost, DB: dvwa",
    "error_class": null,
    "stack_trace_lang": null
  },
  "blue_label": {
    "vuln_effect": "info-leak",
    "is_false_negative": true,
    "expected_crs_rules": ["942100", "942120"],
    "severity": "medium"
  }
}
```

### Example 2: Reflected XSS (False Negative)
```json
{
  "waf_env": {
    "engine": "coraza",
    "profile": "owasp_crs",
    "crs_version": "3.3.5",
    "paranoia_level": 1,
    "anomaly_threshold": null,
    "extra_config": {}
  },
  "app_context": {
    "app_name": "dvwa",
    "endpoint": "/vulnerabilities/xss_r/",
    "http_method": "GET",
    "injection_point": "query_param",
    "tech_stack": "php"
  },
  "attack": {
    "attack_type": "XSS",
    "technique": "obf_script_tag_encoding",
    "payload": "%3Cscript%3Ealert%281%29%3C%2Fscript%3E",
    "red_model": "gemma2_2b_phase3_rl",
    "phase": "rl_phase3",
    "timestamp": "2025-12-02T12:35:10Z"
  },
  "waf_observation": {
    "blocked": false,
    "http_status": 200,
    "anomaly_score": null,
    "matched_rules": [],
    "raw_waf_log": null
  },
  "app_observation": {
    "http_status": 200,
    "resp_body_snippet": "Hello <script>alert(1)</script>",
    "error_class": null,
    "stack_trace_lang": null
  },
  "blue_label": {
    "vuln_effect": "xss-reflected",
    "is_false_negative": true,
    "expected_crs_rules": ["941100", "941110"],
    "severity": "high"
  }
}
```
