#!/usr/bin/env python3
"""
Enhanced WAF Testing Pipeline for Phase 3
Tests model outputs against multiple WAF engines and rules
"""

import json
import requests
import subprocess
import docker
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class WAFTestResult:
    payload: str
    waf_engine: str
    blocked: bool
    response_code: int
    detection_rules: List[str]
    bypass_score: float

class ModernWAFTester:
    """Enhanced WAF testing with multiple engines"""
    
    def __init__(self):
        self.waf_engines = {
            "modsecurity": self.test_modsecurity,
            "cloudflare": self.test_cloudflare_simulation,
            "aws_waf": self.test_aws_waf_simulation,
            "akamai": self.test_akamai_simulation
        }
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup WAF testing containers"""
        self.client = docker.from_env()
        
        # Start ModSecurity container if not running
        try:
            container = self.client.containers.get("modsec-test")
            if container.status != "running":
                container.start()
        except:
            print("🚀 Starting ModSecurity container...")
            self.client.containers.run(
                "owasp/modsecurity-crs:apache",
                ports={"80/tcp": 8080},
                name="modsec-test",
                detach=True,
                remove=False
            )
            time.sleep(10)  # Wait for container to start
    
    def test_modsecurity(self, payload: str) -> WAFTestResult:
        """Test against ModSecurity + OWASP CRS"""
        try:
            # Test XSS payloads
            if "script" in payload.lower() or "<" in payload:
                response = requests.get(
                    "http://localhost:8080/",
                    params={"search": payload},
                    timeout=5
                )
            # Test SQLi payloads
            else:
                response = requests.post(
                    "http://localhost:8080/login",
                    data={"username": payload, "password": "test"},
                    timeout=5
                )
            
            blocked = response.status_code in [403, 406, 410]
            detection_rules = self.extract_modsec_rules(response)
            
            return WAFTestResult(
                payload=payload,
                waf_engine="modsecurity",
                blocked=blocked,
                response_code=response.status_code,
                detection_rules=detection_rules,
                bypass_score=0.0 if blocked else 1.0
            )
            
        except Exception as e:
            return WAFTestResult(payload, "modsecurity", True, 500, [str(e)], 0.0)
    
    def test_cloudflare_simulation(self, payload: str) -> WAFTestResult:
        """Simulate Cloudflare WAF rules"""
        blocked = False
        detection_rules = []
        
        # Cloudflare XSS rules
        xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<img[^>]+onerror",
            r"expression\s*\(",
            r"document\.cookie",
            r"alert\s*\(",
            r"prompt\s*\(",
            r"confirm\s*\("
        ]
        
        # Cloudflare SQLi rules  
        sqli_patterns = [
            r"union\s+select",
            r"1\s*=\s*1",
            r"'\s*or\s*'",
            r"--\s*$",
            r"\/\*.*\*\/",
            r"benchmark\s*\(",
            r"sleep\s*\(",
            r"waitfor\s+delay"
        ]
        
        import re
        all_patterns = xss_patterns + sqli_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                blocked = True
                detection_rules.append(f"CF_{pattern}")
        
        return WAFTestResult(
            payload=payload,
            waf_engine="cloudflare",
            blocked=blocked,
            response_code=403 if blocked else 200,
            detection_rules=detection_rules,
            bypass_score=0.0 if blocked else 1.0
        )
    
    def test_aws_waf_simulation(self, payload: str) -> WAFTestResult:
        """Simulate AWS WAF managed rules"""
        blocked = False
        detection_rules = []
        
        # AWS Core Rule Set simulation
        aws_rules = {
            "AWSManagedRulesCommonRuleSet": [
                r"<script",
                r"javascript:",
                r"on\w+\s*=",
                r"union.*select",
                r"1=1",
                r"'\s*or\s*'"
            ],
            "AWSManagedRulesKnownBadInputsRuleSet": [
                r"\.\.\/",
                r"etc\/passwd",
                r"cmd\.exe",
                r"powershell"
            ],
            "AWSManagedRulesSQLiRuleSet": [
                r"select.*from",
                r"insert.*into", 
                r"delete.*from",
                r"update.*set",
                r"union.*select"
            ]
        }
        
        import re
        for rule_set, patterns in aws_rules.items():
            for pattern in patterns:
                if re.search(pattern, payload, re.IGNORECASE):
                    blocked = True
                    detection_rules.append(f"{rule_set}_{pattern}")
        
        return WAFTestResult(
            payload=payload,
            waf_engine="aws_waf",
            blocked=blocked, 
            response_code=403 if blocked else 200,
            detection_rules=detection_rules,
            bypass_score=0.0 if blocked else 1.0
        )
    
    def test_akamai_simulation(self, payload: str) -> WAFTestResult:
        """Simulate Akamai Kona Site Defender"""
        blocked = False
        detection_rules = []
        
        # Akamai behavioral analysis simulation
        risk_score = 0
        
        # Check for encoded payloads
        if any(enc in payload for enc in ['%3C', '%3E', '%22', '%27']):
            risk_score += 20
            detection_rules.append("ENCODED_CHARS")
        
        # Check for common evasion techniques
        if len(payload) > 100:
            risk_score += 15
            detection_rules.append("LONG_PAYLOAD")
            
        if payload.count('(') > 3 or payload.count('<') > 2:
            risk_score += 25
            detection_rules.append("SUSPICIOUS_CHARS")
        
        # Akamai blocks if risk > 50
        blocked = risk_score > 50
        
        return WAFTestResult(
            payload=payload,
            waf_engine="akamai",
            blocked=blocked,
            response_code=403 if blocked else 200, 
            detection_rules=detection_rules,
            bypass_score=0.0 if blocked else 1.0
        )
    
    def extract_modsec_rules(self, response) -> List[str]:
        """Extract triggered ModSecurity rule IDs"""
        rules = []
        if 'X-ModSecurity-Rule-ID' in response.headers:
            rules = response.headers['X-ModSecurity-Rule-ID'].split(',')
        return rules
    
    def test_payload_comprehensive(self, payload: str) -> Dict[str, WAFTestResult]:
        """Test payload against all WAF engines"""
        results = {}
        
        for engine_name, test_func in self.waf_engines.items():
            try:
                result = test_func(payload)
                results[engine_name] = result
                print(f"✓ {engine_name}: {'BLOCKED' if result.blocked else 'BYPASSED'}")
            except Exception as e:
                print(f"✗ {engine_name}: ERROR - {e}")
                results[engine_name] = WAFTestResult(
                    payload, engine_name, True, 500, [str(e)], 0.0
                )
        
        return results
    
    def calculate_overall_bypass_score(self, results: Dict[str, WAFTestResult]) -> float:
        """Calculate overall bypass effectiveness"""
        if not results:
            return 0.0
            
        bypass_scores = [r.bypass_score for r in results.values()]
        return sum(bypass_scores) / len(bypass_scores)

def test_model_outputs_against_waf(model_outputs_file: str, output_file: str):
    """Test all model outputs against WAF engines"""
    
    tester = ModernWAFTester()
    results = []
    
    print("🛡️ Starting WAF bypass testing...")
    
    with open(model_outputs_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line.strip())
            payload = data.get('payload', data.get('output', ''))
            
            if not payload or payload.strip() == '':
                continue
                
            print(f"\n📝 Testing payload {line_num}: {payload[:50]}...")
            
            # Test against all WAF engines
            waf_results = tester.test_payload_comprehensive(payload)
            overall_score = tester.calculate_overall_bypass_score(waf_results)
            
            # Compile results
            result = {
                "line_number": line_num,
                "instruction": data.get('instruction', ''),
                "payload": payload,
                "waf_results": {name: {
                    "blocked": r.blocked,
                    "response_code": r.response_code,
                    "detection_rules": r.detection_rules,
                    "bypass_score": r.bypass_score
                } for name, r in waf_results.items()},
                "overall_bypass_score": overall_score,
                "bypass_effectiveness": "EXCELLENT" if overall_score > 0.8 else
                                     "GOOD" if overall_score > 0.6 else
                                     "FAIR" if overall_score > 0.4 else "POOR"
            }
            
            results.append(result)
            
            # Progress update
            if line_num % 10 == 0:
                avg_score = sum(r['overall_bypass_score'] for r in results) / len(results)
                print(f"🎯 Progress: {line_num} payloads tested, avg bypass: {avg_score:.3f}")
    
    # Save comprehensive results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_payloads": len(results),
                "waf_engines": list(tester.waf_engines.keys()),
                "avg_bypass_score": sum(r['overall_bypass_score'] for r in results) / len(results),
                "test_timestamp": time.strftime("%Y%m%d_%H%M%S")
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total = len(results)
    avg_score = sum(r['overall_bypass_score'] for r in results) / total
    excellent = sum(1 for r in results if r['overall_bypass_score'] > 0.8)
    
    print(f"\n🏆 WAF BYPASS TESTING COMPLETE:")
    print(f"   Total Payloads: {total}")
    print(f"   Average Bypass Score: {avg_score:.3f}")
    print(f"   Excellent Bypasses: {excellent} ({excellent/total*100:.1f}%)")
    print(f"   Results saved: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python enhanced_waf_test.py <model_outputs.jsonl> <results.json>")
        sys.exit(1)
    
    test_model_outputs_against_waf(sys.argv[1], sys.argv[2])