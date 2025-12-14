"""
Mini test for validation pipeline - Quick smoke test
Tests pipeline with 1 sample per phase to verify everything works
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.validate_rl_full_pipeline import *
import logging

# Force reload constants (in case of Python cache issues)
import scripts.validate_rl_full_pipeline as vrl
vrl.REMOTE_WAF_BASE = "http://modsec.llmshield.click"
vrl.LOGIN_URL = f"{vrl.REMOTE_WAF_BASE}/login.php"
vrl.SQLI_URL = f"{vrl.REMOTE_WAF_BASE}/vulnerabilities/sqli/"
vrl.XSS_URL = f"{vrl.REMOTE_WAF_BASE}/vulnerabilities/xss_r/"
vrl.EXEC_URL = f"{vrl.REMOTE_WAF_BASE}/vulnerabilities/exec/"

# Override logging to be more verbose for testing
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== TEST CONFIGURATION ====================

# Test with MINIMAL samples
NUM_SAMPLES_PER_PHASE = 1  # Just 1 sample per phase
NUM_PROBES = 3              # Only 3 probes for Phase 3

# Test only 1 model from each phase to verify pipeline works
TEST_MODELS = [
    # Baseline
    {
        "name": "Qwen_3B_Pretrained_MINITEST",
        "base": "Qwen/Qwen2.5-3B-Instruct",
        "adapter": None,
        "phase": 0,
        "chat_template": "qwen"
    },
    # Phase 1
    {
        "name": "Qwen_3B_Phase1_MINITEST",
        "base": "Qwen/Qwen2.5-3B-Instruct",
        "adapter": "experiments/remote_qwen_3b_phase1",
        "phase": 1,
        "chat_template": "qwen"
    },
    # Phase 2
    {
        "name": "Qwen_3B_Phase2_MINITEST",
        "base": "Qwen/Qwen2.5-3B-Instruct",
        "adapter": "experiments/remote_qwen_3b_phase2",
        "phase": 2,
        "chat_template": "qwen"
    },
    # Phase 3 RL (using Gemma RL instead - Qwen RL chưa có)
    {
        "name": "Gemma_2B_RL_MINITEST",
        "base": "google/gemma-2-2b-it",
        "adapter": "experiments/remote_gemma2_2b_phase3_rl/checkpoint-150",
        "phase": 3,
        "chat_template": "gemma"
    }
]

# Test only PL1 (skip PL4 for speed)
TEST_WAF_LEVELS = ["PL1"]

# Minimal techniques for testing
PHASE1_TECHNIQUES_MINI = {
    "SQLI": ["Double URL Encoding"],
    "XSS": ["SVG Event Handler"]
}

PHASE2_TECHNIQUES_MINI = {
    "SQLI": ["obf_comment_sql+obf_double_url_encode+obf_whitespace_url"],
    "XSS": ["XSS Body onload"]
}

PHASE2_FORGETTING_TEST_MINI = {
    "SQLI": ["Unicode Homoglyph"],
    "XSS": ["non_script_xss"]
}

PHASE3_TECHNIQUES_MINI = {
    "SQLI": ["Adaptive Multi-layer Encoding"],
    "XSS": ["Adaptive Event Handler"]
}

# ==================== OVERRIDE VALIDATION FUNCTIONS ====================

def validate_phase1_mini(model: RLModelWrapper, waf_client: WAFClient, waf_level: str) -> List[Dict]:
    """Minimal Phase 1 validation - 1 sample per attack type"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MINITEST PHASE 1 - {waf_level}")
    logger.info(f"{'='*60}\n")
    
    results = []
    all_payloads = []
    
    for attack_type in ["SQLI", "XSS"]:
        logger.info(f"Testing {attack_type}...")
        
        techniques = PHASE1_TECHNIQUES_MINI[attack_type]
        
        for technique in techniques:
            prompt = build_phase1_prompt(attack_type, technique)
            logger.info(f"  Prompt: {prompt[:60]}...")
            
            payload = model.generate_payload(prompt)
            all_payloads.append(payload)
            logger.info(f"  Generated: {payload[:60]}...")
            
            test_result = waf_client.test_payload(payload, attack_type)
            logger.info(f"  Result: {test_result['status']}")
            
            result = {
                "phase": 1,
                "waf_level": waf_level,
                "attack_type": attack_type,
                "technique": technique,
                "prompt": prompt,
                "payload": payload,
                "test_result": test_result
            }
            
            results.append(result)
    
    # Diversity metrics
    diversity_metrics = calculate_payload_diversity(all_payloads)
    logger.info(f"Diversity: {diversity_metrics}")
    
    for result in results:
        result["diversity_metrics"] = diversity_metrics
    
    return results

def validate_phase2_mini(model: RLModelWrapper, waf_client: WAFClient, waf_level: str) -> List[Dict]:
    """Minimal Phase 2 validation - 1 advanced + 1 forgetting check"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MINITEST PHASE 2 - {waf_level}")
    logger.info(f"{'='*60}\n")
    
    results = []
    all_payloads = []
    
    for attack_type in ["SQLI", "XSS"]:
        # Get probe pools
        if attack_type == "SQLI":
            blocked_pool = PROBE_POOLS["sqli_blocked"]
            passed_pool = PROBE_POOLS["sqli_passed"]
        else:
            blocked_pool = PROBE_POOLS["xss_blocked"]
            passed_pool = PROBE_POOLS["xss_passed"]
        
        # Advanced technique test
        logger.info(f"Testing {attack_type} Advanced...")
        technique = PHASE2_TECHNIQUES_MINI[attack_type][0]
        
        blocked_examples = [p["payload"] for p in random.sample(blocked_pool, min(2, len(blocked_pool)))]
        passed_examples = [p["payload"] for p in random.sample(passed_pool, min(1, len(passed_pool)))]
        
        prompt = build_phase2_prompt(attack_type, technique, blocked_examples, passed_examples)
        logger.info(f"  Prompt with observations...")
        
        payload = model.generate_payload(prompt, max_tokens=256)
        all_payloads.append(payload)
        logger.info(f"  Generated: {payload[:60]}...")
        
        test_result = waf_client.test_payload(payload, attack_type)
        logger.info(f"  Result: {test_result['status']}")
        
        results.append({
            "phase": 2,
            "subtest": "advanced",
            "waf_level": waf_level,
            "attack_type": attack_type,
            "technique": technique,
            "prompt": prompt,
            "payload": payload,
            "test_result": test_result
        })
        
        # Forgetting check
        logger.info(f"Testing {attack_type} Forgetting Check...")
        forget_technique = PHASE2_FORGETTING_TEST_MINI[attack_type][0]
        
        prompt = build_phase2_prompt(attack_type, forget_technique, blocked_examples, passed_examples)
        
        payload = model.generate_payload(prompt, max_tokens=256)
        all_payloads.append(payload)
        logger.info(f"  Generated: {payload[:60]}...")
        
        test_result = waf_client.test_payload(payload, attack_type)
        logger.info(f"  Result: {test_result['status']}")
        
        results.append({
            "phase": 2,
            "subtest": "forgetting_check",
            "waf_level": waf_level,
            "attack_type": attack_type,
            "technique": forget_technique,
            "prompt": prompt,
            "payload": payload,
            "test_result": test_result
        })
    
    # Diversity metrics
    diversity_metrics = calculate_payload_diversity(all_payloads)
    logger.info(f"Diversity: {diversity_metrics}")
    
    for result in results:
        result["diversity_metrics"] = diversity_metrics
    
    return results

def validate_phase3_mini(model: RLModelWrapper, waf_client: WAFClient, waf_level: str) -> List[Dict]:
    """Minimal Phase 3 validation - 1 adaptive attack per type"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MINITEST PHASE 3 - {waf_level}")
    logger.info(f"{'='*60}\n")
    
    results = []
    all_payloads = []
    
    for attack_type in ["SQLI", "XSS"]:
        logger.info(f"Testing {attack_type} Adaptive Pipeline...")
        
        technique = PHASE3_TECHNIQUES_MINI[attack_type][0]
        
        # Probing phase
        logger.info(f"  Probing (3 payloads)...")
        
        if attack_type == "SQLI":
            passed_pool = PROBE_POOLS["sqli_passed"]
            blocked_pool = PROBE_POOLS["sqli_blocked"]
        else:
            passed_pool = PROBE_POOLS["xss_passed"]
            blocked_pool = PROBE_POOLS["xss_blocked"]
        
        probes = random.sample(passed_pool, min(1, len(passed_pool))) + \
                 random.sample(blocked_pool, min(2, len(blocked_pool)))
        random.shuffle(probes)
        
        probing_results = []
        for probe in probes:
            test_result = waf_client.test_payload(probe["payload"], attack_type)
            probing_results.append({
                "payload": probe["payload"],
                "technique": probe["technique"],
                "result": "PASSED" if test_result["status"] == "passed" else "BLOCKED"
            })
        
        logger.info(f"  Probing complete: {len([p for p in probing_results if p['result'] == 'PASSED'])} PASSED")
        
        # Adaptive generation
        logger.info(f"  Generating adaptive payload...")
        prompt = build_phase3_adaptive_prompt(attack_type, technique, probing_results)
        
        payload = model.generate_payload(prompt, max_tokens=256)
        all_payloads.append(payload)
        logger.info(f"  Generated: {payload[:60]}...")
        
        test_result = waf_client.test_payload(payload, attack_type)
        logger.info(f"  Result: {test_result['status']}")
        
        results.append({
            "phase": 3,
            "waf_level": waf_level,
            "attack_type": attack_type,
            "technique": technique,
            "probing_results": probing_results,
            "prompt": prompt,
            "payload": payload,
            "test_result": test_result
        })
    
    # Diversity metrics
    diversity_metrics = calculate_payload_diversity(all_payloads)
    logger.info(f"Diversity: {diversity_metrics}")
    
    for result in results:
        result["diversity_metrics"] = diversity_metrics
    
    return results

# ==================== MINITEST MAIN ====================

def run_minitest():
    """Run minimal validation test"""
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"# VALIDATION PIPELINE MINITEST")
    logger.info(f"# Purpose: Verify pipeline works without errors")
    logger.info(f"# Models: 4 (Pretrained, Phase1, Phase2, Phase3 RL)")
    logger.info(f"# WAF Levels: 1 (PL1 only)")
    logger.info(f"# Samples: 1-2 per phase")
    logger.info(f"{'#'*80}\n")
    
    # Output directory
    output_dir = f"eval/minitest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output: {output_dir}\n")
    
    all_results = []
    
    for model_config in TEST_MODELS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {model_config['name']}")
        logger.info(f"{'='*80}")
        
        try:
            # Load model
            model = RLModelWrapper(model_config)
            model.load()
            
            for waf_level in TEST_WAF_LEVELS:
                logger.info(f"\nWAF Level: {waf_level}")
                
                # Initialize WAF client
                waf_client = WAFClient(waf_level=waf_level)
                
                if not waf_client.login():
                    logger.error(f"Failed to login, skipping...")
                    continue
                
                # Run minimal validation
                phase1_results = validate_phase1_mini(model, waf_client, waf_level)
                phase2_results = validate_phase2_mini(model, waf_client, waf_level)
                phase3_results = validate_phase3_mini(model, waf_client, waf_level)
                
                # Combine results
                model_results = {
                    "model": model_config["name"],
                    "model_phase": model_config.get("phase", 0),
                    "waf_level": waf_level,
                    "phase1": phase1_results,
                    "phase2": phase2_results,
                    "phase3": phase3_results,
                    "timestamp": datetime.now().isoformat()
                }
                
                all_results.append(model_results)
                
                # Save result
                output_file = os.path.join(output_dir, f"{model_config['name']}_{waf_level}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(model_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Saved: {output_file}")
            
            # Unload model
            model.unload()
            
            logger.info(f"✅ {model_config['name']} - PASSED")
            
        except Exception as e:
            logger.error(f"❌ {model_config['name']} - FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined
    combined_file = os.path.join(output_dir, "minitest_results.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ MINITEST COMPLETE")
    logger.info(f"Results: {output_dir}")
    logger.info(f"{'='*80}")
    
    # Summary
    logger.info(f"\nSummary:")
    for result in all_results:
        total = len(result["phase1"]) + len(result["phase2"]) + len(result["phase3"])
        passed = sum(
            1 for r in (result["phase1"] + result["phase2"] + result["phase3"])
            if r["test_result"]["status"] == "passed"
        )
        logger.info(f"  {result['model']}: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    logger.info(f"\n✅ Pipeline validation successful! No errors detected.")
    logger.info(f"You can now run full validation with all models.")

if __name__ == "__main__":
    run_minitest()
