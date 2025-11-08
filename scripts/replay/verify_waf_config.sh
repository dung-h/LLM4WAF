#!/bin/bash
# Verify WAF is in blocking mode

echo "=== Step 1: WAF Configuration Verification ==="

echo -e "\n1.1: Checking for BLOCKING-EVALUATION rules..."
docker exec waf_nginx_modsec sh -c 'grep -R "REQUEST-949-BLOCKING-EVALUATION" /etc/modsecurity/crs/rules/ 2>/dev/null | head -5'

echo -e "\n1.2: Checking anomaly score thresholds..."
docker exec waf_nginx_modsec sh -c 'grep -R "anomaly_score_threshold" /etc/modsecurity/ 2>/dev/null | grep -v "^#"'

echo -e "\n1.3: Checking SecRuleEngine..."
docker exec waf_nginx_modsec sh -c 'grep -R "SecRuleEngine" /etc/modsecurity/ /etc/nginx/ 2>/dev/null | grep -v "^#" | head -5'

echo -e "\n1.4: Checking current paranoia level..."
docker exec waf_nginx_modsec sh -c 'grep -R "paranoia_level" /etc/modsecurity/ 2>/dev/null | grep -v "^#"'

echo -e "\n1.5: Checking if rules are actually loaded..."
docker exec waf_nginx_modsec sh -c 'ls -la /etc/modsecurity/crs/rules/ | head -20'
