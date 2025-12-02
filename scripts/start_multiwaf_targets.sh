#!/bin/bash

echo "[*] Starting Multi-WAF Test Environment..."

# 1. Create network if not exists
docker network inspect waf-network >/dev/null 2>&1 || \
    docker network create waf-network

# 2. Start containers (ModSecurity, Coraza + Apps)
echo "[*] Running docker-compose up..."
docker-compose -f docker-compose.multiwaf.yml up -d --build --remove-orphans

# 3. Check status
echo "[*] Checking container status..."
sleep 5
docker-compose -f docker-compose.multiwaf.yml ps

echo "[*] Multi-WAF Environment Started."
echo "    - ModSecurity: http://localhost:8000"
echo "    - Coraza:      http://localhost:9005"
echo "    - Apps:        /dvwa/, /juice/"