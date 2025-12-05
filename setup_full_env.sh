#!/bin/bash
set -e  # Stop on error

echo "🚀 Starting Full-Stack Environment Setup (Training + Docker WAF)..."

# 1. System Updates
echo "📦 Updating system packages..."
# Check if we have sudo, if not assume root
if command -v sudo &> /dev/null; then
    SUDO="sudo"
else
    SUDO=""
fi

$SUDO apt-get update && $SUDO apt-get install -y git python3-pip python3-venv unzip curl wget

# 2. Install Docker (if missing)
if ! command -v docker &> /dev/null; then
    echo "🐳 Docker not found. Installing..."
    curl -fsSL https://get.docker.sh -o get-docker.sh
    $SUDO sh get-docker.sh
    rm get-docker.sh
else
    echo "✅ Docker is already installed."
fi

# 3. Install Docker Compose
echo "🐳 Installing Docker Compose..."
$SUDO apt-get install -y docker-compose-plugin || true


# 4. Setup Python Virtual Environment
echo "[+] Setting up Python venv..."
python3 -m venv .venv
source .venv/bin/activate

# 5. Install Python Libraries
echo "[+] Installing Python dependencies..."
# Core libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Or cu121 depending on driver version
pip install transformers peft bitsandbytes datasets trl accelerate scipy tensorboard pyyaml httpx tqdm pandas matplotlib seaborn gdown

# Flash Attention (Optional but recommended for speed)
echo "⚡ Installing Flash Attention 2 (This might take a while)..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention install failed (non-critical), continuing..."

# 6. Create Project-specific Directories
echo "[+] Creating project directories..."
mkdir -p experiments/remote_adapters/experiments_remote_optimized
mkdir -p logs
mkdir -p data/rl_stage # For iterative RL if implemented

# 7. Start WAF Targets
echo "🛡️ Starting WAF Targets (Docker)..."
if [ -f "docker-compose.multiwaf.yml" ]; then
    # Ensure docker permissions if not root
    if [ "$EUID" -ne 0 ] && ! groups | grep -q "docker"; then
        echo "⚠️ User not in docker group. Trying with sudo..."
        DOCKER_CMD="$SUDO docker"
    else
        DOCKER_CMD="docker"
    fi

    $DOCKER_CMD compose -f docker-compose.multiwaf.yml down || true
    $DOCKER_CMD compose -f docker-compose.multiwaf.yml up -d
    
    echo "⏳ Waiting for WAF services (15s)..."
    sleep 15
    
    if curl -s http://localhost:8000/login.php | grep "DVWA" > /dev/null; then
        echo "✅ WAF Target (ModSec PL1) is ONLINE at port 8000!"
    else
        echo "⚠️ Warning: Could not reach WAF at port 8000. Check 'docker compose logs'."
    fi
else
    echo "⚠️ docker-compose.multiwaf.yml not found! Skipping WAF setup."
fi

echo "----------------------------------------------------------------"
echo "🎉 Full Remote Setup Complete!"
echo "----------------------------------------------------------------"
echo "Next Steps (on the remote server):"
echo "1. Export HF Token: export HF_TOKEN='hf_YOUR_TOKEN_HERE'"
echo "2. Download adapters from Google Drive using gdown (e.g., gdown YOUR_FILE_ID -O phi3_phase2_adapter.tar.gz)"
echo "3. Extract adapters: tar -xzvf phi3_phase2_adapter.tar.gz -C experiments/remote_adapters/experiments_remote_optimized/"
echo "   (Repeat for Qwen adapter)"
echo "4. Run Training: python scripts/train_rl_reinforce.py --config configs/phi3_mini_phase3_rl.yaml"
echo "----------------------------------------------------------------"