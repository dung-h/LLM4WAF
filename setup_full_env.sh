#!/bin/bash
set -e

echo "🚀 Initializing LLM4WAF Remote Environment (GPU Optimized)..."

# 1. System dependencies
echo "📦 Installing system packages..."
if command -v sudo &> /dev/null; then
    SUDO="sudo"
else
    SUDO=""
fi
$SUDO apt-get update && $SUDO apt-get install -y git python3-pip python3-venv unzip curl wget build-essential

# 2. Install Docker (if missing)
if ! command -v docker &> /dev/null; then
    echo "🐳 Docker not found. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    $SUDO sh get-docker.sh
    rm get-docker.sh
else
    echo "✅ Docker is already installed."
fi

# 3. Install Docker Compose
echo "🐳 Installing Docker Compose..."
$SUDO apt-get install -y docker-compose-plugin || true

# 4. Python Virtual Environment
echo "🐍 Setting up Python venv..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 5. Install Python Libraries
echo "📚 Installing Python dependencies..."
pip install --upgrade pip

# Install Torch with CUDA support explicitly first
# Change cu121 to cu118 if your server driver is older
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies from requirements
pip install -r requirements.txt

echo "⚡ Installing Flash Attention 2 (Optional, takes time)..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention install failed (non-critical), continuing..."

# 6. Directory Structure
echo "Tk Creating project directories..."
mkdir -p experiments/remote_adapters
mkdir -p data/processed
mkdir -p logs

# 7. Start WAF Targets
echo "🛡️ Starting WAF Targets (Docker)..."
if [ -f "docker-compose.multiwaf.yml" ]; then
    # Ensure docker permissions
    if [ "$EUID" -ne 0 ] && ! groups | grep -q "docker"; then
        DOCKER_CMD="$SUDO docker"
    else
        DOCKER_CMD="docker"
    fi

    $DOCKER_CMD compose -f docker-compose.multiwaf.yml up -d
    echo "✅ WAF Containers started."
else
    echo "⚠️ docker-compose.multiwaf.yml not found! Skipping WAF setup."
fi

echo "----------------------------------------------------------------"
echo "✅ Environment Ready!"
echo "👉 Run: source .venv/bin/activate"
echo "👉 To download adapters: check GEMINI-FAKE.md for gdown links."
echo "👉 To start training:"
echo "   nohup python scripts/train_red.py --config configs/red_phase3_adaptive_gemma.yaml > logs/train_gemma.log 2>&1 &"
echo "----------------------------------------------------------------"