#!/usr/bin/env python3
"""
HuggingFace Authentication Setup for Phase 3 Training
Handles model downloads for Qwen, DeepSeek, LLaMA, Phi-3
"""

import os
import subprocess
import sys
from pathlib import Path

def check_hf_token():
    """Check if HF token is configured"""
    
    # Check environment variable
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if hf_token:
        print("✅ HF_TOKEN found in environment")
        return True
    
    # Check HF CLI login status
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Logged in via huggingface-cli")
            print(f"   User: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("❌ No HuggingFace authentication found!")
    return False

def setup_hf_token():
    """Interactive HF token setup"""
    
    print("\n🔑 HuggingFace Token Required for Model Downloads")
    print("=" * 50)
    print("Models needed: Qwen2-7B, DeepSeek-Coder-7B, LLaMA-3-8B, Phi-3-14B")
    print("\n📝 Get your token: https://huggingface.co/settings/tokens")
    print("   Permissions needed: Read access to repositories")
    
    choice = input("\nChoose setup method:\n1. Environment variable (recommended)\n2. CLI login\n3. Skip (use cached models only)\n\nChoice [1-3]: ").strip()
    
    if choice == "1":
        token = input("\n🔑 Enter your HuggingFace token: ").strip()
        if token:
            # Save to shell profile for persistence
            shell_profile = Path.home() / ".bashrc"
            if not shell_profile.exists():
                shell_profile = Path.home() / ".bash_profile"
            
            with open(shell_profile, "a") as f:
                f.write(f"\n# HuggingFace token for LLM training\nexport HF_TOKEN='{token}'\n")
            
            # Set for current session
            os.environ['HF_TOKEN'] = token
            
            print(f"✅ Token saved to {shell_profile}")
            print("   Run: source ~/.bashrc (or restart terminal)")
            return True
        else:
            print("❌ No token provided")
            return False
    
    elif choice == "2":
        print("\n🔧 Running: huggingface-cli login")
        try:
            subprocess.run(['huggingface-cli', 'login'], check=True)
            print("✅ CLI login successful")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ CLI login failed: {e}")
            return False
    
    elif choice == "3":
        print("⚠️ Skipping token setup - using cached models only")
        return False
    
    else:
        print("❌ Invalid choice")
        return False

def test_model_access():
    """Test access to required models"""
    
    models = [
        "Qwen/Qwen2-7B-Instruct",
        "deepseek-ai/DeepSeek-Coder-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct", 
        "microsoft/Phi-3-medium-14b-instruct"
    ]
    
    print("\n🧪 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        
        accessible = []
        failed = []
        
        for model in models:
            try:
                print(f"   Testing {model}...", end=" ")
                tokenizer = AutoTokenizer.from_pretrained(model)
                print("✅")
                accessible.append(model)
            except Exception as e:
                print(f"❌ {str(e)[:50]}...")
                failed.append(model)
        
        print(f"\n📊 Results: {len(accessible)}/{len(models)} models accessible")
        
        if failed:
            print("\n⚠️ Failed models:")
            for model in failed:
                print(f"   - {model}")
            print("\nPossible issues:")
            print("   - Missing HF token for gated models")
            print("   - Network connectivity")
            print("   - Model repository private/removed")
        
        return len(accessible) > 0
        
    except ImportError:
        print("❌ transformers library not installed")
        print("   Run: pip install transformers")
        return False

def setup_model_cache():
    """Setup model cache directory"""
    
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set HF cache directory
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
    
    print(f"📁 HF cache: {cache_dir}")
    print(f"   Available space: {get_disk_space(cache_dir):.1f} GB")
    
    return cache_dir

def get_disk_space(path):
    """Get available disk space in GB"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return free / (1024**3)  # Convert to GB
    except:
        return 0

def main():
    """Main HF setup workflow"""
    
    print("🤗 HuggingFace Authentication Setup")
    print("For Phase 3 Model Training")
    print("=" * 35)
    
    # Check current status
    if check_hf_token():
        print("\n✅ Authentication already configured")
        
        # Test model access
        if "--test" in sys.argv:
            test_model_access()
        return True
    
    # Setup cache
    setup_model_cache()
    
    # Setup authentication
    if setup_hf_token():
        print("\n✅ Authentication setup complete")
        
        # Verify setup
        if check_hf_token():
            test_model_access()
            return True
        else:
            print("❌ Setup verification failed")
            return False
    else:
        print("\n⚠️ Continuing without authentication")
        print("   Training may fail if models are not cached")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Ready for Phase 3 training!")
    else:
        print("\n⚠️ Manual setup required")
        print("   Alternative: Pre-download models on machine with internet")
        sys.exit(1)