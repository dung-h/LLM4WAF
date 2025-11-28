#!/usr/bin/env python3
"""
Pre-download all Phase 3 models in parallel to avoid delays during training
Saves models to HuggingFace cache for offline use
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

def setup_hf_token():
    """Ensure HF token is set"""
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not hf_token:
        # Try to load from cache
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            with open(token_path, 'r') as f:
                hf_token = f.read().strip()
                os.environ['HF_TOKEN'] = hf_token
                print(f"✅ Loaded HF_TOKEN from {token_path}")
                return True
        
        print("❌ HF_TOKEN not found!")
        print("   Set with: export HF_TOKEN='your_token_here'")
        print("   Or run: huggingface-cli login")
        return False
    
    print("✅ HF_TOKEN found")
    return True

def download_model(model_info):
    """Download a single model"""
    model_name, model_type = model_info
    start_time = time.time()
    
    try:
        print(f"\n📥 Downloading {model_type}: {model_name}")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Download tokenizer
        print(f"   ⏳ Tokenizer for {model_type}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=os.getenv('HF_TOKEN')
        )
        
        # Download model (4-bit to save bandwidth/disk)
        print(f"   ⏳ Model weights for {model_type}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # Don't load to GPU yet
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=os.getenv('HF_TOKEN'),
            low_cpu_mem_usage=True
        )
        
        # Free memory
        del model
        del tokenizer
        
        duration = time.time() - start_time
        print(f"   ✅ {model_type} downloaded in {duration:.1f}s")
        
        return True, model_type, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ {model_type} failed: {str(e)[:100]}")
        return False, model_type, duration

def download_all_models_parallel():
    """Download all Phase 3 models in parallel"""
    
    # All Phase 3 models
    models = [
        ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B (Local)"),
        ("Qwen/Qwen2-7B-Instruct", "Qwen2-7B"),
        ("deepseek-ai/DeepSeek-Coder-7B-Instruct", "DeepSeek-7B"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "LLaMA-3-8B"),
        ("microsoft/Phi-3-medium-14b-instruct", "Phi-3-14B"),
    ]
    
    print("🚀 Starting parallel model downloads...")
    print(f"📊 Total models: {len(models)}")
    print("=" * 60)
    
    # Download in parallel (max 3 at once to avoid overwhelming network)
    max_workers = 3
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_model = {
            executor.submit(download_model, model): model 
            for model in models
        }
        
        # Process completed downloads
        for future in as_completed(future_to_model):
            model_info = future_to_model[future]
            try:
                success, model_type, duration = future.result()
                if success:
                    successful.append((model_type, duration))
                else:
                    failed.append(model_type)
            except Exception as e:
                print(f"❌ Unexpected error for {model_info[1]}: {e}")
                failed.append(model_info[1])
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    
    if successful:
        print(f"\n✅ Successfully downloaded ({len(successful)}/{len(models)}):")
        total_time = sum(d for _, d in successful)
        for model_type, duration in successful:
            print(f"   • {model_type}: {duration:.1f}s")
        print(f"\n⏱️  Total download time: {total_time:.1f}s")
    
    if failed:
        print(f"\n❌ Failed downloads ({len(failed)}/{len(models)}):")
        for model_type in failed:
            print(f"   • {model_type}")
        print("\n⚠️  Note: LLaMA models require acceptance of license agreement")
        print("   Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Cache location
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        try:
            import shutil
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            print(f"\n💾 Cache size: {size_gb:.1f} GB")
            print(f"📁 Cache location: {cache_dir}")
        except:
            pass
    
    return len(failed) == 0

def main():
    """Main download workflow"""
    
    print("🤗 HuggingFace Model Pre-download")
    print("Phase 3 Training Preparation")
    print("=" * 35)
    
    # Check HF token
    if not setup_hf_token():
        print("\n⚠️  Continuing without token (may fail for gated models)")
    
    # Check available disk space
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / (1024**3)
        print(f"\n💾 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 100:
            print("⚠️  Warning: Less than 100GB free")
            print("   Phase 3 models need ~80-100GB total")
            response = input("\n   Continue anyway? [y/N]: ").lower()
            if response != 'y':
                print("❌ Download cancelled")
                return False
    except:
        pass
    
    print("\n🚀 Starting downloads...")
    print("   (This may take 30-60 minutes depending on connection)")
    
    start_time = time.time()
    success = download_all_models_parallel()
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total elapsed time: {total_time/60:.1f} minutes")
    
    if success:
        print("\n🎯 All models downloaded successfully!")
        print("✅ Ready for Phase 3 training")
        print("\n🚀 Next steps:")
        print("   python scripts/train_red.py --config configs/phase3_qwen_7b_fixed.yaml")
        return True
    else:
        print("\n⚠️  Some models failed to download")
        print("   Training may fail if models are not cached")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)