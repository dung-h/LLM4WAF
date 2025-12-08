#!/bin/bash
# Cleanup workspace for remote training upload

set -e

echo "🧹 Cleaning LLM4WAF workspace for remote training..."
echo ""

# Safety check
read -p "⚠️  This will delete eval results, logs, and caches. Continue? (y/N): " confirm
if [ "$confirm" != "y" ]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

# Remove evaluation results
echo "📊 Removing evaluation results..."
rm -rf eval/
rm -rf reports/

# Remove logs
echo "📝 Removing log files..."
rm -f *.log
rm -f SFT_*.log
rm -f rl_training_*.log
rm -f training_*.log
rm -f eval_*.log

# Remove Python cache
echo "🐍 Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove temp files
echo "🗑️  Removing temp files..."
rm -rf .git_temp
rm -rf .vscode/
rm -rf .idea/
rm -f .DS_Store

# Remove archives (keep only processed data)
echo "📦 Removing data archives..."
rm -rf data/archive_*
rm -rf archive_legacy/

# Ask about experiments
echo ""
read -p "⚠️  Delete ALL experiment checkpoints? This will save ~10-50GB (y/N): " del_exp
if [ "$del_exp" = "y" ]; then
    echo "🗂️  Removing experiments..."
    rm -rf experiments/*
    echo "✅ Experiments deleted"
else
    echo "⏭️  Keeping experiments"
fi

# Show final size
echo ""
echo "📊 Final directory size:"
du -sh .
echo ""
echo "📂 Largest directories:"
du -sh */ | sort -rh | head -10

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "1. Run: head -n 10000 data/processed/red_v40_balanced_final_v13.jsonl > data/processed/red_v40_phase1_10k.jsonl"
echo "2. Create configs in configs/remote_*.yaml"
echo "3. Compress: tar -czf llm4waf.tar.gz --exclude='.git' ."
echo "4. Upload to remote server"
