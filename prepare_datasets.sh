#!/bin/bash
# Prepare datasets for remote training

set -e

echo "📦 Preparing datasets for remote training..."
echo ""

# Phase 1 dataset (10k samples with balanced technique distribution)
if [ ! -f "data/processed/phase1_balanced_10k.jsonl" ]; then
    echo "Creating Phase 1 balanced dataset (10k samples, 509 techniques)..."
    python scripts/create_balanced_phase1_10k.py
    echo "✅ Created: data/processed/phase1_balanced_10k.jsonl"
else
    echo "⏭️  Phase 1 balanced dataset already exists"
fi

# Phase 2 dataset (20k observations + 20% Phase 1 replay = 24k total)
if [ ! -f "data/processed/phase2_with_replay_22k.jsonl" ]; then
    echo ""
    echo "Creating Phase 2 with replay buffer (24k samples)..."
    python scripts/create_phase2_with_replay.py
    echo "✅ Created: data/processed/phase2_with_replay_22k.jsonl"
else
    echo "⏭️  Phase 2 with replay dataset already exists"
fi

echo ""
echo "📊 Dataset Summary:"
echo "  Phase 1 (balanced): $(wc -l < data/processed/phase1_balanced_10k.jsonl 2>/dev/null || echo '0') samples (509 techniques)"
echo "  Phase 2 (with replay): $(wc -l < data/processed/phase2_with_replay_22k.jsonl 2>/dev/null || echo '0') samples (20k obs + 4k replay)"
echo "  Phase 3: Live RL testing with WAF (no static dataset)"
echo ""
echo "✅ All datasets ready for remote training!"
echo ""

# Show sample from Phase 1
echo "📝 Phase 1 Sample (first line):"
head -n 1 data/processed/phase1_balanced_10k.jsonl | python3 -m json.tool 2>/dev/null || head -n 1 data/processed/phase1_balanced_10k.jsonl

echo ""
echo "✅ Dataset preparation complete!"
