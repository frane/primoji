#!/bin/bash
# Full 125M training experiment: primoji vs BPE
# Run this and let it go overnight.
#
# Usage:
#   cd /Users/frane/workspace/primoji
#   source .venv/bin/activate
#   bash code/scripts/run_experiment.sh 2>&1 | tee code/data/experiment/experiment.log

set -e

echo "=== PRIMOJI vs BPE: 125M TRAINING EXPERIMENT ==="
echo "Started: $(date)"
echo ""

cd /Users/frane/workspace/primoji

# Step 1: Prepare data (50K docs)
echo ">>> Step 1: Preparing training data (50K docs)..."
python -m scripts.prepare_training_data --n-docs 50000

# Step 2: Train primoji model
echo ""
echo ">>> Step 2: Training primoji model..."
python -m scripts.train_125m --tokenizer primoji --batch-size 8 --seq-len 1024

# Step 3: Train BPE model
echo ""
echo ">>> Step 3: Training mistral BPE model..."
python -m scripts.train_125m --tokenizer mistral --batch-size 8 --seq-len 1024

# Step 4: Compare results
echo ""
echo ">>> Step 4: Generating comparison plots and report..."
python -m scripts.compare_results

echo ""
echo "=== EXPERIMENT COMPLETE ==="
echo "Finished: $(date)"
echo "Results in: code/data/experiment/"
echo "Plots in: code/data/experiment/plots/"
