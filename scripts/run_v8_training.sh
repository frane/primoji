#!/bin/bash
# V8.7 Cloud Training Commands
# Run on GPU machine with the training data from HuggingFace
#
# Prerequisites:
#   pip install torch numpy huggingface_hub
#   huggingface-cli login --token $HF_TOKEN
#   huggingface-cli download fbandov/primoji-v8-training-data --local-dir ./data
#
# The training script reads pre-tokenized binary data from ./data/

set -e

DATA_DIR="./data"

# ── V8.7a: 3x 125M Primoji seeds ────────────────────────────────────────────
# Same hyperparameters as paper Table 1. 500K docs, ~565M tokens, 1 epoch.
# Different random seeds via CUDA determinism differences.

for SEED in 0 1 2; do
    echo "=== V8 125M seed $SEED ==="
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
        --tokenizer primoji \
        --data-dir "$DATA_DIR" \
        --model-size 125m \
        --batch-size 32 \
        --seq-len 1024 \
        --v2 \
        --byte-weight 0.7 \
        --device cuda

    # Rename outputs for this seed
    mv "$DATA_DIR/primoji_training_log.json" "$DATA_DIR/primoji_v8_125m_seed${SEED}_log.json"
    mv "$DATA_DIR/primoji_checkpoints/model_final.pt" "$DATA_DIR/primoji_v8_125m_seed${SEED}.pt"

    echo "Seed $SEED complete"
done

# ── V8.7b: 1x 1B Primoji ────────────────────────────────────────────────────
# NOTE: Needs at least 40GB VRAM (A100 or 2x A6000)
# Uses gradient accumulation to simulate larger batch size.

echo "=== V8 1B ==="
python -m scripts.train \
    --tokenizer primoji \
    --data-dir "$DATA_DIR" \
    --model-size 1b \
    --batch-size 4 \
    --grad-accum 8 \
    --seq-len 1024 \
    --v2 \
    --byte-weight 0.7 \
    --device cuda

mv "$DATA_DIR/primoji_training_log.json" "$DATA_DIR/primoji_v8_1b_log.json"
mv "$DATA_DIR/primoji_checkpoints/model_final.pt" "$DATA_DIR/primoji_v8_1b.pt"

# ── V8.7c: 1x 125M BPE 3-epoch ─────────────────────────────────────────────
# Tests whether alias gain is alias-specific or data-exposure.

echo "=== BPE 125M 3-epoch ==="
python -m scripts.train \
    --tokenizer mistral \
    --data-dir "$DATA_DIR" \
    --model-size 125m \
    --batch-size 32 \
    --seq-len 1024 \
    --device cuda

mv "$DATA_DIR/mistral_training_log.json" "$DATA_DIR/bpe_125m_3epoch_log.json"
mv "$DATA_DIR/mistral_checkpoints/model_final.pt" "$DATA_DIR/bpe_125m_3epoch.pt"

echo "All V8.7 training complete!"
