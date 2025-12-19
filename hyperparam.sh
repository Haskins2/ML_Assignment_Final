#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Common settings
MAX_ITERS=2000

echo "Starting hyperparameter ablation study..."

# 1. Original Hyperparameters (with block_size=24 fix)
# Note: max_iters is set to 2000 for all runs as requested
# Note: block_size set to 24 for all runs to avoid dataset size errors
echo "Running 1: Original Hyperparameters (block_size=24)"
python gpt.py \
    --model_name "model_01_original" \
    --max_iters $MAX_ITERS \
    --batch_size 64 \
    --block_size 24 \
    --eval_interval 500 \
    --learning_rate 3e-4 \
    --n_embd 384 \
    --n_head 6 \
    --n_layer 6 \
    --dropout 0.2

# 2. Change batch_size: 64 -> 32
echo "Running 2: batch_size -> 32"
python gpt.py \
    --model_name "model_02_batch_size" \
    --max_iters $MAX_ITERS \
    --batch_size 32 \
    --block_size 24 \
    --eval_interval 500 \
    --learning_rate 3e-4 \
    --n_embd 384 \
    --n_head 6 \
    --n_layer 6 \
    --dropout 0.2


# 4. Change eval_interval: 500 -> 5
echo "Running 4: eval_interval -> 5"
python gpt.py \
    --model_name "model_04_eval_interval" \
    --max_iters $MAX_ITERS \
    --batch_size 32 \
    --block_size 24 \
    --eval_interval 5 \
    --learning_rate 3e-4 \
    --n_embd 384 \
    --n_head 6 \
    --n_layer 6 \
    --dropout 0.2

# 5. Change learning_rate: 3e-4 -> 1e-3
echo "Running 5: learning_rate -> 1e-3"
python gpt.py \
    --model_name "model_05_learning_rate" \
    --max_iters $MAX_ITERS \
    --batch_size 32 \
    --block_size 24 \
    --eval_interval 5 \
    --learning_rate 1e-3 \
    --n_embd 384 \
    --n_head 6 \
    --n_layer 6 \
    --dropout 0.2

# 6. Change n_embd: 384 -> 64
echo "Running 6: n_embd -> 64"
python gpt.py \
    --model_name "model_06_n_embd" \
    --max_iters $MAX_ITERS \
    --batch_size 32 \
    --block_size 24 \
    --eval_interval 5 \
    --learning_rate 1e-3 \
    --n_embd 64 \
    --n_head 6 \
    --n_layer 6 \
    --dropout 0.2

# 7. Change n_head: 6 -> 4
echo "Running 7: n_head -> 4"
python gpt.py \
    --model_name "model_07_n_head" \
    --max_iters $MAX_ITERS \
    --batch_size 32 \
    --block_size 24 \
    --eval_interval 5 \
    --learning_rate 1e-3 \
    --n_embd 64 \
    --n_head 4 \
    --n_layer 6 \
    --dropout 0.2

# 8. Change n_layer: 6 -> 4
echo "Running 8: n_layer -> 4"
python gpt.py \
    --model_name "model_08_n_layer" \
    --max_iters $MAX_ITERS \
    --batch_size 32 \
    --block_size 24 \
    --eval_interval 5 \
    --learning_rate 1e-3 \
    --n_embd 64 \
    --n_head 4 \
    --n_layer 4 \
    --dropout 0.2

# 9. Change dropout: 0.2 -> 0.0 (Final Target)
echo "Running 9: dropout -> 0.0 (All Target)"
python gpt.py \
    --model_name "model_09_dropout_target" \
    --max_iters $MAX_ITERS \
    --batch_size 32 \
    --block_size 24 \
    --eval_interval 5 \
    --learning_rate 1e-3 \
    --n_embd 64 \
    --n_head 4 \
    --n_layer 4 \
    --dropout 0.0

echo "All runs completed."
