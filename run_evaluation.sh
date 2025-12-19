#!/bin/bash

source .venv/bin/activate

echo "Starting evaluation of all models in models/ directory..."

# Loop through all .pth files in the models directory
for model_file in models/*.pth; do
    if [ -f "$model_file" ]; then
        echo "Evaluating model: $model_file"
        python evaluate.py --model "$model_file"
        echo "-------------------------------------------------------"
    fi
done

echo "All evaluations completed."
