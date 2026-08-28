#!/bin/bash
set -e

# This script runs the C1 Phase for all 3 models sequentially on a T4 GPU.
# Requires vllm==0.27.0 and bitsandbytes.

DATASET="data/processed/cases.json"
MODELS=(
    "models/Qwen2.5-7B-Instruct"
    "models/Mistral-7B-Instruct-v0.3"
    "models/Meta-Llama-3-8B-Instruct"
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "========================================"
    echo "Running full C1 for $MODEL_NAME..."
    echo "========================================"
    
    # Ensure no vLLM instance is hanging
    pkill -f vllm || true
    while nc -z localhost 8000; do sleep 1; done

    # Start vLLM with strict T4 memory and 8-bit quantization limits
    vllm serve "$MODEL_NAME" \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.99 \
        --enforce-eager &
    
    VLLM_PID=$!
    
    echo "Waiting for $MODEL_NAME to start on port 8000..."
    while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
    echo "$MODEL_NAME is up! Running C1 evaluations..."
    
    python -m src.cli.main run-c1 "$DATASET" "$MODEL_NAME" "$MODEL_NAME" \
        --provider "vllm" \
        --endpoint "http://localhost:8000/v1" \
        --n 10 \
        --temperature 0.0
    
    echo "Killing vLLM for $MODEL_NAME..."
    kill $VLLM_PID
    pkill -f vllm || true
    while nc -z localhost 8000; do sleep 1; done
    echo "Finished $MODEL_NAME!"
done

echo "ALL C1 EVALUATIONS COMPLETE!"
