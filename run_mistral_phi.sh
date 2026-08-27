#!/bin/bash
set -e

run_model() {
    MODEL_NAME=$1
    echo "Running C2 for $MODEL_NAME..."
    
    # Kill any existing vLLM
    pkill -f vllm || true
    while nc -z localhost 8000; do sleep 1; done
    echo "Port 8000 is free."

    # Start vLLM
    vllm serve $MODEL_NAME --quantization bitsandbytes --load-format bitsandbytes --max-model-len 2048 &
    VLLM_PID=$!
    
    echo "Waiting for $MODEL_NAME to start on port 8000..."
    while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
    echo "$MODEL_NAME is up! Running evaluations..."
    
    python run_c2_single.py data/processed/cases.json "$MODEL_NAME" "$MODEL_NAME" "vllm" "http://localhost:8000/v1"
    
    # Kill vLLM
    kill $VLLM_PID
    pkill -f vllm || true
    while nc -z localhost 8000; do sleep 1; done
}

run_model "models/Mistral-7B-Instruct-v0.3"
run_model "models/phi-4"

echo "All C2 runs complete!"
