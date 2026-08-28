#!/bin/bash
set -e

MODEL_NAME="models/Mistral-7B-Instruct-v0.3"
pkill -f vllm || true
while nc -z localhost 8000; do sleep 1; done

# Enforce eager mode as requested by Test 4, and start vLLM (added T4 memory limits and correct quant flags)
vllm serve "$MODEL_NAME" --quantization bitsandbytes --load-format bitsandbytes --max-model-len 1024 --gpu-memory-utilization 0.99 --enforce-eager &
VLLM_PID=$!

while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
echo "vLLM is up! Running diagnostics..."

python testing/diagnostics.py

kill $VLLM_PID
pkill -f vllm || true
