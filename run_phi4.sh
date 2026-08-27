#!/bin/bash
set -e

# Kill any existing vLLM
pkill -f vllm || true
while nc -z localhost 8000; do sleep 1; done
echo "Port 8000 is free."

# Remove the broken phi-4 experiment dir
rm -rf experiments/EXP_*_C2_phi-4 || true

# Start vLLM with enforce eager and smaller max model len
vllm serve models/phi-4 --quantization bitsandbytes --load-format bitsandbytes --max-model-len 1024 --enforce-eager &
VLLM_PID=$!

echo "Waiting for Phi-4 to start on port 8000..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
echo "Phi-4 is up! Running evaluations..."

python run_c2_single.py data/processed/cases.json "models/phi-4" "models/phi-4" "vllm" "http://localhost:8000/v1"

# Kill vLLM
kill $VLLM_PID
pkill -f vllm || true
while nc -z localhost 8000; do sleep 1; done
echo "Phi-4 C2 run complete!"
