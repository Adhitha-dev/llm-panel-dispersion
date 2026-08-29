#!/bin/bash
set -e

DATASET="data/processed/cases.json"
MODEL_NAME="models/Meta-Llama-3-8B-Instruct"
MASTER_C2_DIR="experiments/EXP_20260829_004952_C2_MASTER"

echo "Killing any lingering vLLM processes..."
pkill -f vllm || true
sleep 10

echo "Running C2 for $MODEL_NAME..."
vllm serve "$MODEL_NAME" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.99 \
    --enforce-eager &
    
VLLM_PID=$!

echo "Waiting for $MODEL_NAME to start on port 8000..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
# Add a small safety sleep just in case
sleep 5

echo "$MODEL_NAME is up! Running C2 evaluations..."
python -m src.cli.main run-c2 "$DATASET" \
    --model "$MODEL_NAME" \
    --snapshot "$MODEL_NAME" \
    --provider "vllm" \
    --endpoint "http://localhost:8000/v1" \
    --temperature 0.0

LATEST_C2=$(ls -td experiments/EXP_*_C2 | head -1)

echo "Appending valid responses to master JSONL..."
cat "$LATEST_C2/parsed/valid_responses.jsonl" >> "$MASTER_C2_DIR/parsed/valid_responses.jsonl" || true

echo "Killing vLLM for $MODEL_NAME..."
kill $VLLM_PID
pkill -f vllm || true

echo "Rebuilding master CSV..."
python -c "
import json
import pandas as pd
records = []
with open('$MASTER_C2_DIR/parsed/valid_responses.jsonl', 'r') as f:
    for line in f:
        if line.strip(): records.append(json.loads(line))
pd.DataFrame(records).to_csv('$MASTER_C2_DIR/tables/results.csv', index=False)
print(f'Saved {len(records)} records to CSV')
"
