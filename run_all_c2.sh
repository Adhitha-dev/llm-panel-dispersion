#!/bin/bash
set -e

DATASET="data/processed/cases.json"
MODELS=(
    "models/Qwen2.5-7B-Instruct"
    "models/Mistral-7B-Instruct-v0.3"
    "models/Meta-Llama-3-8B-Instruct"
)

# Create a master folder for C2
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_C2_DIR="experiments/EXP_${TIMESTAMP}_C2_MASTER"
mkdir -p "$MASTER_C2_DIR/tables"
mkdir -p "$MASTER_C2_DIR/parsed"

echo "timestamp,condition,dataset_path,model,snapshot,provider,endpoint,temperature,prompt_hash" > "$MASTER_C2_DIR/manifest_summary.csv"

for MODEL_NAME in "${MODELS[@]}"; do
    echo "========================================"
    echo "Running C2 for $MODEL_NAME..."
    echo "========================================"
    
    pkill -f vllm || true
    sleep 15

    vllm serve "$MODEL_NAME" \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.99 \
        --enforce-eager &
    
    VLLM_PID=$!
    
    echo "Waiting for $MODEL_NAME to start on port 8000..."
    while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
    echo "$MODEL_NAME is up! Running C2 evaluations..."
    
    # Run C2 for this single model
    python -m src.cli.main run-c2 "$DATASET" \
        --model "$MODEL_NAME" \
        --snapshot "$MODEL_NAME" \
        --provider "vllm" \
        --endpoint "http://localhost:8000/v1" \
        --temperature 0.0
    
    # The script created a folder in experiments/EXP_*_C2. Let's find it.
    LATEST_C2=$(ls -td experiments/EXP_*_C2 | head -1)
    
    # Append its jsonl to the master jsonl
    cat "$LATEST_C2/parsed/valid_responses.jsonl" >> "$MASTER_C2_DIR/parsed/valid_responses.jsonl" || true
    
    echo "Killing vLLM for $MODEL_NAME..."
    kill $VLLM_PID
    pkill -f vllm || true
    sleep 15
    echo "Finished C2 for $MODEL_NAME!"
done

# Now convert the master jsonl to CSV
python -c "
import json
import pandas as pd
records = []
with open('$MASTER_C2_DIR/parsed/valid_responses.jsonl', 'r') as f:
    for line in f:
        if line.strip(): records.append(json.loads(line))
pd.DataFrame(records).to_csv('$MASTER_C2_DIR/tables/results.csv', index=False)
"

echo "ALL C2 EVALUATIONS COMPLETE! Results saved to $MASTER_C2_DIR"
