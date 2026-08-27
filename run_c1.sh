#!/bin/bash
echo "Waiting for vLLM to start on port 8000..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
echo "Server is up! Running C1 experiment..."
python -m src.cli.main run-c1 data/processed/cases.json "models/Qwen2.5-7B-Instruct" "models/Qwen2.5-7B-Instruct" --n 10 --endpoint "http://localhost:8000/v1"
echo "Done!"
