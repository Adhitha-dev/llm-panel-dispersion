#!/bin/bash
set -e

echo "Downloading Qwen3.6-27B..."
hf download Qwen/Qwen3.6-27B

echo "Downloading gemma-4-31B-it..."
hf download google/gemma-4-31B-it

echo "Downloading Mistral-Small-3.2-24B-Instruct-2506..."
hf download mistralai/Mistral-Small-3.2-24B-Instruct-2506

echo "Downloading gpt-oss-20b..."
hf download openai/gpt-oss-20b

echo "All models downloaded successfully!"
