#!/bin/bash
set -e

echo "Downloading Meta-Llama-3.1-8B-Instruct..."
hf download meta-llama/Meta-Llama-3.1-8B-Instruct

echo "Downloading gemma-2-9b-it..."
hf download google/gemma-2-9b-it

echo "Downloading Mistral-7B-Instruct-v0.3..."
hf download mistralai/Mistral-7B-Instruct-v0.3

echo "Downloading Qwen2.5-7B-Instruct..."
hf download Qwen/Qwen2.5-7B-Instruct

echo "Downloading phi-4..."
hf download microsoft/phi-4

echo "All models downloaded successfully!"
