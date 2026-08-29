import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not found in .env file. Please add it.")
    exit(1)

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "NousResearch/Meta-Llama-3-8B-Instruct"
]

os.makedirs("models", exist_ok=True)

for repo_id in MODELS:
    local_dir = f"models/{repo_id.split('/')[-1]}"
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token)
        print(f"Finished downloading {repo_id}.")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

print("All downloads completed.")
