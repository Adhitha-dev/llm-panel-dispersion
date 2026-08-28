import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

repo_id = "microsoft/Phi-3-small-128k-instruct"
local_dir = "models/Phi-3-small-128k-instruct"

print(f"Downloading {repo_id} to {local_dir}...")
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token)
print(f"Finished {repo_id}.")
