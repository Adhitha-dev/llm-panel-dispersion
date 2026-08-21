import os
from pathlib import Path
from huggingface_hub import snapshot_download

# List of models specified in models.md
MODELS = [
    "Qwen/Qwen3.6-27B",
    "google/gemma-4-31B-it",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "openai/gpt-oss-20b"
]

def download_models():
    # By default, store models locally in the project repository under 'models/'
    # This keeps them contained so the system can mount them directly for inference
    download_dir = Path("models")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting download of {len(MODELS)} models to {download_dir.absolute()} ...")
    print("This will take significant time and storage (~200GB+).")
    
    # Retrieve HF_TOKEN if available (needed for gated models like Gemma)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nWARNING: HF_TOKEN environment variable is not set.")
        print("Models like 'gemma-4-31B-it' are typically gated and require an authenticated token.")
        print("If a download fails with a 401/403, please run: export HF_TOKEN='your_token_here'\n")

    for repo_id in MODELS:
        print(f"\n--- Downloading: {repo_id} ---")
        model_name = repo_id.split("/")[-1]
        local_model_path = download_dir / model_name
        
        try:
            # We use snapshot_download to reliably pull the whole repo, skipping symlinks if needed.
            # You can add `ignore_patterns=["*.msgpack", "*.h5"]` to exclude specific file types if needed.
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_model_path),
                local_dir_use_symlinks=False, # Set to False to ensure actual files are downloaded to the folder
                token=token
            )
            print(f"✅ Successfully downloaded {repo_id} to {local_model_path}")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    download_models()
