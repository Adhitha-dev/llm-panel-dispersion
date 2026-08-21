import os
import subprocess
import sys
from pathlib import Path

def print_step(msg):
    print(f"\n{'-'*60}\n> {msg}\n{'-'*60}")

def run_cmd(cmd, error_msg="Command failed"):
    try:
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {error_msg}")
        print(f"Details: {e}")
        sys.exit(1)

def setup_project():
    print_step("1. Creating Project Directory Structure")
    dirs = [
        "data/raw/kickstarter",
        "data/processed",
        "data/validation",
        "experiments",
        "models",
        "src/config",
        "src/dataset",
        "src/inference",
        "src/schemas",
        "src/runner",
        "src/storage",
        "src/metrics",
        "src/reporting",
        "src/cli",
        "scripts",
        "tests"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created.")

    print_step("2. Installing Dependencies")
    run_cmd("pip install -r requirements.txt pydantic-settings", "Failed to install dependencies.")
    print("✅ Dependencies installed.")

    print_step("3. Generating Synthetic Case Baseline")
    if Path("generate_synthetic.py").exists():
        run_cmd("python generate_synthetic.py", "Failed to generate synthetic cases.")
    else:
        print("⚠️ generate_synthetic.py not found. Skipping.")

    print_step("4. Downloading Raw Kickstarter Dataset (Kaggle)")
    print("Requires Kaggle API token (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars).")
    try:
        subprocess.run("python scripts/download_kickstarter.py", shell=True, check=True)
        print("✅ Raw dataset downloaded.")
    except subprocess.CalledProcessError:
        print("⚠️ Kaggle download failed or token not found. Proceeding with existing data if present.")

    print_step("5. Processing and Cleaning Dataset")
    raw_file = Path("data/raw/kickstarter/Kickstarter Full Campaigns_694062_25_April_2026.csv")
    if raw_file.exists():
        run_cmd("python scripts/process_kickstarter.py", "Failed to clean and process dataset.")
        print("✅ Dataset processed into data/processed/cases.json.")
    else:
        print("⚠️ Raw dataset not found. Skipping processing step.")

    print_step("6. Model Weights Download Reminder")
    print("To download the ~200GB of LLM weights (Qwen, Gemma, Mistral, gpt-oss) locally to the models/ dir:")
    print("Ensure you have set 'export HF_TOKEN=<your_token>' and run:")
    print("    python scripts/download_models.py")

    print_step("🎉 Setup Complete!")
    print("""
You are fully ready to run the evaluation pipeline once your models are served.

To run the noise floor (C1) condition:
    python -m src.cli.main run-c1 data/processed/cases.json "Qwen/Qwen3.6-27B" "qwen-snapshot" --n 10 --endpoint "http://localhost:8000/v1"

To run the panel (C2) condition:
    python -m src.cli.main run-c2 data/processed/cases.json "Qwen/Qwen3.6-27B" "google/gemma-4-31B-it" "snapshot-1" "snapshot-2" "openai" "openai" "http://localhost:8000/v1" "http://localhost:8001/v1"

To calculate all metrics and generate paper figures:
    python -m src.cli.main compute-metrics experiments/EXP_C1_XYZ/tables/results.csv experiments/EXP_C2_ABC/tables/results.csv experiments/final_metrics
    python -m src.cli.main make-figures ...
""")

if __name__ == "__main__":
    setup_project()
