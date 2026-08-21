from pathlib import Path
import subprocess

dataset = "domingosun/kickstarter-dataset-629147-projects-may-2025"
output_dir = Path("data/raw/kickstarter")
output_dir.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(output_dir),
        "--unzip",
    ],
    check=True,
)

print(f"Downloaded Kickstarter data to {output_dir}")
