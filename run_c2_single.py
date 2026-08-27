import asyncio
from pathlib import Path
from src.cli.main import setup_experiment, create_manifest
from src.dataset.loader import load_cases
from src.inference.client import InferenceClient
from src.runner.executor import run_evaluation_task
from src.storage.logger import RunLogger
from datetime import datetime
import sys

dataset_path = sys.argv[1]
model = sys.argv[2]
snapshot = sys.argv[3]
provider = sys.argv[4]
endpoint = sys.argv[5]

cases = load_cases(dataset_path)
exp_dir, logger, prompt_hash = setup_experiment(f"C2_{model.split('/')[-1]}")

manifest = {
    "timestamp": datetime.utcnow().isoformat(),
    "condition": "C2",
    "dataset_path": dataset_path,
    "models": [{"name": model, "snapshot": snapshot, "provider": provider, "endpoint": endpoint}],
    "replicates": 1,
    "temperature": 0.0,
    "prompt_hash": prompt_hash
}
create_manifest(exp_dir, manifest)

condition_params = [{
    "model": model, "snapshot": snapshot, "provider": provider, 
    "endpoint": endpoint, "condition": "C2", "replicate_index": 1, "temperature": 0.0
}]

client = InferenceClient(provider=provider, base_url=endpoint)

async def run_c2_tasks():
    tasks = []
    for case in cases:
        for params in condition_params:
            tasks.append(run_evaluation_task(
                client=client,
                logger=logger,
                case=case,
                model=params["model"],
                model_snapshot=params["snapshot"],
                provider=params["provider"],
                endpoint=params["endpoint"],
                condition=params["condition"],
                replicate_index=params["replicate_index"],
                temperature=params["temperature"],
                prompt_hash=prompt_hash
            ))
    semaphore = asyncio.Semaphore(10)
    async def sem_task(t):
        async with semaphore:
            await t
    await asyncio.gather(*(sem_task(t) for t in tasks))

asyncio.run(run_c2_tasks())
RunLogger.jsonl_to_csv(exp_dir / "parsed" / "valid_responses.jsonl", exp_dir / "tables" / "results.csv")
print(f"C2 run completed. Results saved to {exp_dir}")
