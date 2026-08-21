import typer
import asyncio
from pathlib import Path
from datetime import datetime
import yaml
from typing import List, Optional

from src.dataset.loader import load_cases
from src.inference.client import InferenceClient
from src.storage.logger import RunLogger
from src.runner.executor import run_evaluation_task
from src.config.prompt import EVALUATION_PROMPT_TEMPLATE
from src.config.settings import get_prompt_hash, settings

app = typer.Typer()

def setup_experiment(exp_name: str) -> tuple[Path, RunLogger, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(settings.experiment_root) / f"EXP_{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / "raw").mkdir(exist_ok=True)
    (exp_dir / "parsed").mkdir(exist_ok=True)
    (exp_dir / "tables").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    logger = RunLogger(
        raw_log_path=exp_dir / "raw" / "inference_attempts.jsonl",
        parsed_valid_path=exp_dir / "parsed" / "valid_responses.jsonl",
        parsed_invalid_path=exp_dir / "parsed" / "invalid_responses.jsonl"
    )
    
    return exp_dir, logger, get_prompt_hash(EVALUATION_PROMPT_TEMPLATE)

def create_manifest(exp_dir: Path, config: dict):
    with open(exp_dir / "manifest.yaml", "w") as f:
        yaml.dump(config, f)

async def _run_condition(cases, client, logger, condition_params, prompt_hash):
    tasks = []
    # condition_params format: [{"model": "...", "snapshot": "...", "provider": "...", "endpoint": "...", "condition": "C1", "replicate_index": i, "temperature": 0.0}]
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
    
    # We could batch these properly, but for ~60 cases * 10 replicates = 600 tasks, we can use asyncio.gather with a semaphore in a real scenario
    # Given the L40s context, we'll bound concurrency to 10
    semaphore = asyncio.Semaphore(10)
    async def sem_task(t):
        async with semaphore:
            await t
            
    await asyncio.gather(*(sem_task(t) for t in tasks))

@app.command()
def run_c1(
    dataset_path: str,
    model: str,
    snapshot: str,
    provider: str = "openai",
    endpoint: str = "http://localhost:8000/v1",
    n: int = 10,
    temperature: float = 0.0
):
    cases = load_cases(dataset_path)
    client = InferenceClient(provider=provider, base_url=endpoint)
    exp_dir, logger, prompt_hash = setup_experiment("C1")
    
    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "condition": "C1",
        "dataset_path": dataset_path,
        "models": [{"name": model, "snapshot": snapshot, "provider": provider, "endpoint": endpoint}],
        "replicates": n,
        "temperature": temperature,
        "prompt_hash": prompt_hash
    }
    create_manifest(exp_dir, manifest)
    
    condition_params = []
    for i in range(1, n + 1):
        condition_params.append({
            "model": model, "snapshot": snapshot, "provider": provider, 
            "endpoint": endpoint, "condition": "C1", "replicate_index": i, "temperature": temperature
        })
        
    asyncio.run(_run_condition(cases, client, logger, condition_params, prompt_hash))
    
    # generate CSV
    RunLogger.jsonl_to_csv(exp_dir / "parsed" / "valid_responses.jsonl", exp_dir / "tables" / "results.csv")
    print(f"C1 run completed. Results saved to {exp_dir}")

@app.command()
def run_c2(
    dataset_path: str,
    models: List[str],
    snapshots: List[str],
    providers: List[str],
    endpoints: List[str],
    temperature: float = 0.0
):
    cases = load_cases(dataset_path)
    exp_dir, logger, prompt_hash = setup_experiment("C2")
    
    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "condition": "C2",
        "dataset_path": dataset_path,
        "models": [{"name": m, "snapshot": s, "provider": p, "endpoint": e} for m, s, p, e in zip(models, snapshots, providers, endpoints)],
        "replicates": 1,
        "temperature": temperature,
        "prompt_hash": prompt_hash
    }
    create_manifest(exp_dir, manifest)
    
    condition_params = []
    for m, s, p, e in zip(models, snapshots, providers, endpoints):
        condition_params.append({
            "model": m, "snapshot": s, "provider": p, 
            "endpoint": e, "condition": "C2", "replicate_index": 1, "temperature": temperature
        })
        
    # Assume single provider logic for this example or need a client per provider
    # For now, initialize per model
    clients = {p: InferenceClient(provider=p, base_url=e) for p, e in zip(providers, endpoints)}
    
    async def run_c2_tasks():
        tasks = []
        for case in cases:
            for params in condition_params:
                tasks.append(run_evaluation_task(
                    client=clients[params["provider"]],
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

if __name__ == "__main__":
    app()

from src.metrics.calculator import compute_c1_metrics, compute_c2_metrics, generate_four_pattern_grid
from src.reporting.plots import plot_c1_score_distribution, plot_correlation_heatmap, plot_verdict_entropy, plot_c1_vs_c2_variability, plot_pairwise_agreement_heatmap
import pandas as pd

@app.command()
def compute_metrics(
    c1_results_csv: str,
    c2_results_csv: str,
    output_dir: str
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df_c1 = pd.read_csv(c1_results_csv)
    df_c2 = pd.read_csv(c2_results_csv)
    
    # C1 metrics
    c1_metrics = compute_c1_metrics(df_c1)
    c1_metrics.to_csv(out_path / "c1_metrics.csv", index=False)
    
    # C2 metrics
    c2_res = compute_c2_metrics(df_c2)
    c2_metrics_df = c2_res["per_case"]
    c2_metrics_df.to_csv(out_path / "c2_metrics.csv", index=False)
    c2_res["correlation_matrix"].to_csv(out_path / "c2_correlation_matrix.csv")
    c2_res["agreement_matrix"].to_csv(out_path / "c2_agreement_matrix.csv")
    
    print(f"Global C2 Effective Viewpoints: {c2_res['effective_viewpoints']:.2f}")
    
    # 4-pattern grid
    grid_df = generate_four_pattern_grid(c1_metrics, c2_metrics_df)
    grid_df.to_csv(out_path / "four_pattern_grid.csv", index=False)
    
    print(f"Metrics computed and saved to {output_dir}")
    
@app.command()
def make_figures(
    c1_results_csv: str,
    c2_correlation_csv: str,
    c2_agreement_csv: str,
    c2_metrics_csv: str,
    grid_csv: str,
    output_dir: str
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df_c1 = pd.read_csv(c1_results_csv)
    plot_c1_score_distribution(df_c1, out_path)
    
    corr_matrix = pd.read_csv(c2_correlation_csv, index_col=0)
    plot_correlation_heatmap(corr_matrix, out_path)
    
    agreement_matrix = pd.read_csv(c2_agreement_csv, index_col=0)
    plot_pairwise_agreement_heatmap(agreement_matrix, out_path)
    
    df_c2_metrics = pd.read_csv(c2_metrics_csv)
    plot_verdict_entropy(df_c2_metrics, out_path)
    
    df_grid = pd.read_csv(grid_csv)
    plot_c1_vs_c2_variability(df_grid, out_path)
    
    print(f"Figures generated in {output_dir}")
