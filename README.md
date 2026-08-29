# LLM Panel Disagreement Harness — Startup Idea Evaluation

This repository contains the dataset, execution harness, and experimental results for evaluating whether a panel of Large Language Models (LLMs) disagreeing on a startup idea is pure stochastic noise, or if it represents structured, substantive pluralism. 

This work specifically addresses domain-specific mechanisms in startup evaluation: the absence of eventual ground truth, persuasion-susceptibility, and memorization/contamination confounds.

## Repository Structure

To make reviewing the empirical results as easy as possible, the raw data and figures are kept at the repository root, while the harness code is isolated in a dedicated folder.

* **`experiments/`**: Contains the complete artifacts from the experimental runs.
  * `FINAL_FIGURES/`: The generated plots used in the paper (e.g., correlation heatmaps, 4-quadrant interpretation grids).
  * `FINAL_METRICS/`: The calculated CSV matrices for C1 (within-model stability) and C2 (cross-model disagreement).
  * `EXP_.../`: The raw JSONL logs, parsed outputs, and verdict tables for every single API call made during the experiment.
* **`reproduction_code/`**: Contains the full Python and Bash execution harness required to run the evaluations from scratch.
  * `src/`: The core Python package handling prompt generation, API inference, logging, and metric calculation.
  * `data/`: The startup cases dataset (`cases.json`) used for the evaluations.
  * `download_models.py`: A secure script for downloading the exact model architectures using a HuggingFace token.
  * `run_all_c1.sh` / `run_all_c2.sh`: The master bash pipelines that boot `vLLM`, enforce strict T4 GPU memory limits, execute the trials, and safely spin down the server.
* **`AGENTS.md`**: The strict architectural blueprint and build specification guiding the experimental design.
* **`paper_draft.pdf`**: The manuscript draft associated with this repository.

## Methodology & Calculations

The experiment is divided into two distinct conditions to isolate hardware/sampling noise from genuine model disagreement. The metric calculation logic is implemented in `reproduction_code/src/metrics/calculator.py`.

### Condition 1 (C1): Same-Model Replicates (The Noise Floor)
**Goal:** Measure a model's internal stochastic instability on identical prompts at Temperature 0.0.
* **Flip Rate:** The fraction of replicates disagreeing with the model's own majority verdict.
* **Score Standard Deviation:** The variance of the numerical scores (1-10) across 10 identical runs.
* **Rubric Profile Distance:** The mean absolute deviation across the 3 sub-dimensions (Market, Feasibility, Business Viability) to detect if a model reached the same final score via different reasoning paths.

### Condition 2 (C2): Cross-Model Panel Disagreement
**Goal:** Measure genuine substantive disagreement between distinct LLM families (Qwen, Mistral, Llama).
* **Verdict Entropy:** Calculates how split the panel is on a specific idea ($0 = unanimous$, max at even split).
* **Mean Pairwise Absolute Deviation (MPAD):** Direct dispersion metric on numeric scores across the panel members.
* **Effective Viewpoints:** Using Pearson score correlation as a similarity measure, we calculate Kish's effective sample size to determine how many *statistically independent* viewpoints a panel of $N$ models actually provides.

## Reproduction Instructions

To reproduce these exact results on a fresh T4 (16GB VRAM) instance:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Adhitha-dev/conferece.git
   cd conferece
   ```
2. **Install exact dependencies:**
   ```bash
   cd reproduction_code
   pip install -r requirements.txt
   ```
   *Note: `vllm==0.27.0` and `bitsandbytes` are strictly required to fit the models onto a T4 without OOM errors.*
3. **Download the models:**
   Copy `.env.example` to `.env`, insert your HuggingFace token, and run:
   ```bash
   python download_models.py
   ```
4. **Run the experiments:**
   ```bash
   ./run_all_c1.sh
   ./run_all_c2.sh
   ```
   The scripts will automatically boot the server, apply the exact 8-bit quantization flags, run the inferences, and output the data directly into the `../experiments/` directory at the root.

