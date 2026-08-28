# PROJECT BUILD SPECIFICATION
## LLM Panel Disagreement Harness — Startup Idea Evaluation

**Document type:** Agent-facing implementation specification
**Status:** Build specification
**Timeline:** 1–2 days, solo.

**Primary objective:** Implement a fast, honest, reproducible experiment that tests whether a panel of LLM evaluators disagreeing on a startup idea is (a) structured/informative rather than pure noise, and (b) whether that structure behaves differently in startup evaluation than it does in already-studied domains (NLI, video-judge tasks, exam grading), because of domain-specific mechanisms: no eventual ground truth, persuasion-susceptibility, and memorization/contamination confounds.

**Framing note (aligns build with the paper draft):** C1 is *not* interpreted as pure stochastic noise by default. A startup idea is multidimensional — a model can validly emphasize different trade-offs (market size vs. feasibility vs. unit economics) on different reruns and land on different-but-defensible scores. So C1 is reported at three levels — **outcome** (score/verdict, §10.1a), **rubric** (per-dimension profile, §10.1b) — with **reasoning** similarity (§10.1c) as an optional exploratory extra — rather than collapsed into a single "noise" number. The term "noise floor" below is shorthand for the *outcome-level* component only; whether it's noise or legitimate pluralism is what the rubric-level and four-pattern analysis (§10.7) actually tests, not an assumption baked into the metric.

---

# 1. SYSTEM OBJECTIVE

Build a harness that:

1. Loads a dataset of startup idea cases (idea text + optional real-world outcome label).
2. Runs two experiment conditions:
   - **C1 — Same-model replicates:** one model, same idea, N repeated calls → measures the model's own instability (noise floor).
   - **C2 — Cross-model panel:** several different models, same idea, one call each → measures real cross-model disagreement.
3. Forces every evaluator call to return a strict structured JSON output: a **judgment profile** consisting of score, verdict, confidence, short reasoning, *and* a reduced 3-dimension rubric (§7) — not just a single scalar score.
4. Logs raw requests/responses, parsed outputs, and full run metadata — nothing overwritten.
5. Computes a fixed set of stability, agreement, and panel-spread metrics from the two conditions.
6. Produces a results table and a handful of plots directly usable in the paper.

The system MUST distinguish:

- **Stochastic instability** (C1): does the same model's judgment move across reruns, even at temperature 0?
- **Cross-model disagreement** (C2): do different models land on different judgments/verdicts on the same idea?
- **Correlated non-independence**: are the "different" models actually just echoing each other (high pairwise score correlation), meaning the panel has fewer effective viewpoints than its nominal size?

The system MUST NOT treat raw disagreement counts as automatically meaningful — it must compare C2 disagreement against the C1 noise floor before concluding anything.

---

# 2. NON-GOALS

Do not build:

- Evidence retrieval or web search during evaluation (each judge sees only the idea text and the fixed prompt).
- Evidence-credibility scoring or claim-level source auditing.
- A temporal leakage detection pipeline (name the risk in the paper's limitations instead — see §9).
- Personalized per-evaluator preference modeling.
- Agent-to-agent debate, shared memory between evaluators, or evaluators seeing each other's output.
- Persona-conditioned prompting.
- Pairwise AB/BA comparative judging (this harness scores ideas independently, not head-to-head — see §7 for why this matters for what metrics apply).
- Any UI.

Every evaluator call MUST be conditionally independent at inference time — it sees only the idea text and the fixed prompt, nothing else, no tools, no retrieval, no memory, no mid-run prompt changes.

---

# 3. ARCHITECTURE

```text
                 ┌───────────────────┐
                 │ Case Dataset (CSV/JSON) │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Task Generator   │  (idea × model × condition × replicate_index)
                 └─────────┬─────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Model A calls  Model B calls  Model C/D calls
              │            │            │
              └────────────┼────────────┘
                           ▼
                 ┌───────────────────┐
                 │  Raw Run Logger   │  (append-only JSONL)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │  JSON Validator   │  (schema check, reject/retry)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Results Table     │  (one row per valid judgment)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │  Metrics Script   │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Tables + Figures  │
                 └───────────────────┘
```

---

# 4. TECHNOLOGY STACK

## 4.1 Runtime

- Python 3.10+.
- T4-class local GPU (16GB VRAM) for any locally-served model(s); remote API calls for hosted models (GPT/Claude/Gemini as available). Because the T4 has limited VRAM, 7B parameter models are the absolute maximum size for unquantized inference, and memory utilization must be strictly managed.
- OpenAI-compatible HTTP interface where possible, so the runner isn't hard-coded to one server.
- **CRITICAL DEPENDENCIES:** Must use exactly `vllm==0.27.0`. Newer versions (0.28.0+) drop support for the required `bitsandbytes` quantization. You must also install the `bitsandbytes` and `seaborn` Python libraries manually if using a fresh clone.
- **CRITICAL VLLM ARGS (For T4):** When serving 7B-8B models on a 16GB T4, you MUST use: `--quantization bitsandbytes --load-format bitsandbytes --max-model-len 2048 --gpu-memory-utilization 0.99`.

## 4.2 Environment variables

```text
LLM_BASE_URL_<PROVIDER>
LLM_API_KEY_<PROVIDER>
EXPERIMENT_ROOT
MODEL_CONFIG_PATH
```

## 4.3 Python components

```text
src/
  config/        # model list, prompt text, run config
  dataset/       # load + validate cases
  inference/     # per-provider client wrappers, unified call interface
  schemas/       # pydantic schema for the judge output
  runner/        # task generation + execution loop
  storage/       # append-only raw logger, parsed table writer
  metrics/       # stability, agreement, panel-spread calculations
  reporting/     # figures from stored tables
  cli/
tests/
data/
experiments/
```

Recommended libraries: `pydantic`, `httpx`, `pandas`, `numpy`, `scipy` (for kappa/ICC if used), `matplotlib`, `tenacity`, `typer`.

---

# 5. DATASET SPECIFICATION

## 5.1 Case schema

```json
{
  "case_id": "CASE_0001",
  "idea_text": "One-paragraph description of the startup idea, written as a founder would pitch it.",
  "domain": "cybersecurity",
  "known_outcome": "funded",
  "outcome_confidence": "public_record",
  "source": "YC public batch list / Kaggle startup dataset / synthetic"
}
```

- `case_id` immutable once an experiment starts.
- `known_outcome` optional, may be null.
- No `evidence` object — judges see `idea_text` only, plus the fixed prompt.
- Mix real ideas (YC batch lists / Kaggle startup datasets, for outcome-alignment) with a few synthetic ones to reduce memorization risk (§9). Both go in the same file, `source` field distinguishes them.

## 5.2 Dataset size

30–60 cases. Do not exceed 60.

---

# 6. FIXED PARAMETERS (lock these before any run — record all of them in the manifest)

Uncontrolled choices here can flip conclusions on their own, independent of anything about the models. Fix every one of these and log the exact value used.

| Parameter | Value | Why |
|---|---|---|
| Primary temperature | **0.0** | Lower temperature is more stable/reproducible — this is your headline condition, not 0.7. Note for the paper: temperature 0 is *not* guaranteed deterministic across providers, so C1 (replicate reruns) is what actually measures residual instability, not an assumption. |
| Robustness subset | 0.7, run only on a subset of borderline/high-entropy cases identified from the temp-0 run | Exposes latent uncertainty as a secondary, exploratory check — not the headline result. |
| Prompt text | One fixed template, byte-identical across all models and both conditions; only `idea_text` changes | Prompt wording alone can flip judge verdicts. |
| Rubric/scale | One scale throughout: 1–10 overall score, `invest`/`pass` verdict, 0–100 confidence, plus 3 rubric dimensions on a 0–5 integer scale each (§7) | Mixing scales changes human/model alignment and makes cross-run comparison meaningless; the rubric dimensions are what make criterion-level drift visible instead of hidden behind a stable total score. |
| Output schema | Strict JSON-only, fixed fields, unknown fields rejected | See §7. |
| Inference mode | Single-pass — no retrieval, no tools, no memory, no self-correction, no few-shot examples | Keeps every call conditionally independent (required by §2). |
| Caching | Disabled for all C1 replicate calls | Cached/reused responses would silently fake independence and deflate flip rate to zero. |
| Retries | Logged explicitly with a new `attempt_index`; original failure never overwritten | Hidden retries also break independence if not tracked. |
| Model provenance | Record model name, exact snapshot/version string, provider, endpoint, and call date for every model used — nicknames like "gpt" or "claude" are not sufficient | Silent version drift mid-experiment invalidates comparisons; this is the single easiest thing to forget and the easiest to regret later. |
| Generation params (max tokens, top_p, etc.) | Identical across all judges in C2 | Variation should reflect model differences, not sampling configuration differences. |

---

# 7. STRUCTURED EVALUATOR OUTPUT (JUDGMENT PROFILE)

Each call returns a **judgment profile**, not a bare score — this is what the paper's §III-D and the profile-distance metric (§10.1b) depend on. The rubric set is deliberately reduced to 3 dimensions (not the earlier 6-dimension design) to keep parsing/validation cost low while still making criterion-level drift visible.

```json
{
  "score": 7,
  "verdict": "invest",
  "confidence": 75,
  "rubric_scores": {
    "market_potential": 4,
    "technical_feasibility": 3,
    "business_viability": 4
  },
  "reasoning": "One to two sentences, no more."
}
```

## 7.1 Validation rules

- `score`: integer, 1–10.
- `verdict`: `"invest"` or `"pass"` only.
- `confidence`: 0–100.
- `rubric_scores`: object with exactly the 3 keys above, each an integer 0–5. Missing key, extra key, or out-of-range value → `SCHEMA_INVALID` (§11), not silently coerced or defaulted.
- `reasoning`: soft cap ~60 words, don't over-enforce.
- Unknown top-level fields rejected.
- Invalid JSON logged as a failed inference, never patched.
- Retries get a new `attempt_index`; original failure record untouched.

## 7.2 What to log per call (not just the parsed fields)

Save, for every call, whether it succeeds or fails: `case_id`, `model`, `model_snapshot`, `condition` (C1/C2), `replicate_index`, `prompt_version_hash`, `temperature`, `raw_response_text`, `parsed_score`, `parsed_verdict`, `parsed_confidence`, `parsed_rubric_market_potential`, `parsed_rubric_technical_feasibility`, `parsed_rubric_business_viability`, `parsed_reasoning`, `validity_flag`, `timestamp`. Raw text and validity flags matter as much as the parsed numbers — invalid-output rate and formatting failures are themselves a reliability signal worth reporting, not noise to discard.

## 7.3 Reasoning-level similarity (optional, exploratory — paper §III-E-3)

Not required for the headline results. If time allows: embed each `reasoning` string with a lightweight sentence-embedding model (e.g. a small `sentence-transformers` model, CPU is fine at this volume) and compute cosine similarity across C1 reruns and across C2 judges per case. Report as a rough exploratory signal only — never as a validated agreement measure — per the caveat in §III-E-3 about model explanations reflecting surface heuristics rather than substantive judgment. If skipped, say so plainly in Limitations rather than silently dropping it from the paper.

Since this harness does **not** do pairwise AB/BA comparative judging (each judge scores each idea independently, not head-to-head), skip anything that assumes paired orderings — position bias and AB/BA mirroring don't apply here. That's a legitimate difference from some of the literature's setup, worth one sentence in your methods section so a reviewer doesn't wonder why it's missing.

---

# 8. EXPERIMENT CONDITIONS

## 8.1 C1 — SAME_MODEL_REPLICATES

- One model (your strongest/most-used one).
- Every case, N=10 independent calls at temperature 0.0. Caching off.
- Purpose: noise floor — how much does one model's judgment move on identical repeated input?

## 8.2 C2 — CROSS_MODEL_PANEL

- 3 strictly different model families: `Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct-v0.3`, and `Meta-Llama-3-8B-Instruct`. (Phi-3 architectures are fundamentally unsupported on `vLLM 0.27.0` and should not be used).
- Every case, one call each, temperature 0.0, identical generation params.
- Purpose: real cross-model disagreement, to be compared against the C1 floor.

## 8.3 Optional robustness slice

- Re-run C1 and C2 at temperature 0.7 on whichever cases came out highest-entropy/highest-variance at temperature 0.0. Report separately, framed explicitly as exploratory, not the headline result.

Every task (`case_id`, `model`, `condition`, `replicate_index`, `temperature`) gets a deterministic task ID so re-running the manifest regenerates identical task lists.

---

# 9. NAMED LIMITATION: MEMORIZATION / CONTAMINATION

Not solved by a detection system — named honestly instead:

- Flag which cases are real/potentially-famous startups vs. synthetic (`source` field).
- Optionally split metrics by `source` in analysis.
- State in Limitations: models may be recognizing real companies from training data rather than reasoning fresh, which could inflate or deflate agreement in ways this study doesn't control for.

---

# 10. METRICS AND FORMULAS

Compute all of these on the parsed, valid outputs only (validity flag = true). Report invalid-output rate separately as its own reliability signal.

## 10.1 Within-model judgment variability (from C1, per case, R=10 replicates)

### 10.1a Outcome level

**Flip rate** — fraction of replicates disagreeing with the model's own majority verdict:
```
FlipRate_i = 1 - max_c(n_ic) / R
```
where `n_ic` is the count of verdict category `c` across the R replicates for case `i`.

**Self-consistency rate** (equivalent, reported alongside flip rate since it's the form used in temperature-selection literature):
```
SCR_i = 1 - FlipRate_i
```

**Score variance / standard deviation** across replicates:
```
s_i^2 = (1/(R-1)) * Σ_r (x_ir - x̄_i)^2
s_i = sqrt(s_i^2)
```

Report flip rate, self-consistency, and score SD per model as the **primary stability** result at the outcome level.

### 10.1b Rubric level

**Rubric profile distance** between every pair of replicate runs `(A, B)` on the same case:
```
D(A,B) = Σ_i |A_i - B_i| / (n_dims * 5)
```
where `A_i`, `B_i` are the two runs' scores on rubric dimension `i` (of the 3 dimensions in §7), and `n_dims = 3`. Normalized to `[0, 1]`. Report the mean `D(A,B)` across all replicate pairs, per case and per model.

This is the metric that actually tests the framing note in the header: a case can have **low outcome-level flip rate but high rubric-level D(A,B)** — same final verdict reached via different emphasis each time — which is evidence for legitimate trade-off pluralism rather than instability. Conversely, high flip rate *and* high D(A,B) together is evidence the model is genuinely unstable, not just re-weighting the same dimensions differently. Do not report only one level and call it "C1 instability" — report both, and let §10.7 classify what the combination means.

### 10.1c Reasoning level (optional — see §7.3)

If reasoning embeddings were computed: report mean pairwise cosine similarity across C1 reruns per case, labeled explicitly as exploratory.

## 10.2 Cross-model / panel disagreement (from C2, per case, k judges)

**Verdict entropy** — how split the panel is on this idea (0 = unanimous, max at even split):
```
H_i = -Σ_c p_c * log2(p_c)
```
where `p_c` is the proportion of the panel choosing verdict `c` on case `i`. This is your key per-idea disagreement signal.

**Mean pairwise absolute deviation (MPAD)** — direct dispersion metric on numeric scores:
```
MPAD_i = (2 / (k_i * (k_i - 1))) * Σ_{a<b} |s_ia - s_ib|
```
over the member-level panel scores for item `i`.

**Range spread** (cheap, informative under adversarial/borderline cases):
```
Range_i = max_j(s_ij) - min_j(s_ij)
```

Report MPAD, range, and per-case entropy/majority-split as the **primary panel disagreement** result — this replaces plain score-spread as the headline dispersion metric.

## 10.3 Across all cases (structure, not per-idea)

**Pairwise model agreement matrix** — for every model pair, fraction of cases where verdicts match.

**Pairwise score correlation matrix** (Pearson) — do model A's and model B's scores move together across ideas? High correlation = redundant judges, not independent ones.

**Effective viewpoints** (Kish-style), using score correlation as the similarity measure:
```
EffectiveViewpoints = N / (1 + (N-1) * mean_pairwise_similarity)
```
With N=4–5 nominal models, report the effective number — "nominally 5 judges, effectively X." This is the single headline structure number.

## 10.4 Agreement against a label (only for cases with `known_outcome` — secondary, exploratory)

If treating `known_outcome` as a coarse ground-truth label (e.g. funded/not, survived/failed) and comparing against panel majority verdict:

**Accuracy** (report but don't lead with, since it overstates agreement under imbalanced classes):
```
Acc = (TP + TN) / N
```

**Cohen's kappa** — chance-corrected agreement, preferred over raw accuracy:
```
κ = (p_o - p_e) / (1 - p_e)
```
where `p_o = Acc` and `p_e` is chance agreement computed from the observed marginals.

Only compute this if there's a meaningful number of labeled cases (roughly 15+) — with fewer, report descriptively instead of leaning on a kappa value that isn't statistically stable at that N.

## 10.5 The comparison that actually answers the research question

- **C1 within-model variability vs. C2 disagreement**: is cross-model MPAD/entropy meaningfully larger than same-model score variance (outcome level, §10.1a)? If not, the panel isn't adding information beyond rerunning one model — this is the core test.
- **Entropy vs. perceived ambiguity**: bucket cases into low/medium/high verdict-entropy, spot-check whether high-entropy cases are the ones that read as genuinely contestable (your own read, or `known_outcome` where it exists and is itself borderline).
- **Outcome alignment** (where `known_outcome` exists): does high panel disagreement correlate with outcomes that were themselves surprising (failed despite good scores, succeeded despite mixed scores)? Exploratory only — small N, don't overclaim significance.

## 10.6 Explicitly out of scope, name as future work in the paper

Position bias and verbosity bias (require pairwise AB/BA judging, which this harness doesn't do), Krippendorff's alpha and ICC (add real value with more judges/replicates than this pilot's budget supports), chief–panel gap and leave-one-out sensitivity (assume an arbitration step this harness doesn't have). Listing these by name in a "future work" paragraph signals literature awareness without costing build time.

## 10.7 Four-pattern interpretation grid (paper §III-F — required, cheap to compute)

Classify each case (and the dataset as a whole) using the outcome-level C1 measure (§10.1a, e.g. score SD or flip rate, thresholded low/high by median split or a fixed cutoff you state explicitly) against the C2 measure (§10.2, e.g. MPAD or entropy, same thresholding approach):

| C1 (within-model) | C2 (between-model) | Interpretation |
|---|---|---|
| Low | Low | Stable task; judges converge |
| Low | High | Distinct, stable model dispositions dominate |
| High | Low | Individually unstable, but average behavior converges |
| High | High | Evaluation setting itself is highly uncertain |

Output: a count/proportion of cases per quadrant, plus 1–2 example `case_id`s per quadrant for the write-up. This is a bucketing/labeling step over metrics already computed in §10.1–10.2 — no new inference calls, no new schema, just a `pandas` groupby. Do not skip this: it's a named contribution in the paper draft (§III-F) and costs under an hour to implement.

---

# 11. ERROR HANDLING

```text
Error categories:
HTTP_ERROR
TIMEOUT
INVALID_JSON
SCHEMA_INVALID
EMPTY_RESPONSE
UNKNOWN_ERROR
```

- Log every failed attempt; preserve raw output if any exists.
- Retry: max 2, only for `HTTP_ERROR` / `TIMEOUT`. `INVALID_JSON` / `SCHEMA_INVALID` not retried — recorded as-is (a model that can't follow schema is itself a data point, and its invalid-output rate should be reported).
- Every retry gets a new `attempt_index`; nothing overwrites a prior record.

---

# 12. REPRODUCIBILITY

```text
experiments/EXP_<date>_<id>/
  manifest.yaml          # models + exact snapshot/version, temperatures, N replicates, prompt hash, dataset hash
  raw/inference_attempts.jsonl
  parsed/valid_responses.jsonl
  parsed/invalid_responses.jsonl
  tables/results.csv      # case_id, model, model_snapshot, condition, replicate_index, temperature, score, verdict, confidence, rubric_market_potential, rubric_technical_feasibility, rubric_business_viability, reasoning, validity_flag
  metrics/metrics.csv
  figures/
  logs/runner.log
```

`manifest.yaml` must record: model list with exact snapshot/version + provider + endpoint, temperature(s) used per condition, N replicates for C1, prompt text hash, dataset file hash, timestamp.

---

# 13. CLI

```text
project run-c1 --model <name> --n 10 --temperature 0.0
project run-c2 --models <name1,name2,...> --temperature 0.0
project run-robustness-subset --temperature 0.7 --case-ids <...>
project parse-results
project compute-metrics
project make-figures
```

---

# 14. FIGURES

1. Same-model score distribution across 10 reruns — the outcome-level within-model variability.
2. Pairwise model agreement heatmap (C2).
3. Score correlation heatmap (C2) — supports the effective-viewpoints number.
4. Verdict entropy per case, sorted — which ideas the panel actually disagrees on.
5. C1 outcome-level variability vs. C2 MPAD, side-by-side — the core comparison.
6. (Optional, if time) Rubric-level D(A,B) distribution across C1 replicate pairs, alongside figure 1 — shows whether variability concentrates at outcome or rubric level.

All figures generated from `tables/results.csv` and `metrics/metrics.csv` only.

---

# 15. IMPLEMENTATION ORDER

## Phase 1 — Core
1. Dataset loader + schema check.
2. Fixed prompt template.
3. Inference client wrapper (model name + idea text → raw response).
4. JSON schema validator.
5. Raw append-only logger.
6. Results table writer.

## Phase 2 — Run conditions
1. C1 at temperature 0.0.
2. C2 at temperature 0.0.
3. (If time) robustness subset at 0.7 on the highest-entropy/highest-variance cases from step 1–2.

## Phase 3 — Metrics
Order: flip rate → self-consistency → score variance/SD (C1 outcome, §10.1a) → rubric profile distance D(A,B) (C1 rubric, §10.1b) → verdict entropy → MPAD → range (C2, §10.2) → pairwise agreement matrix → score correlation matrix → effective viewpoints (§10.3) → C1-vs-C2 comparison (§10.5) → four-pattern classification (§10.7) → entropy-vs-ambiguity bucketing → kappa/outcome alignment (if labels exist) → reasoning-embedding similarity (§7.3, only if time remains).

## Phase 4 — Figures + write-up
Generate figures from stored tables only. Write in parallel with figure polish, not after.

---

# 16. MINIMUM ACCEPTANCE CHECKS

1. Re-running task generation from the same manifest produces the same task list.
2. A deliberately malformed model response shows up in `invalid_responses.jsonl`, not silently dropped or repaired.
3. Feeding identical judgments manually: variance, MPAD, and entropy all compute to ~0.
4. No evaluator prompt in the logs contains anything beyond `idea_text` + the fixed instruction.
5. Model snapshot/version strings are actually recorded and non-empty in the manifest and results table for every model used.
6. A response missing a `rubric_scores` key, or with a rubric value outside 0–5, is rejected as `SCHEMA_INVALID` — not defaulted to a mid-scale value or silently dropped.
7. Feeding two identical judgment profiles (same score, verdict, and all 3 rubric scores) computes `D(A,B) = 0`; feeding maximally different rubric profiles (0 vs. 5 on every dimension) computes `D(A,B) = 1`.

---

# 17. DEFAULT PILOT CONFIGURATION

```text
cases: 40
conditions: 2  (C1, C2)
same_model_replicates: 10
cross_model_panel_size: 4
primary_temperature: 0.0
robustness_temperature: 0.7 (subset only)
max_output_tokens: 200
caching: disabled
```

Smoke test before the full run:

```text
cases: 3
replicates: 3
models: 2
temperature: 0.0
```

---

# 18. PRIMARY RESEARCH OUTPUT

The system must let this statement be supported or rejected with measured numbers:

> Cross-model disagreement on startup idea evaluation (MPAD, entropy) exceeds same-model rerun variability (flip rate, score SD) at temperature 0.0, and this disagreement is structured — concentrated on specific ideas rather than spread uniformly. Within-model variability itself is decomposed into outcome-level and rubric-level components (§10.1), distinguishing genuine instability from legitimate trade-off pluralism where the final verdict is stable but the underlying rubric emphasis shifts between runs. This is consistent with, but not identical to, disagreement patterns reported for LLM-judge panels in other domains (exam grading, NLI, video quality). Startup evaluation's domain-specific properties — no eventual ground truth, persuadability of pitch text, memorization risk for real/famous companies — are examined as plausible reasons the disagreement structure might differ.

Report via the tables/figures in §14 — not prose claims without the numbers behind them.

---

# 19. IMPLEMENTATION PRINCIPLE

- Fix the protocol before computing any statistic — temperature, prompt, schema, provenance, caching — because uncontrolled choices here can change conclusions on their own.
- Raw data immutable, every inference traceable, invalid output preserved not repaired.
- The 3-dimension rubric (§7) and profile distance D(A,B) (§10.1b) and four-pattern grid (§10.7) are required, not optional — they're what the paper's contributions list actually claims. The reasoning-embedding metric (§7.3, §10.1c) is the one piece that stays optional/exploratory; if cut, say so in Limitations rather than leaving the paper's claim unsupported.
- Keep the metric set in §10 fixed; the §10.6 list stays out unless there's genuinely spare time.
- If a choice is between "more rigorous" and "actually finished by the deadline," finished wins — state the resulting limitations honestly.
