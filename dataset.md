GENTS.md
## Kickstarter Startup-Idea Dataset Builder

**Project:** LLM Panel Disagreement Harness — Startup Idea Evaluation  
**Purpose:** Build a reproducible dataset for measuring within-model variability and between-model disagreement when LLMs evaluate startup ideas.

---

## 1. Objective

Build a clean, leakage-controlled dataset of startup-oriented Kickstarter campaign pitches for the experiment described in the project specification.

The dataset must support:

1. **C1 — Same-model replicates:** repeated evaluation of the same case.
2. **C2 — Cross-model panel:** one evaluation per model for the same case.
3. Outcome-level analysis using score and verdict.
4. Rubric-level analysis using market potential, technical feasibility, and business viability.
5. Confidence analysis.
6. Source and recognizability analysis.
7. Optional comparison between real Kickstarter cases and synthetic cases.

The dataset is not intended to estimate general startup success. The primary observed label is **Kickstarter campaign outcome**.

Do not describe `successful` or `failed` as universal startup-success labels.

---

## 2. Scope Decision

Use Kickstarter as the primary and, unless explicitly changed in the experiment manifest, the only real-world source.

Do not mix YC, Product Hunt, Hacker News, Indiegogo, Crunchbase, or other sources into the main dataset. Mixing sources would introduce differences in writing style, selection process, outcome definitions, and platform behavior that could be mistaken for model disagreement.

A small synthetic split is allowed and recommended for contamination analysis, but it must remain clearly separated from Kickstarter cases.

Recommended final composition:

```text
40 Kickstarter cases
10 synthetic matched cases
50 cases total
```

If time or access is limited, the minimum acceptable version is:

```text
40–50 Kickstarter cases only
```

If synthetic cases are omitted, record that decision as a limitation.

---

## 3. Required Output Format

The evaluator-facing dataset MUST be valid JSON and MUST strictly follow this structure:

```json
{
  "cases": [
    {
      "case_id": "CASE_KS_001",
      "idea_text": "Cleaned pre-outcome Kickstarter campaign description.",
      "domain": "hardware",
      "known_outcome": "successful",
      "outcome_confidence": "public_record",
      "source": "kickstarter"
    }
  ]
}
```

For synthetic cases:

```json
{
  "case_id": "CASE_SYNTH_001",
  "idea_text": "Researcher-authored startup idea.",
  "domain": "enterprise software",
  "known_outcome": null,
  "outcome_confidence": null,
  "source": "synthetic"
}
```

### Strict schema rules

- The top-level object must contain exactly one key: `cases`.
- Each case must contain exactly these six keys:
  - `case_id`
  - `idea_text`
  - `domain`
  - `known_outcome`
  - `outcome_confidence`
  - `source`
- Do not add URLs, creator names, funding amounts, dates, categories, or metadata to the evaluator-facing JSON.
- Use valid JSON, not Python dictionary syntax.
- Do not include comments or trailing commas.
- `case_id` values must be unique and immutable after the experiment begins.

---

## 4. Data Sources

### 4.1 Primary source: Kickstarter

Use a historical Kickstarter dataset or a documented, permitted collection process. Prefer a reproducible historical snapshot over an undocumented live scrape.

Potential source types include:

- Web Robots Kickstarter historical snapshots.
- A documented research dataset with clear provenance.
- A permitted, reproducible export or scrape if no suitable snapshot contains the required text.

Record the exact source, snapshot date, retrieval date, URL, file name, and hash in the private metadata and experiment manifest.

### 4.2 Synthetic source

Synthetic cases must be authored specifically for this experiment. They are not assigned real-world outcomes.

Synthetic cases should be used to reduce memorization concerns and test controlled mechanisms such as:

- Persuasive versus neutral wording.
- Strong versus weak technical feasibility.
- Broad versus narrow market positioning.
- Clear versus ambiguous business models.
- High versus low regulatory burden.

Synthetic cases must be analyzed separately from Kickstarter cases.

---

## 5. Kickstarter Inclusion Criteria

Include campaigns that satisfy all of the following:

- Final campaign outcome is available and reliable.
- Campaign description contains enough substantive text for evaluation.
- Campaign was publicly available before its outcome was determined.
- Campaign fits the operational definition of a startup-oriented product, service, or venture.
- Campaign belongs to a business-relevant category such as technology, software, hardware, gadgets, product design, business, or an equivalent category.
- Description can be cleaned without destroying its meaning.
- Campaign is not an obvious duplicate.
- Creator duplication is controlled.

### Preferred content profile

Keep descriptions that contain enough information to judge at least some of:

- The problem.
- The proposed solution.
- Intended users or customers.
- Product or service mechanism.
- Business or funding rationale.
- Feasibility clues.

A description may be incomplete. Do not rewrite it to make it stronger or more coherent.

---

## 6. Kickstarter Exclusion Criteria

Exclude campaigns that satisfy any of the following:

- Empty, extremely short, or unusable description.
- Music, film, theatre, personal event, charity, political campaign, or purely artistic project unless explicitly included in the pre-registered domain design.
- Description is mostly reward tiers, shipping information, biographies, or thank-you text.
- Duplicate or near-duplicate campaign.
- Multiple campaigns from the same creator for the same product.
- Campaign has a title or description that explicitly reveals the final outcome.
- Campaign includes post-outcome updates or comments in the text field used as `idea_text`.
- Campaign metadata is internally contradictory or impossible to verify.
- Text is dominated by HTML, navigation content, tracking text, or scraper artifacts.
- The project is clearly an established company fundraising for a mature product rather than an early-stage idea, unless the sampling plan explicitly includes it.
- The campaign cannot be assigned to a domain with reasonable confidence.

Do not exclude difficult or strange ideas merely because they appear weak. The experiment is about judgment behavior, not creating a collection of high-quality startups.

---

## 7. Outcome Definition

The real-world label must represent the final Kickstarter campaign state, not general company success.

Use these canonical values:

```text
successful
failed
canceled
suspended
```

For the primary balanced analysis:

- Prefer `successful` and `failed`.
- Exclude `canceled` and `suspended` from the main binary comparison unless there are enough cases to analyze them separately.
- Never silently map `canceled` or `suspended` to `failed`.

For the evaluator-facing file:

```json
"known_outcome": "successful",
"outcome_confidence": "public_record"
```

or:

```json
"known_outcome": "failed",
"outcome_confidence": "public_record"
```

For cases with uncertain or incomplete status:

```json
"known_outcome": null,
"outcome_confidence": null
```

The model must not receive the outcome during inference. The field is retained in the dataset because the runner must separate visible prompt fields from hidden analysis metadata.

---

## 8. Sampling Design

Do not take the first 50 usable records. Use stratified selection.

### Recommended 40-case Kickstarter sample

```text
20 successful campaigns
20 failed campaigns
```

Balance as much as practical across:

- Domain.
- Campaign year.
- Pitch length.
- Funding goal scale.
- Outcome margin.
- Recognizability.

### Domain target

Use approximately 5–10 cases per domain group, depending on data availability:

- Software and developer tools.
- Hardware and electronics.
- Consumer products and gadgets.
- Energy, climate, or sustainability technology.
- Health, education, business, or other technology-oriented products.

Do not force exact balance if it requires changing the inclusion rules. Record the final counts.

### Outcome-margin strata

Where funding fields are available, retain hidden metadata for:

- Clearly successful: substantially above goal.
- Borderline successful: close to goal but reached it.
- Borderline failed: close to goal but did not reach it.
- Clearly failed: substantially below goal.

Do not use outcome-margin information in `idea_text`.

Recommended minimum for 40 Kickstarter cases:

```text
10 clearly successful
10 borderline successful
10 borderline failed
10 clearly failed
```

If this cannot be achieved, use a balanced successful/failed sample and document the actual distribution.

### Creator and category controls

- No more than one campaign per creator in the default sample.
- No more than two cases from one narrow subcategory.
- Do not allow multiple versions of the same product.
- Avoid selecting only famous campaigns.

---

## 9. Recognizability and Contamination Controls

Models may recognize famous campaigns or companies from training data. This can inflate agreement or produce knowledge-based judgments rather than fresh evaluation.

For every real case, assign private metadata:

```text
recognizability:
  high | medium | low | unknown

memorization_risk:
  high | medium | low | unknown
```

### Recognizability guidance

`high` may include:

- Famous campaigns.
- Widely reported products.
- Campaigns associated with major brands.
- Projects with substantial media coverage.

`low` may include:

- Older, obscure campaigns.
- Limited-coverage projects.
- Small creators with weak web presence.

Do not remove all recognizable cases. Keep a small high-recognizability subset so contamination can be measured descriptively.

Recommended distribution:

```text
Low recognizability: 20–25 cases
Medium recognizability: 10–15 cases
High recognizability: 5–10 cases
```

Do not claim that an “obscure” case is uncontaminated. Use the wording “lower estimated memorization risk.”

### Text anonymization variant

If feasible, create an internal anonymized variant for a subset:

- Remove campaign title.
- Remove creator name.
- Remove company or product brand names.
- Remove URLs and social handles.
- Replace identifying names with neutral placeholders.

Do not use the anonymized and visible versions as independent cases in the same main analysis. They are paired variants.

---

## 10. Text Construction Rules

The `idea_text` must be a cleaned version of the pre-outcome campaign description.

### Keep

- Problem statement.
- Proposed solution.
- Intended users.
- Product mechanism.
- Relevant business model information.
- Technical description.
- Claims made by the campaign creator.
- Persuasive language that was part of the original pitch.

### Remove

- Funding state.
- Pledged amount.
- Backer count.
- Percentage funded.
- Outcome terms such as “we succeeded,” “fully funded,” or “we failed.”
- Post-campaign updates.
- Comments and reactions.
- Staff Pick or platform-ranking labels.
- Navigation and scraper artifacts.
- Tracking links and unnecessary URLs.
- Direct instructions to backers to pledge.
- Reward tiers unless they are necessary to understand the business model.

Do not manually improve grammar, strengthen claims, summarize weaknesses, or rewrite the pitch into a neutral form for the main dataset.

The original persuasive wording is scientifically relevant because the project concerns judgment under persuasive startup-pitch text.

### Length rules

Recommended:

- Minimum: 80 words after cleaning.
- Preferred: 150–1,500 words.
- Maximum: 2,500 words for the primary dataset.

If a description is longer than 2,500 words:

1. Prefer a deterministic truncation rule, or
2. Extract a predefined section such as the main campaign description, or
3. Create a separate long-text subset.

Do not summarize long descriptions with another LLM before evaluation unless summarization is itself a declared experimental condition.

Record word count and character count in private metadata.

---

## 11. Hidden Metadata Schema

The evaluator-facing JSON must remain minimal. Store the following in a separate internal metadata file such as `case_metadata.jsonl`:

```json
{
  "case_id": "CASE_KS_001",
  "source": "kickstarter",
  "source_id": "platform-specific-id",
  "source_url": "https://...",
  "retrieved_at": "2026-08-21T00:00:00Z",
  "snapshot_date": "2018-06-01",
  "raw_title": "Original campaign title",
  "raw_description_hash": "sha256:...",
  "clean_text_hash": "sha256:...",
  "campaign_state": "successful",
  "outcome_definition": "reached_funding_goal",
  "goal_amount": 25000.0,
  "pledged_amount": 30000.0,
  "backer_count": 500,
  "currency": "USD",
  "category_raw": "Technology",
  "category_normalized": "hardware",
  "launch_date": "2018-01-01",
  "deadline": "2018-02-01",
  "duration_days": 31,
  "description_word_count": 420,
  "recognizability": "low",
  "memorization_risk": "low",
  "creator_id_hash": "sha256:...",
  "duplicate_group": null,
  "cleaning_flags": [],
  "manual_review_status": "approved"
}
```

Do not include all metadata in the prompt.

---

## 12. Manual Review

Every selected case must pass manual review before entering the final dataset.

Review each case for:

- Is this genuinely startup-oriented?
- Is the idea understandable without external context?
- Is the text pre-outcome?
- Is the outcome label reliable?
- Is the domain assignment reasonable?
- Is the text too recognizable?
- Is there a duplicate or near-duplicate?
- Are there scraper artifacts?
- Does the case contain information that would trivially reveal success or failure?

Use these review statuses:

```text
pending
approved
rejected
needs_revision
```

Do not allow unreviewed cases into the full experiment.

---

## 13. Deduplication

Perform at least three duplicate checks:

1. Exact text hash.
2. Normalized title or campaign identifier.
3. Text similarity using TF-IDF or sentence embeddings.

Potential near-duplicates must be manually reviewed.

Do not use one product’s relaunch as a separate case unless the study explicitly investigates repeated campaigns.

If repeated campaigns are retained, assign a shared `duplicate_group` or `product_family_id` in hidden metadata and keep them out of the default independent-case analysis.

---

## 14. Data Leakage Tests

Before finalizing the dataset, run these checks:

- Search `idea_text` for outcome terms such as `successful`, `funded`, `failed`, `backers`, `pledged`, and `goal reached`.
- Check for dollar amounts that reveal campaign performance.
- Check whether HTML contains status badges.
- Check whether URLs contain state information.
- Check whether updates or comments were accidentally concatenated.
- Check whether campaign titles contain success indicators.
- Check whether metadata fields were accidentally included in the prompt.

Produce a leakage report containing:

```text
case_id
leakage_flag
matched_terms
manual_decision
```

Any unresolved leakage flag excludes the case from the primary experiment.

---

## 15. Synthetic Case Design

If the synthetic split is used, create 10 cases as five matched pairs.

Each pair should preserve the core business idea while changing one controlled factor:

1. Persuasive versus neutral wording.
2. Strong versus weak technical feasibility.
3. Broad versus narrow target market.
4. Clear versus ambiguous revenue model.
5. Low versus high regulatory complexity.

Do not assign real-world outcomes to synthetic cases.

Use this exact format:

```json
{
  "case_id": "CASE_SYNTH_001",
  "idea_text": "A researcher-authored startup idea.",
  "domain": "enterprise software",
  "known_outcome": null,
  "outcome_confidence": null,
  "source": "synthetic"
}
```

Store pair IDs and manipulation descriptions only in hidden metadata.

---

## 16. Dataset Splits

Create explicit splits rather than relying on random analysis after inference.

Recommended splits:

```text
all_cases
kickstarter_only
successful_kickstarter
failed_kickstarter
borderline_kickstarter
low_memorization_risk
medium_memorization_risk
high_memorization_risk
synthetic_only
persuasion_pairs
```

The primary paper result should be computed on `kickstarter_only`.

Synthetic and recognizability analyses are secondary or robustness analyses.

Do not select cases for a split based on model outputs from the same experiment unless the selection rule was predeclared.

---

## 17. Required Files

Create this directory structure:

```text
data/
  raw/
    source_snapshot_manifest.json
    raw_kickstarter_records.jsonl
  processed/
    cases.json
    case_metadata.jsonl
    case_review.csv
    data_dictionary.md
    dataset_card.md
  validation/
    leakage_report.csv
    duplicate_report.csv
    balance_report.csv
    validation_report.md
```

### `cases.json`

The exact evaluator-facing JSON structure with only the six required fields per case.

### `case_metadata.jsonl`

Private metadata for analysis and provenance.

### `case_review.csv`

Manual review decisions and reviewer notes.

### `data_dictionary.md`

Definitions for every field, including outcome semantics.

### `dataset_card.md`

Document source, collection period, inclusion rules, exclusions, limitations, licenses, and known biases.

---

## 18. Validation Requirements

The dataset builder must fail loudly if:

- `cases` is missing.
- Any case has an extra or missing field.
- A `case_id` is duplicated.
- `idea_text` is empty.
- `source` is not `kickstarter` or `synthetic`.
- A Kickstarter case has an invalid outcome value.
- A synthetic case has a non-null outcome.
- An outcome-bearing case lacks `public_record` or an explicitly documented confidence value.
- The same raw campaign appears more than once.
- A leakage flag is unresolved.
- A case lacks manual approval.

Run a final JSON parser and schema validator before the data is used by the inference runner.

---

## 19. Balance Report

Generate a report containing counts by:

- Source.
- Outcome.
- Domain.
- Campaign year.
- Pitch length bucket.
- Outcome-margin bucket.
- Recognizability.
- Memorization risk.
- Manual review status.

The report must identify severe imbalance but must not automatically delete valid cases solely to achieve perfect balance.

If the final sample is imbalanced, document the imbalance and use stratified or descriptive analysis accordingly.

---

## 20. Reproducibility

Record:

- Source URL or dataset location.
- Snapshot date.
- Retrieval date.
- Collection method.
- Code version or Git commit.
- Cleaning script hash.
- Raw data hash.
- Processed data hash.
- Manual-review completion date.
- Inclusion and exclusion counts.
- Final case IDs.

Never overwrite raw source files. If a case changes, create a new dataset version.

Use immutable version names such as:

```text
data_v001
 data_v002
```

Do not silently replace `cases.json` after the experiment has started.

---

## 21. Recommended Dataset Card Language

Use wording similar to this:

> This dataset contains a curated set of startup-oriented Kickstarter campaign descriptions and researcher-authored synthetic startup ideas. Kickstarter labels represent campaign outcomes—whether a campaign reached its funding goal—not general startup success, company survival, or investment quality. The dataset is designed to study within-model and between-model judgment variability under a fixed evaluation prompt. It is not representative of all startups or all entrepreneurial ventures. Potential limitations include platform selection bias, category imbalance, creator and marketing effects, survivorship and visibility bias, temporal drift, and memorization or contamination by language models.

---

## 22. Final Acceptance Criteria

The dataset is ready only when all of the following are true:

- `cases.json` follows the exact required format.
- The primary Kickstarter subset contains a documented successful/failed sampling plan.
- All cases have unique immutable IDs.
- All real cases have hidden source and outcome metadata.
- All synthetic cases have null outcomes.
- No evaluator-visible text contains obvious outcome leakage.
- Duplicates and near-duplicates were checked.
- Every case received manual review.
- A balance report exists.
- A leakage report exists.
- A dataset card exists.
- Raw and processed hashes are recorded.
- The final dataset can be regenerated from the manifest and scripts.

Do not start the LLM evaluation run until these acceptance criteria pass.

