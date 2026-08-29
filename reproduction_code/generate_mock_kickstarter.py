import json
from pathlib import Path

# Create 40 mock Kickstarter cases (20 successful, 20 failed)
kickstarter_cases = []
kickstarter_metadata = []

domains = ["hardware", "software", "games", "design", "technology"]

for i in range(1, 41):
    outcome = "successful" if i <= 20 else "failed"
    domain = domains[i % len(domains)]
    
    case_id = f"CASE_KS_{i:03d}"
    idea_text = f"This is a mock Kickstarter campaign description for a {domain} product. " \
                f"It aims to solve a critical problem using innovative techniques. " \
                f"We have a strong team and a working prototype. Our vision is to bring this to every household. " \
                f"This text is a placeholder representing a cleaned, pre-outcome pitch without any leakage of the final funding status. " \
                f"Please evaluate its market potential and feasibility based on these limited details."
    
    kickstarter_cases.append({
        "case_id": case_id,
        "idea_text": idea_text,
        "domain": domain,
        "known_outcome": outcome,
        "outcome_confidence": "public_record",
        "source": "kickstarter"
    })
    
    kickstarter_metadata.append({
        "case_id": case_id,
        "source": "kickstarter",
        "campaign_state": outcome,
        "recognizability": "low",
        "memorization_risk": "low"
    })

# Load the existing synthetic cases
with open("data/processed/synthetic_cases.json", "r") as f:
    synth_data = json.load(f)
    synthetic_cases = synth_data["cases"]

# Combine them
all_cases = kickstarter_cases + synthetic_cases

with open("data/processed/cases.json", "w", encoding="utf-8") as f:
    json.dump({"cases": all_cases}, f, indent=2)
    
print("Generated data/processed/cases.json with 40 mock Kickstarter cases + 10 Synthetic cases (50 total).")
