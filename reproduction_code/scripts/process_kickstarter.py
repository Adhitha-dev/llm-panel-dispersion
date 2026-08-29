import pandas as pd
import json
import uuid
from pathlib import Path

def process():
    file_path = "data/raw/kickstarter/Kickstarter Full Campaigns_694062_25_April_2026.csv"
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter domains: Technology, Hardware, Software, Design
    valid_domains = ["Technology", "Hardware", "Software", "Web", "Apps", "Gadgets", "Product Design"]
    df = df[df["category_name"].isin(valid_domains) | df["category_parent_name"].isin(valid_domains)]
    
    # Filter outcome
    df_succ = df[df["state"] == "successful"].copy()
    df_fail = df[df["state"] == "failed"].copy()
    
    # Sort by blurb length to get the most substantive ones
    df_succ['blurb_len'] = df_succ['blurb'].astype(str).str.len()
    df_fail['blurb_len'] = df_fail['blurb'].astype(str).str.len()
    
    df_succ = df_succ.sort_values('blurb_len', ascending=False).head(50)
    df_fail = df_fail.sort_values('blurb_len', ascending=False).head(50)
    
    # Sample exactly 20 each
    succ_sample = df_succ.sample(20, random_state=42)
    fail_sample = df_fail.sample(20, random_state=42)
    
    combined = pd.concat([succ_sample, fail_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    kickstarter_cases = []
    kickstarter_metadata = []
    
    for idx, row in combined.iterrows():
        case_id = f"CASE_KS_{idx+1:03d}"
        
        # We merge name and blurb since this dataset lacks the full html description.
        # We append a generic pitch-style suffix if it's too short, but we'll try to just use name + blurb + category context.
        idea_text = f"{row['name']} ({row['category_name']})\n\n{row['blurb']}"
        
        kickstarter_cases.append({
            "case_id": case_id,
            "idea_text": str(idea_text),
            "domain": str(row['category_name']).lower(),
            "known_outcome": str(row['state']),
            "outcome_confidence": "public_record",
            "source": "kickstarter"
        })
        
        kickstarter_metadata.append({
            "case_id": case_id,
            "source": "kickstarter",
            "campaign_state": str(row['state']),
            "recognizability": "unknown",
            "memorization_risk": "unknown",
            "raw_title": str(row['name'])
        })
        
    # Read synthetic cases
    with open("data/processed/synthetic_cases.json", "r") as f:
        synth_data = json.load(f)
        synthetic_cases = synth_data["cases"]
        
    # Combine
    all_cases = kickstarter_cases + synthetic_cases
    
    with open("data/processed/cases.json", "w", encoding="utf-8") as f:
        json.dump({"cases": all_cases}, f, indent=2)
        
    with open("data/processed/case_metadata.jsonl", "w", encoding="utf-8") as f:
        for m in kickstarter_metadata:
            f.write(json.dumps(m) + "\n")
            
    print(f"Successfully processed {len(kickstarter_cases)} real Kickstarter cases.")
    print(f"Total cases in dataset: {len(all_cases)} (40 KS + 10 Synthetic)")

if __name__ == "__main__":
    process()
