import asyncio
import httpx
import json
import json
import sys

sys.path.append('.')
from src.dataset.loader import load_cases
from src.config.prompt import EVALUATION_PROMPT_TEMPLATE

async def main():
    cases = {c.case_id: c for c in load_cases("data/processed/cases.json")}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # TEST 1: Temperature 0.7 Sanity Check
        print("\n--- TEST 1: Temp 0.7 on CASE_KS_002 ---")
        prompt = EVALUATION_PROMPT_TEMPLATE.format(idea_text=cases["CASE_KS_002"].idea_text)
        payload = {
            "model": "models/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        for i in range(5):
            resp = await client.post("http://localhost:8000/v1/chat/completions", json=payload)
            text = resp.json()["choices"][0]["message"]["content"]
            print(f"Run {i} Response excerpt: {text.strip()[:100].replace(chr(10), ' ')}")
            
        # TEST 2: Invalidate Prefix Caching (Temp 0.0)
        print("\n--- TEST 2: Prefix Invalidation on CASE_KS_001 (Temp 0.0) ---")
        base_prompt = EVALUATION_PROMPT_TEMPLATE.format(idea_text=cases["CASE_KS_001"].idea_text)
        for i in range(5):
            perturbed_prompt = base_prompt + f"\n<!-- eval_run_id: {i} -->"
            payload = {
                "model": "models/Mistral-7B-Instruct-v0.3",
                "messages": [{"role": "user", "content": perturbed_prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
            }
            resp = await client.post("http://localhost:8000/v1/chat/completions", json=payload)
            text = resp.json()["choices"][0]["message"]["content"]
            print(f"Run {i} Response excerpt: {text.strip()[:100].replace(chr(10), ' ')}")
            
        # TEST 3: Logprob Margin Analysis
        print("\n--- TEST 3: Logprob Analysis ---")
        for case_id in ["CASE_KS_003", "CASE_KS_014"]:
            print(f"Checking {case_id}:")
            prompt = EVALUATION_PROMPT_TEMPLATE.format(idea_text=cases[case_id].idea_text)
            payload = {
                "model": "models/Mistral-7B-Instruct-v0.3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100,
                "logprobs": True,
                "top_logprobs": 5
            }
            resp = await client.post("http://localhost:8000/v1/chat/completions", json=payload)
            data = resp.json()
            # Just grab the first few tokens' logprobs
            if "choices" in data and len(data["choices"]) > 0:
                logprobs_content = data["choices"][0]["logprobs"]["content"]
                for tok_info in logprobs_content[:15]:  # show first 15 tokens
                    token = tok_info["token"]
                    if "score" in token.lower() or "verdict" in token.lower() or token.strip().isdigit():
                        print(f"Token: {repr(token)}")
                        for top in tok_info.get("top_logprobs", []):
                            print(f"  {repr(top['token'])}: {top['logprob']}")

if __name__ == "__main__":
    asyncio.run(main())
