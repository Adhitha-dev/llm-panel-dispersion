import os
import json
from typing import Dict, Any, Tuple
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

class InferenceClient:
    def __init__(self, provider: str = "openai", base_url: str = None, api_key: str = None):
        # Allow overriding via args, fallback to env vars based on provider
        env_base_url = os.getenv(f"LLM_BASE_URL_{provider.upper()}")
        env_api_key = os.getenv(f"LLM_API_KEY_{provider.upper()}")
        
        self.client = AsyncOpenAI(
            base_url=base_url or env_base_url,
            api_key=api_key or env_api_key or "sk-dummy"  # local inference doesn't always need an API key
        )
        
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_response(self, model: str, prompt: str, temperature: float = 0.0) -> str:
        """Call the LLM with retry logic for HTTP/network errors."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

def extract_json_from_response(raw_text: str) -> Tuple[Dict[str, Any], bool]:
    """Attempts to extract and parse JSON from the raw response text."""
    # Heuristically find JSON block in case the model added markdown blocks
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    # Strip any potential leading/trailing non-json stuff if the first char isn't {
    if text and text[0] != '{':
        start_idx = text.find('{')
        if start_idx != -1:
            text = text[start_idx:]
            
    if text and text[-1] != '}':
        end_idx = text.rfind('}')
        if end_idx != -1:
            text = text[:end_idx+1]

    try:
        parsed = json.loads(text)
        return parsed, True
    except json.JSONDecodeError:
        return {}, False
