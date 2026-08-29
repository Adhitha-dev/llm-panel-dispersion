import asyncio
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.schemas.models import Case, JudgmentProfile
from src.dataset.loader import load_cases
from src.inference.client import InferenceClient, extract_json_from_response
from src.storage.logger import RunLogger
from src.config.prompt import EVALUATION_PROMPT_TEMPLATE
from src.config.settings import get_prompt_hash, settings
from pydantic import ValidationError

async def run_evaluation_task(
    client: InferenceClient,
    logger: RunLogger,
    case: Case,
    model: str,
    model_snapshot: str,
    provider: str,
    endpoint: str,
    condition: str,
    replicate_index: int,
    temperature: float,
    prompt_hash: str
):
    # Stagger execution based on replicate index to prevent concurrent request deduplication in vLLM
    await asyncio.sleep(replicate_index * 0.2)
    
    prompt_text = EVALUATION_PROMPT_TEMPLATE.format(idea_text=case.idea_text)
    
    # Task base info
    task_info = {
        "case_id": case.case_id,
        "model": model,
        "model_snapshot": model_snapshot,
        "provider": provider,
        "endpoint": endpoint,
        "condition": condition,
        "replicate_index": replicate_index,
        "prompt_version_hash": prompt_hash,
        "temperature": temperature,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    attempt = 1
    max_attempts = 3
    success = False
    raw_response = ""
    
    while attempt <= max_attempts and not success:
        log_entry = task_info.copy()
        log_entry["attempt_index"] = attempt
        
        try:
            raw_response = await client.generate_response(model=model_snapshot, prompt=prompt_text, temperature=temperature)
            log_entry["raw_response_text"] = raw_response
            
            # Parse & Validate
            parsed_json, json_valid = extract_json_from_response(raw_response)
            
            if json_valid:
                try:
                    profile = JudgmentProfile(**parsed_json)
                    log_entry.update({
                        "parsed_score": profile.score,
                        "parsed_verdict": profile.verdict,
                        "parsed_confidence": profile.confidence,
                        "parsed_rubric_market_potential": profile.rubric_scores.market_potential,
                        "parsed_rubric_technical_feasibility": profile.rubric_scores.technical_feasibility,
                        "parsed_rubric_business_viability": profile.rubric_scores.business_viability,
                        "parsed_reasoning": profile.reasoning,
                        "validity_flag": True
                    })
                    success = True
                except ValidationError as e:
                    log_entry["validity_flag"] = False
                    log_entry["error"] = "SCHEMA_INVALID"
                    log_entry["error_details"] = str(e)
                    # We do NOT retry schema invalid outputs based on the spec
                    success = True # loop ends
            else:
                log_entry["validity_flag"] = False
                log_entry["error"] = "INVALID_JSON"
                # We do NOT retry invalid JSON outputs based on the spec
                success = True # loop ends
                
        except Exception as e:
            log_entry["validity_flag"] = False
            log_entry["error"] = "HTTP_ERROR_OR_TIMEOUT"
            log_entry["error_details"] = str(e)
            attempt += 1 # We retry HTTP errors

        # Append to raw log
        logger.log_raw_attempt(log_entry)
        
        # If the attempt was conclusive (even if invalid schema), log to parsed outputs and break
        if success or attempt > max_attempts:
            logger.log_parsed_result(log_entry)
            break
