import json
from pathlib import Path
from typing import List
from src.schemas.models import Case

def load_cases(file_path: str | Path) -> List[Case]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
        
    cases = []
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and "cases" in data:
                data = data["cases"]
            for item in data:
                cases.append(Case(**item))
    elif file_path.suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    cases.append(Case(**json.loads(line)))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Expected .json or .jsonl")
        
    return cases
