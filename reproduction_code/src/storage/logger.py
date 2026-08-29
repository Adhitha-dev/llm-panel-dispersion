import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

class RunLogger:
    def __init__(self, raw_log_path: str | Path, parsed_valid_path: str | Path, parsed_invalid_path: str | Path):
        self.raw_log_path = Path(raw_log_path)
        self.parsed_valid_path = Path(parsed_valid_path)
        self.parsed_invalid_path = Path(parsed_invalid_path)
        
        for path in [self.raw_log_path, self.parsed_valid_path, self.parsed_invalid_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
            
    def log_raw_attempt(self, record: Dict[str, Any]):
        """Logs every single inference attempt (append-only JSONL)"""
        with open(self.raw_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
            
    def log_parsed_result(self, record: Dict[str, Any]):
        """Logs parsed outputs (valid or invalid)"""
        is_valid = record.get("validity_flag", False)
        target_path = self.parsed_valid_path if is_valid else self.parsed_invalid_path
        with open(target_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
            
    @staticmethod
    def jsonl_to_csv(jsonl_path: str | Path, csv_path: str | Path):
        """Converts a JSONL file to a CSV file for analysis"""
        records = []
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            return
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    
        if records:
            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False)
