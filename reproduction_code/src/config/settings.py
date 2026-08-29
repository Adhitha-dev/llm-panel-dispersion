import os
import hashlib
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    experiment_root: str = "../experiments"
    data_dir: str = "data"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# A utility to hash prompt strings
def get_prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:8]
