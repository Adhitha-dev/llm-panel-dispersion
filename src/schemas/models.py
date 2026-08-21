from typing import Literal, Optional
from pydantic import BaseModel, Field

class Case(BaseModel):
    case_id: str
    idea_text: str
    domain: str
    known_outcome: Optional[str] = None
    outcome_confidence: Optional[str] = None
    source: str

class RubricScores(BaseModel):
    market_potential: int = Field(ge=0, le=5)
    technical_feasibility: int = Field(ge=0, le=5)
    business_viability: int = Field(ge=0, le=5)

class JudgmentProfile(BaseModel):
    score: int = Field(ge=1, le=10)
    verdict: Literal["invest", "pass"]
    confidence: int = Field(ge=0, le=100)
    rubric_scores: RubricScores
    reasoning: str
