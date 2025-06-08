from pydantic import BaseModel
from typing import List, Optional

class MatchRequest(BaseModel):
    name1: str
    name2: str

# Output schema
class MatchResponse(BaseModel):
    match_probability: float
    is_match: bool
    threshold: float = 0.5

class BulkMatchRequest(BaseModel):
    input_name: str
    top_n: int = 5

class MatchCandidate(BaseModel):
    ofac_name: str
    match_probability: float
    is_match: bool
    threshold: float

class MatchSanction(BaseModel):
    ent_num: int
    sdn_name: str
    sdn_type: str
    country: str
    cleaned_name: str
    fuzz_ratio: Optional[float] = 0

class BulkMatchResponse(BaseModel):
    input_name: str
    candidates: List[MatchCandidate]