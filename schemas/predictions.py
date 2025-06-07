from pydantic import BaseModel

class MatchRequest(BaseModel):
    name1: str
    name2: str

# Output schema
class MatchResponse(BaseModel):
    match_probability: float
    is_match: bool
    threshold: float = 0.5