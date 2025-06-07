from pydantic import BaseModel

class Sanction(BaseModel):
    ent_num: int
    sdn_name: str
    sdn_type: str
    country: str
    cleaned_name: str
    fuzz_ratio: float