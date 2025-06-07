import logging

from fastapi import APIRouter, HTTPException
from schemas.match import MatchSanction
from database.db import fetch_sanctions
from utils.preprocessing import standardize_name
from utils.utils import get_fuzz_ratio
from typing import List

import joblib
import logging

model = joblib.load("models/xgb_sanction_model.pkl")

router = APIRouter()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

@router.get("/matches", response_model=List[MatchSanction])
async def get_matches(name: str):
    try:
        name_std = standardize_name(name)
        df = fetch_sanctions()

        threshold = 50

        df["fuzz_ratio"] = df["cleaned_name"].apply(lambda x: get_fuzz_ratio(x, name_std, "ratio"))
        matches = df[df['fuzz_ratio'] > threshold].fillna('-').to_dict(orient="records")

        if len(matches) == 0:
            raise HTTPException(status_code=404, detail="No matches found")
        else:
            return matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))