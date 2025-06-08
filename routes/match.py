import logging
import joblib
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from schemas.match import MatchSanction, MatchResponse, BulkMatchRequest
from database.db import fetch_sanctions
from utils.preprocessing import standardize_name
from utils.utils import get_fuzz_ratio
from utils.features import compute_features
from database.tables import insert_prediction_log
from typing import List

model = joblib.load("models/xgb_sanction_model.pkl")

router = APIRouter()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

THRESHOLD = 0.5

@router.get("/", response_model=List[MatchSanction])
def get_matches(name: str):
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
    
@router.post("/bulk", response_model=List[MatchResponse])
def bulk_match(request: BulkMatchRequest):
    try:
        input_name = standardize_name(request.input_name)
        df = fetch_sanctions()

        df["features"] = df["cleaned_name"].apply(lambda name2: compute_features(input_name, name2))

        features_df = pd.DataFrame(df["features"].tolist())
        df["probability"] = model.predict_proba(features_df)[:, 1]
        df["is_match"] = df["probability"] >= THRESHOLD

        matched = df[df["is_match"] == True]

        for _, row in matched.iterrows():
            insert_prediction_log(
                input_name=input_name,
                name=row["cleaned_name"],
                prob=row["probability"],
                is_match=row["is_match"]
            )
        
        return [
            MatchResponse(
                match_probability=round(row["probability"], 4),
                is_match=row["is_match"],
                threshold=THRESHOLD
            )
            for _, row in matched.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))