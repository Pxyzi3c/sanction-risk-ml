from fastapi import APIRouter, HTTPException
from schemas.match import (
    MatchRequest, MatchResponse,
    BulkMatchRequest, BulkMatchResponse, MatchCandidate
)
from database.db import fetch_sanctions
from utils.preprocessing import standardize_name
from utils.features import compute_features

import pandas as pd
import joblib
import logging
import os

model = joblib.load("models/xgb_sanction_model.pkl")

router = APIRouter()

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

THRESHOLD = 0.5

@router.post("", response_model=MatchResponse)
def predict_match(request: MatchRequest):
    try:
        name1 = standardize_name(request.name1)
        name2 = standardize_name(request.name2)

        features = compute_features(name1, name2)
        df_features = pd.DataFrame([features])

        prob = model.predict_proba(df_features)[:, 1][0]
        is_match = prob >= THRESHOLD

        logging.info(f"{request.name1}, {request.name2}, {prob:.4f},{is_match}")

        return MatchResponse(
            match_probability=round(prob, 4),
            is_match=is_match,
            threshold=THRESHOLD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk", response_model=BulkMatchResponse)
def bulk_match(request: BulkMatchRequest):
    try:
        df = fetch_sanctions()
        input_name = standardize_name(request.input_name)

        feature_rows = df["cleaned_name"].apply(lambda x: compute_features(input_name, x))
        df_features = pd.DataFrame(list(feature_rows))

        probs = model.predict_proba(df_features)[:, 1]
        df["match_probability"] = probs
        df["is_match"] = df["match_probability"] >= THRESHOLD

        top_matches = df.sort_values("match_probability", ascending=False).head(request.top_n)
        
        response = BulkMatchResponse(
            input_name=request.input_name,
            candidates=[
                MatchCandidate(
                    ofac_name=row["cleaned_name"],
                    match_probability=round(row["match_probability"], 4),
                    is_match=row["is_match"],
                    threshold=THRESHOLD
                )
                for _, row in top_matches.iterrows()
            ]
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))