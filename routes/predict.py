from fastapi import APIRouter, HTTPException
from schemas.match import (
    MatchRequest, MatchResponse,
    BulkMatchRequest
)
from database.db import fetch_sanctions
from utils.preprocessing import prepare_bulk_predictions
from utils.preprocessing import standardize_name
from utils.features import compute_features
from database.db import insert_prediction_log
from typing import List
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

@router.post("/bulk", response_model=List[MatchResponse])
def bulk_match(request: BulkMatchRequest):
    try:
        input_name = standardize_name(request.input_name)
        df = prepare_bulk_predictions(fetch_sanctions(), input_name, THRESHOLD)

        matched = df[df["is_match"] == True]

        for _, row in matched.iterrows():
            insert_prediction_log(
                input_text=input_name,
                name=row["cleaned_name"],
                prob=row["probability"],
                is_match=row["is_match"],
                threshold=THRESHOLD,
                source_route="predict_match/bulk"
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

@router.post("/top", response_model=MatchResponse)
def bulk_match_top_only(request: BulkMatchRequest):
    try:
        input_name = standardize_name(request.input_name)
        df = prepare_bulk_predictions(fetch_sanctions(), input_name, THRESHOLD)
        
        top_match = df.sort_values("probability", ascending=False).iloc[0]

        insert_prediction_log(
            input_text=input_name,
            name=top_match["cleaned_name"],
            prob=top_match["probability"],
            is_match=top_match["is_match"],
            threshold=THRESHOLD,
            source_route="predict_match/top"
        )

        return MatchResponse(
            match_probability=round(top_match["probability"], 4),
            is_match=top_match["is_match"],
            threshold=THRESHOLD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))