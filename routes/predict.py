from fastapi import APIRouter, HTTPException
from schemas.match import (
    MatchRequest, MatchResponse,
    BulkMatchRequest, BulkMatchResponse, MatchCandidate
)
# from database.db import get_db
from utils.preprocessing import standardize_name
from utils.features import compute_features
from sqlalchemy import text

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

@router.post("/predict_match", response_model=MatchResponse)
def predict_match(request: MatchRequest):
    try:
        # Step 1: Preprocess
        name1 = standardize_name(request.name1)
        name2 = standardize_name(request.name2)

        # Step 2: Feature Engineering
        features = compute_features(name1, name2)
        df_features = pd.DataFrame([features])

        # Step 3: Predict
        prob = model.predict_proba(df_features)[:, 1][0]
        threshold = 0.5
        is_match = prob >= threshold

        # Step 4: Log
        logging.info(f"{request.name1}, {request.name2}, {prob:.4f},{is_match}")

        return MatchResponse(
            match_probability=round(prob, 4),
            is_match=is_match,
            threshold=threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))