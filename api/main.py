from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.features import compute_features
from utils.preprocessing import standardize_name
import joblib
import numpy as np
import pandas as pd
import datetime
import logging

app = FastAPI(
    title="Sanction Match Predictor API",
    version="1.0.0"
)

model = joblib.load("models/xgb_sanction_model.pkl")

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# Input schema
class MatchRequest(BaseModel):
    name1: str
    name2: str

# Output schema
class MatchResponse(BaseModel):
    match_probability: float
    is_match: bool
    threshol: float = 0.8

@app.post("/predict_match", response_model=MatchResponse)
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