from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from utils.features import compute_features
from utils.preprocessing import standardize_name
from utils.db import fetch_sanctions
from utils.utils import get_fuzz_ratio
from schemas.predictions import MatchRequest, MatchResponse
from schemas.sanctions import Sanction
import joblib
import numpy as np
import pandas as pd
import datetime
import logging
import os

app = FastAPI(
    title="Sanction Match Predictor API",
    version="1.0.0",
)

model = joblib.load("models/xgb_sanction_model.pkl")

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

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
    
@app.get("/matches", response_model=List[Sanction])
async def get_matches(name: str):
    try:
        name_std = standardize_name(name)
        df = fetch_sanctions()
        threshold = 80

        df["fuzz_ratio"] = df["cleaned_name"].apply(lambda x: get_fuzz_ratio(x, name_std, "ratio"))
        matches = df[df['fuzz_ratio'] > threshold].fillna('-').to_dict(orient="records")

        if len(matches) == 0:
            raise HTTPException(status_code=404, detail="No matches found")
        else:
            return df[df['fuzz_ratio'] > threshold].fillna('-').to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))