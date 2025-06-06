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

