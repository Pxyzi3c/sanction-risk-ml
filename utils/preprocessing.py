import re

import joblib
import pandas as pd
from utils.features import compute_features

model = joblib.load("models/xgb_sanction_model.pkl")

def standardize_name(name: str) -> str:
    name = re.sub(r"[/-]", " ", name).upper()
    name = re.sub(r"[^A-Z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def prepare_bulk_predictions(
    df: pd.DataFrame, 
    input_name: str, 
    threshold: float
) -> pd.DataFrame:
    df = df.copy()
    
    df["features"] = df["cleaned_name"].apply(
        lambda name: compute_features(input_name, name)
    )

    features_df = pd.DataFrame(df["features"].tolist())
    df["probability"] = model.predict_proba(features_df)[:, 1]

    df["is_match"] = df["probability"] >= threshold

    return df