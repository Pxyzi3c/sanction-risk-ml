from sqlalchemy import text
from database.db import get_engine

def insert_prediction_log(input_text: str, name: str, prob: float, is_match: bool):
    with get_engine.connect() as conn:
        stmt = text("""
           INSERT INTO prediction_log (input_text, name, probability, is_match)
            VALUES (:input_text, :name, :probability, :is_match);         
        """)
        conn.execute(stmt, {
            "input_text": input_text,
            "name": name,
            "probability": prob,
            "is_match": is_match
        })