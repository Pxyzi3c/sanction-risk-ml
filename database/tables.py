from sqlalchemy import text
from database.db import get_engine

def insert_prediction_log(name1: str, name2: str, prob: float, is_match: bool):
    with get_engine.connect() as conn:
        stmt = text("""
           INSERT INTO prediction_log (input_name1, input_name2, prob, is_match)
            VALUES (:input_name1, :input_name2, :prob, :is_match);         
        """)
        conn.execute(stmt, {
            "input_name1": name1,
            "input_name2": name2,
            "prob": prob,
            "is_match": is_match
        })