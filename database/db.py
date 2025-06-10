import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    url = URL.create(
        drivername="postgresql+psycopg2",
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
    )

    return create_engine(url)

def fetch_sanctions():
    engine = get_engine()
    query_text = text("SELECT * FROM ofac_consolidated")

    with engine.connect() as connection:
        result = connection.execute(query_text)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def fetch_sanctions_by_country(country: str) -> pd.DataFrame:
    engine = get_engine()
    query_text = text(f"SELECT * FROM ofac_consolidated WHERE country ILIKE :country_param")

    params = {"country_param": f"%{country}%"}
    
    with engine.connect() as connection:
        result = connection.execute(query_text, params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def insert_prediction_log(
    input_text: str, 
    name: str, 
    prob: float, 
    is_match: bool, 
    threshold: float, 
    source_route: str
):
    engine = get_engine()
    query_text = text("""
        INSERT INTO prediction_log (input_name, name, probability, is_match, threshold, source_route)
        VALUES (:input_name, :name, :probability, :is_match, :threshold, :source_route);         
    """)

    params = {
        "input_name": input_text,
        "name": name,
        "probability": float(prob),
        "is_match": bool(is_match),
        "threshold": float(threshold),
        "source_route": source_route
    }

    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(query_text, params)
        return None
    except Exception as e:
        print(f"Error inserting prediction log: {e}") 
        raise