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