import os
import pandas as pd
from sqlalchemy import create_engine
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
    return pd.read_sql("SELECT * FROM ofac_consolidated", con=engine)