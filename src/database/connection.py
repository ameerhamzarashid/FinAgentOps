import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "finagentops")
    user = os.getenv("POSTGRES_USER", "finagentops_user")
    password = os.getenv("POSTGRES_PASSWORD", "finagentops_password")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def get_engine():
    return create_engine(get_database_url())