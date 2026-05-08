from pathlib import Path
from sqlalchemy import text

from src.database.connection import get_engine


def initialise_database():
    schema_path = Path("src/database/schema.sql")

    if not schema_path.exists():
        raise FileNotFoundError("schema.sql not found inside src/database/")

    engine = get_engine()

    with engine.begin() as connection:
        schema_sql = schema_path.read_text()
        connection.execute(text(schema_sql))

    print("Database schema created successfully.")


if __name__ == "__main__":
    initialise_database()