import pandas as pd

from src.database.connection import get_engine


def check_pipeline_logs():
    engine = get_engine()

    query = """
    SELECT
        pipeline_name,
        status,
        message,
        created_at
    FROM pipeline_logs
    ORDER BY created_at DESC
    LIMIT 20;
    """

    data = pd.read_sql(query, engine)

    print("\nLatest pipeline logs:\n")
    print(data)


if __name__ == "__main__":
    check_pipeline_logs()