import pandas as pd

from src.database.connection import get_engine


def check_technical_features():
    engine = get_engine()

    summary_query = """
    SELECT
        ticker,
        COUNT(*) AS total_rows,
        MIN(price_date) AS first_date,
        MAX(price_date) AS latest_date,
        ROUND(AVG(daily_return), 6) AS avg_daily_return,
        ROUND(AVG(volatility_30), 6) AS avg_volatility_30,
        ROUND(AVG(rsi_14), 2) AS avg_rsi_14
    FROM technical_features
    GROUP BY ticker
    ORDER BY ticker;
    """

    summary = pd.read_sql(summary_query, engine)

    print("\nTechnical feature summary:\n")
    print(summary)

    latest_query = """
    SELECT
        ticker,
        price_date,
        daily_return,
        moving_avg_7,
        moving_avg_30,
        volatility_30,
        rsi_14,
        macd,
        signal_line,
        bollinger_upper,
        bollinger_lower
    FROM technical_features
    ORDER BY price_date DESC
    LIMIT 20;
    """

    latest = pd.read_sql(latest_query, engine)

    print("\nLatest feature records:\n")
    print(latest)


if __name__ == "__main__":
    check_technical_features()