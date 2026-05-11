import numpy as np
import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine


def load_price_data():
    engine = get_engine()

    query = """
    SELECT
        ticker,
        price_date,
        open_price,
        high_price,
        low_price,
        close_price,
        adjusted_close,
        volume
    FROM ticker_prices
    ORDER BY ticker, price_date;
    """

    data = pd.read_sql(query, engine)

    data["price_date"] = pd.to_datetime(data["price_date"])

    numeric_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close",
        "volume",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def calculate_rsi(series, window=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    average_gain = gain.rolling(window=window, min_periods=window).mean()
    average_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def build_features_for_ticker(group):
    group = group.copy()
    group = group.sort_values("price_date")

    group["daily_return"] = group["adjusted_close"].pct_change()
    group["log_return"] = np.log(group["adjusted_close"] / group["adjusted_close"].shift(1))

    group["moving_avg_7"] = group["adjusted_close"].rolling(window=7).mean()
    group["moving_avg_14"] = group["adjusted_close"].rolling(window=14).mean()
    group["moving_avg_30"] = group["adjusted_close"].rolling(window=30).mean()

    group["volatility_7"] = group["daily_return"].rolling(window=7).std()
    group["volatility_30"] = group["daily_return"].rolling(window=30).std()

    group["rsi_14"] = calculate_rsi(group["adjusted_close"], window=14)

    ema_12 = group["adjusted_close"].ewm(span=12, adjust=False).mean()
    ema_26 = group["adjusted_close"].ewm(span=26, adjust=False).mean()

    group["macd"] = ema_12 - ema_26
    group["signal_line"] = group["macd"].ewm(span=9, adjust=False).mean()

    rolling_mean_20 = group["adjusted_close"].rolling(window=20).mean()
    rolling_std_20 = group["adjusted_close"].rolling(window=20).std()

    group["bollinger_upper"] = rolling_mean_20 + (2 * rolling_std_20)
    group["bollinger_lower"] = rolling_mean_20 - (2 * rolling_std_20)

    group["momentum_7"] = group["adjusted_close"] - group["adjusted_close"].shift(7)
    group["momentum_14"] = group["adjusted_close"] - group["adjusted_close"].shift(14)

    return group


def build_all_features(price_data):
    feature_data = (
        price_data
        .groupby("ticker", group_keys=False)
        .apply(build_features_for_ticker)
        .reset_index(drop=True)
    )

    selected_columns = [
        "ticker",
        "price_date",
        "daily_return",
        "log_return",
        "moving_avg_7",
        "moving_avg_14",
        "moving_avg_30",
        "volatility_7",
        "volatility_30",
        "rsi_14",
        "macd",
        "signal_line",
        "bollinger_upper",
        "bollinger_lower",
        "momentum_7",
        "momentum_14",
    ]

    feature_data = feature_data[selected_columns]

    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)

    return feature_data


def insert_features_to_database(feature_data):
    engine = get_engine()

    insert_query = text(
        """
        INSERT INTO technical_features (
            ticker,
            price_date,
            daily_return,
            log_return,
            moving_avg_7,
            moving_avg_14,
            moving_avg_30,
            volatility_7,
            volatility_30,
            rsi_14,
            macd,
            signal_line,
            bollinger_upper,
            bollinger_lower
        )
        VALUES (
            :ticker,
            :price_date,
            :daily_return,
            :log_return,
            :moving_avg_7,
            :moving_avg_14,
            :moving_avg_30,
            :volatility_7,
            :volatility_30,
            :rsi_14,
            :macd,
            :signal_line,
            :bollinger_upper,
            :bollinger_lower
        )
        ON CONFLICT (ticker, price_date)
        DO UPDATE SET
            daily_return = EXCLUDED.daily_return,
            log_return = EXCLUDED.log_return,
            moving_avg_7 = EXCLUDED.moving_avg_7,
            moving_avg_14 = EXCLUDED.moving_avg_14,
            moving_avg_30 = EXCLUDED.moving_avg_30,
            volatility_7 = EXCLUDED.volatility_7,
            volatility_30 = EXCLUDED.volatility_30,
            rsi_14 = EXCLUDED.rsi_14,
            macd = EXCLUDED.macd,
            signal_line = EXCLUDED.signal_line,
            bollinger_upper = EXCLUDED.bollinger_upper,
            bollinger_lower = EXCLUDED.bollinger_lower;
        """
    )

    database_columns = [
        "ticker",
        "price_date",
        "daily_return",
        "log_return",
        "moving_avg_7",
        "moving_avg_14",
        "moving_avg_30",
        "volatility_7",
        "volatility_30",
        "rsi_14",
        "macd",
        "signal_line",
        "bollinger_upper",
        "bollinger_lower",
    ]

    records = feature_data[database_columns].to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(insert_query, records)

    print("Technical features inserted into PostgreSQL successfully.")


def save_features_csv(feature_data):
    output_path = "data/processed/technical_features.csv"
    feature_data.to_csv(output_path, index=False)
    print(f"Feature data saved to {output_path}")


def main():
    print("Starting Stage 2 feature engineering pipeline...")

    price_data = load_price_data()

    print(f"Loaded {len(price_data)} price rows from database.")

    feature_data = build_all_features(price_data)

    print(f"Created {len(feature_data)} feature rows.")

    save_features_csv(feature_data)

    insert_features_to_database(feature_data)

    print("Stage 2 feature engineering completed successfully.")


if __name__ == "__main__":
    main()