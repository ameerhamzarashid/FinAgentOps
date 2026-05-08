from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy import text

from src.database.connection import get_engine


TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "V",
    "NFLX",
]

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_PATH = RAW_DATA_DIR / "stock_prices.csv"


def download_stock_data(tickers, start_date, end_date):
    all_data = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )

        if data.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue

        data = data.reset_index()

        print(f"{ticker} columns:", data.columns.tolist())

        data["ticker"] = ticker

        data = data.rename(
            columns={
                "Date": "price_date",
                "Open": "open_price",
                "High": "high_price",
                "Low": "low_price",
                "Close": "close_price",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )

        required_columns = [
            "ticker",
            "price_date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "adjusted_close",
            "volume",
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise KeyError(
                f"Missing columns for {ticker}: {missing_columns}. "
                f"Available columns: {data.columns.tolist()}"
            )

        data = data[required_columns]

        all_data.append(data)

    if not all_data:
        raise ValueError("No stock data was downloaded.")

    final_data = pd.concat(all_data, ignore_index=True)

    final_data["price_date"] = pd.to_datetime(final_data["price_date"]).dt.date

    final_data = final_data.drop_duplicates(subset=["ticker", "price_date"])

    final_data = final_data.sort_values(["ticker", "price_date"])

    return final_data


def save_raw_csv(dataframe):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw stock data saved to {RAW_DATA_PATH}")


def insert_stock_metadata(dataframe):
    engine = get_engine()

    stock_rows = dataframe[["ticker"]].drop_duplicates().copy()
    stock_rows["company_name"] = None
    stock_rows["sector"] = None
    stock_rows["industry"] = None

    with engine.begin() as connection:
        for _, row in stock_rows.iterrows():
            connection.execute(
                text(
                    """
                    INSERT INTO stocks (ticker, company_name, sector, industry)
                    VALUES (:ticker, :company_name, :sector, :industry)
                    ON CONFLICT (ticker) DO NOTHING;
                    """
                ),
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "industry": row["industry"],
                },
            )

    print("Stock metadata inserted successfully.")


def insert_prices_to_database(dataframe):
    engine = get_engine()

    records = dataframe.to_dict(orient="records")

    insert_query = text(
        """
        INSERT INTO ticker_prices (
            ticker,
            price_date,
            open_price,
            high_price,
            low_price,
            close_price,
            adjusted_close,
            volume
        )
        VALUES (
            :ticker,
            :price_date,
            :open_price,
            :high_price,
            :low_price,
            :close_price,
            :adjusted_close,
            :volume
        )
        ON CONFLICT (ticker, price_date)
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            adjusted_close = EXCLUDED.adjusted_close,
            volume = EXCLUDED.volume;
        """
    )

    with engine.begin() as connection:
        connection.execute(insert_query, records)

    print("Stock price data inserted into PostgreSQL successfully.")


def main():
    print("Starting Stage 1 stock data ingestion pipeline...")

    stock_data = download_stock_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print(f"Downloaded {len(stock_data)} rows.")

    save_raw_csv(stock_data)
    insert_stock_metadata(stock_data)
    insert_prices_to_database(stock_data)

    print("Stage 1 pipeline completed successfully.")


if __name__ == "__main__":
    main()