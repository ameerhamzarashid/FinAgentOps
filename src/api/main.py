from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.database.connection import get_engine
from src.models.train_baseline_classifier import FEATURE_COLUMNS as DIRECTION_FEATURES
from src.models.train_return_forecaster import FEATURE_COLUMNS as RETURN_FEATURES


app = FastAPI(
    title="FinAgentOps API",
    description=(
        "Agentic AI and MLOps backend for stock forecasting, "
        "portfolio risk analytics, SEC fundamentals, and automated reports."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DIRECTION_MODEL_PATH = Path("models/baseline_stock_direction_model.pkl")
RETURN_MODEL_PATH = Path("models/next_day_return_forecaster.pkl")
REPORT_DIR = Path("reports/daily")


def dataframe_to_records(data: pd.DataFrame) -> list[dict[str, Any]]:
    data = data.copy()
    data = data.where(pd.notnull(data), None)
    return data.to_dict(orient="records")


def load_latest_features(ticker: str) -> pd.DataFrame:
    engine = get_engine()

    query = """
    SELECT
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
    FROM technical_features
    WHERE ticker = %(ticker)s
    ORDER BY price_date DESC
    LIMIT 1;
    """

    return pd.read_sql(query, engine, params={"ticker": ticker.upper()})


@app.get("/")
def root():
    return {
        "message": "Welcome to FinAgentOps API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "FinAgentOps API",
    }


@app.get("/stocks")
def get_stocks():
    engine = get_engine()

    query = """
    SELECT
        s.ticker,
        COALESCE(m.company_name, s.company_name) AS company_name,
        s.sector,
        s.industry
    FROM stocks s
    LEFT JOIN sec_company_metadata m
        ON s.ticker = m.ticker
    ORDER BY s.ticker;
    """

    data = pd.read_sql(query, engine)

    return {
        "count": len(data),
        "stocks": dataframe_to_records(data),
    }


@app.get("/market/latest/{ticker}")
def get_latest_market_data(ticker: str):
    engine = get_engine()

    query = """
    SELECT
        p.ticker,
        p.price_date,
        p.open_price,
        p.high_price,
        p.low_price,
        p.close_price,
        p.adjusted_close,
        p.volume,
        f.daily_return,
        f.moving_avg_7,
        f.moving_avg_30,
        f.volatility_30,
        f.rsi_14,
        f.macd,
        f.signal_line,
        f.bollinger_upper,
        f.bollinger_lower
    FROM ticker_prices p
    INNER JOIN technical_features f
        ON p.ticker = f.ticker
        AND p.price_date = f.price_date
    WHERE p.ticker = %(ticker)s
    ORDER BY p.price_date DESC
    LIMIT 1;
    """

    data = pd.read_sql(query, engine, params={"ticker": ticker.upper()})

    if data.empty:
        raise HTTPException(status_code=404, detail=f"No market data found for {ticker}")

    return dataframe_to_records(data)[0]


@app.get("/market/history/{ticker}")
def get_market_history(ticker: str, limit: int = 100):
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
    WHERE ticker = %(ticker)s
    ORDER BY price_date DESC
    LIMIT %(limit)s;
    """

    data = pd.read_sql(
        query,
        engine,
        params={
            "ticker": ticker.upper(),
            "limit": limit,
        },
    )

    if data.empty:
        raise HTTPException(status_code=404, detail=f"No price history found for {ticker}")

    return {
        "ticker": ticker.upper(),
        "count": len(data),
        "history": dataframe_to_records(data),
    }


@app.get("/predict/direction/{ticker}")
def predict_direction(ticker: str):
    if not DIRECTION_MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Direction model not found. Run Stage 3 first.",
        )

    latest_data = load_latest_features(ticker)

    if latest_data.empty:
        raise HTTPException(status_code=404, detail=f"No features found for {ticker}")

    model = joblib.load(DIRECTION_MODEL_PATH)

    X = latest_data[DIRECTION_FEATURES]
    prediction = int(model.predict(X)[0])

    direction = "UP" if prediction == 1 else "DOWN"

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(X)[0]))

    return {
        "ticker": ticker.upper(),
        "feature_date": str(latest_data.iloc[0]["price_date"]),
        "predicted_direction": direction,
        "confidence": confidence,
        "disclaimer": "Educational analytics only. Not financial advice.",
    }


@app.get("/predict/return/{ticker}")
def predict_return(ticker: str):
    if not RETURN_MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Return forecasting model not found. Run Stage 5 first.",
        )

    latest_data = load_latest_features(ticker)

    if latest_data.empty:
        raise HTTPException(status_code=404, detail=f"No features found for {ticker}")

    model = joblib.load(RETURN_MODEL_PATH)

    X = latest_data[RETURN_FEATURES]
    predicted_return = float(model.predict(X)[0])

    return {
        "ticker": ticker.upper(),
        "feature_date": str(latest_data.iloc[0]["price_date"]),
        "predicted_next_day_return": predicted_return,
        "predicted_next_day_return_percent": predicted_return * 100,
        "predicted_direction": "UP" if predicted_return > 0 else "DOWN",
        "disclaimer": "Educational analytics only. Not financial advice.",
    }


@app.get("/portfolio/risk")
def get_latest_portfolio_risk():
    engine = get_engine()

    query = """
    SELECT
        portfolio_name,
        calculation_date,
        tickers,
        weights,
        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        max_drawdown,
        value_at_risk_95,
        value_at_risk_99,
        risk_score,
        risk_level,
        created_at
    FROM portfolio_risk
    ORDER BY created_at DESC
    LIMIT 1;
    """

    data = pd.read_sql(query, engine)

    if data.empty:
        raise HTTPException(
            status_code=404,
            detail="No portfolio risk data found. Run Stage 6 first.",
        )

    return dataframe_to_records(data)[0]


@app.get("/fundamentals/{ticker}")
def get_fundamentals(ticker: str):
    engine = get_engine()

    query = """
    SELECT
        ticker,
        company_name,
        fiscal_year,
        fiscal_period,
        form_type,
        filed_date,
        revenue,
        net_income,
        assets,
        liabilities,
        stockholders_equity,
        operating_cash_flow,
        eps_basic,
        eps_diluted
    FROM company_fundamentals
    WHERE ticker = %(ticker)s
    ORDER BY fiscal_year DESC, filed_date DESC
    LIMIT 5;
    """

    data = pd.read_sql(query, engine, params={"ticker": ticker.upper()})

    if data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No fundamentals found for {ticker}. Run Stage 9 first.",
        )

    return {
        "ticker": ticker.upper(),
        "count": len(data),
        "fundamentals": dataframe_to_records(data),
    }


@app.get("/report/{ticker}")
def get_agent_report(ticker: str):
    ticker = ticker.upper()
    report_path = REPORT_DIR / f"{ticker}_agent_stock_report.md"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No report found for {ticker}. Run Stage 8 agent report first.",
        )

    return {
        "ticker": ticker,
        "report_path": str(report_path),
        "report_markdown": report_path.read_text(encoding="utf-8"),
    }


@app.get("/models/status")
def get_model_status():
    direction_results_path = Path("reports/model/baseline_model_results.csv")
    return_results_path = Path("reports/model/forecasting/return_forecasting_results.csv")

    response = {
        "direction_model_exists": DIRECTION_MODEL_PATH.exists(),
        "return_model_exists": RETURN_MODEL_PATH.exists(),
        "direction_model_path": str(DIRECTION_MODEL_PATH),
        "return_model_path": str(RETURN_MODEL_PATH),
        "direction_results": None,
        "return_forecasting_results": None,
    }

    if direction_results_path.exists():
        direction_results = pd.read_csv(direction_results_path)
        response["direction_results"] = dataframe_to_records(direction_results)

    if return_results_path.exists():
        return_results = pd.read_csv(return_results_path)
        response["return_forecasting_results"] = dataframe_to_records(return_results)

    return response


@app.get("/pipeline/status")
def get_pipeline_status():
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

    return {
        "count": len(data),
        "logs": dataframe_to_records(data),
    }