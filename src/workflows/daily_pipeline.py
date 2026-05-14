from datetime import datetime
import traceback

from prefect import flow, task
from sqlalchemy import text

from src.database.connection import get_engine
from src.ingestion.fetch_market_data import main as run_market_data_ingestion
from src.features.build_features import main as run_feature_engineering
from src.models.train_baseline_classifier import main as run_direction_classifier
from src.models.train_return_forecaster import main as run_return_forecaster
from src.models.predict_latest import predict_latest_direction
from src.models.predict_next_day_return import predict_next_day_return
from src.models.portfolio_risk_engine import main as run_portfolio_risk_engine
from src.models.plot_portfolio_risk import main as run_portfolio_charts
from src.agents.financial_report_agent import run_agent_report
from src.ingestion.fetch_sec_fundamentals import main as run_sec_fundamentals
from src.features.fundamental_summary import main as run_fundamental_summary

PIPELINE_NAME = "daily_finagentops_pipeline"

DEFAULT_TICKERS = [
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

AGENT_REPORT_TICKERS = ["AAPL", "MSFT", "NVDA"]


def log_pipeline_status(status, message):
    engine = get_engine()

    query = text(
        """
        INSERT INTO pipeline_logs (
            pipeline_name,
            status,
            message
        )
        VALUES (
            :pipeline_name,
            :status,
            :message
        );
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "pipeline_name": PIPELINE_NAME,
                "status": status,
                "message": message,
            },
        )


@task(name="Log pipeline start")
def log_start():
    message = f"Pipeline started at {datetime.now()}"
    print(message)
    log_pipeline_status("STARTED", message)


@task(name="Run stock data ingestion", retries=2, retry_delay_seconds=30)
def ingest_market_data():
    run_market_data_ingestion()


@task(name="Run feature engineering", retries=2, retry_delay_seconds=30)
def build_technical_features():
    run_feature_engineering()


@task(name="Train direction classifier")
def train_direction_model():
    run_direction_classifier()


@task(name="Train next-day return forecaster")
def train_return_model():
    run_return_forecaster()


@task(name="Generate latest stock predictions")
def generate_latest_predictions():
    for ticker in DEFAULT_TICKERS:
        print(f"\nGenerating direction prediction for {ticker}")
        predict_latest_direction(ticker)

        print(f"\nGenerating next-day return prediction for {ticker}")
        predict_next_day_return(ticker)


@task(name="Run portfolio risk engine")
def calculate_portfolio_risk():
    run_portfolio_risk_engine()


@task(name="Generate portfolio risk charts")
def generate_portfolio_charts():
    run_portfolio_charts()


@task(name="Generate agentic AI reports")
def generate_agentic_reports():
    for ticker in AGENT_REPORT_TICKERS:
        print(f"\nGenerating agentic report for {ticker}")
        run_agent_report(ticker)


@task(name="Log pipeline success")
def log_success():
    message = f"Pipeline completed successfully at {datetime.now()}"
    print(message)
    log_pipeline_status("SUCCESS", message)


@task(name="Log pipeline failure")
def log_failure(error_message):
    message = f"Pipeline failed at {datetime.now()}. Error: {error_message}"
    print(message)
    log_pipeline_status("FAILED", message)

@task(name="Run SEC fundamentals ingestion")
def ingest_sec_fundamentals():
    run_sec_fundamentals()


@task(name="Build fundamentals summary")
def build_fundamentals_summary():
    run_fundamental_summary()
    
@flow(name="FinAgentOps Daily Pipeline", log_prints=True)
def daily_finagentops_pipeline(
    retrain_models: bool = True,
    run_predictions: bool = True,
    run_agent_reports: bool = True,
):
    try:
        log_start()

        ingest_market_data()
        build_technical_features()
        ingest_sec_fundamentals()
        build_fundamentals_summary()

        if retrain_models:
            train_direction_model()
            train_return_model()

        if run_predictions:
            generate_latest_predictions()

        calculate_portfolio_risk()
        generate_portfolio_charts()

        if run_agent_reports:
            generate_agentic_reports()

        log_success()

    except Exception:
        error_details = traceback.format_exc()
        log_failure(error_details)
        raise


if __name__ == "__main__":
    daily_finagentops_pipeline()