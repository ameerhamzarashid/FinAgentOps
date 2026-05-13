# FinAgentOps

FinAgentOps is an end-to-end Agentic AI and MLOps platform for stock forecasting, portfolio risk analytics, and automated financial intelligence.

## Project Goal

The goal of this project is to build a production-style financial analytics system that combines:

- Python-based stock market data pipelines
- PostgreSQL data warehouse
- Machine learning models for stock movement and return forecasting
- MLflow experiment tracking
- Prefect workflow automation
- LangGraph-based agentic AI reporting
- FastAPI backend services
- Power BI dashboards

## Current Stage

Stage 0: Project Setup and Planning

## Planned Stages

1. Project setup and planning
2. Stock data ingestion
3. Feature engineering
4. Baseline machine learning model
5. MLflow experiment tracking
6. Forecasting model
7. Portfolio risk engine
8. Prefect automation
9. Agentic AI layer
10. SEC fundamentals integration
11. FastAPI backend
12. Power BI dashboard
13. Docker, CI/CD, and deployment
14. Portfolio packaging

## Tech Stack

Python, PostgreSQL, Docker, yfinance, scikit-learn, MLflow, Prefect, LangGraph, FastAPI, Power BI.

## Stage 1: Stock Data Ingestion Pipeline

Stage 1 downloads historical stock market data for selected tickers, saves the raw dataset as CSV, and loads the cleaned records into PostgreSQL.

### Current Tickers

AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, V, NFLX

### Pipeline Steps

1. Download historical stock data using yfinance
2. Clean and standardise column names
3. Save raw CSV file to `data/raw/stock_prices.csv`
4. Insert stock metadata into the `stocks` table
5. Insert daily OHLCV price records into the `ticker_prices` table
6. Validate inserted records using a database check script

### Run Database Schema

```bash
python -m src.database.init_db

## Stage 2: Feature Engineering

Stage 2 converts raw stock price data into technical indicators for machine learning.

### Features Created

- Daily return
- Log return
- 7-day moving average
- 14-day moving average
- 30-day moving average
- 7-day volatility
- 30-day volatility
- RSI 14
- MACD
- Signal line
- Bollinger upper band
- Bollinger lower band
- 7-day momentum
- 14-day momentum

### Run Feature Engineering

```bash
python -m src.features.build_features

## Stage 3: Baseline Machine Learning Model

Stage 3 builds the first supervised machine learning model for stock direction prediction.

The target variable predicts whether the next trading day's adjusted closing price will be higher than the current day's adjusted closing price.

### Models Trained

- Logistic Regression
- Random Forest Classifier

### Features Used

- Daily return
- Log return
- Moving averages
- Volatility
- RSI
- MACD
- Signal line
- Bollinger Bands

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

### Run Training

```bash
python -m src.models.train_baseline_classifier

## Stage 4: MLflow MLOps Tracking

Stage 4 adds MLflow experiment tracking to the baseline stock direction models.

### What is tracked

- Model name
- Dataset size
- Train/test split size
- Feature list
- Model parameters
- Accuracy
- Precision
- Recall
- F1 score
- Classification report
- Confusion matrix image
- Saved MLflow model artifact

### Run MLflow Training

```bash
python -m src.mlops.train_with_mlflow

## Stage 5: Next-Day Return Forecasting Model

Stage 5 builds regression models to forecast next-day stock returns.

### Target Variable

The target variable is `next_day_return`, calculated as:

```text
next_day_return = (next_day_adjusted_close - current_adjusted_close) / current_adjusted_close

## Stage 6: Portfolio Risk Engine

Stage 6 calculates portfolio-level risk and return metrics using adjusted closing prices.

### Portfolio Metrics

- Daily portfolio return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Historical Value at Risk at 95%
- Historical Value at Risk at 99%
- Correlation matrix
- Risk score
- Risk level

### Default Portfolio

The default portfolio uses 10 stocks:

AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, V, NFLX

### Run Portfolio Risk Engine

```bash
python -m src.models.portfolio_risk_engine

## Stage 7: Prefect Automation

Stage 7 automates the FinAgentOps workflow using Prefect.

### Automated Pipeline Steps

1. Stock data ingestion
2. Technical feature engineering
3. Direction classifier training
4. Next-day return model training
5. Latest stock predictions
6. Portfolio risk calculation
7. Portfolio chart generation
8. Pipeline status logging

### Run Full Daily Pipeline

```bash
python -m src.workflows.daily_pipeline