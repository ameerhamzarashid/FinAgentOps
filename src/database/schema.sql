CREATE TABLE IF NOT EXISTS stocks (
    ticker VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticker_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20),
    price_date DATE,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    adjusted_close NUMERIC,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, price_date)
);

CREATE TABLE IF NOT EXISTS technical_features (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20),
    price_date DATE,
    daily_return NUMERIC,
    log_return NUMERIC,
    moving_avg_7 NUMERIC,
    moving_avg_14 NUMERIC,
    moving_avg_30 NUMERIC,
    volatility_7 NUMERIC,
    volatility_30 NUMERIC,
    rsi_14 NUMERIC,
    macd NUMERIC,
    signal_line NUMERIC,
    bollinger_upper NUMERIC,
    bollinger_lower NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, price_date)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20),
    prediction_date DATE,
    model_name VARCHAR(100),
    predicted_direction VARCHAR(20),
    confidence NUMERIC,
    predicted_return NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_logs (
    id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100),
    status VARCHAR(50),
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_risk (
    id SERIAL PRIMARY KEY,
    portfolio_name VARCHAR(100),
    calculation_date DATE,
    tickers TEXT,
    weights TEXT,
    annualized_return NUMERIC,
    annualized_volatility NUMERIC,
    sharpe_ratio NUMERIC,
    max_drawdown NUMERIC,
    value_at_risk_95 NUMERIC,
    value_at_risk_99 NUMERIC,
    risk_score NUMERIC,
    risk_level VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);