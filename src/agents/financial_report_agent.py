from pathlib import Path
from typing import TypedDict, Optional

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI

from src.database.connection import get_engine


load_dotenv()

REPORT_DIR = Path("reports/daily")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class AgentState(TypedDict):
    ticker: str
    data_summary: Optional[str]
    model_summary: Optional[str]
    portfolio_summary: Optional[str]
    analyst_summary: Optional[str]
    final_report: Optional[str]


def get_openai_client():
    return OpenAI()


def call_llm(prompt: str) -> str:
    return f"""
# FinAgentOps Daily Stock Intelligence Report

## Ticker Reviewed

This report was generated using the local rule-based fallback report generator.

## Educational Analytics Summary

The FinAgentOps system successfully collected market data, technical indicators, machine learning model outputs, and portfolio risk metrics.

## Source Context

{prompt[:3000]}

## Interpretation

The report combines recent technical indicators, model performance summaries, and portfolio risk metrics. This fallback version is designed to keep the project running locally without requiring an external LLM API key.

## Limitations

This is not a full LLM-generated report. It is a local fallback report for development and portfolio demonstration.

## Disclaimer

This report is for educational analytics only. It is not financial advice.
"""


def data_agent(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    engine = get_engine()

    query = """
    SELECT
        p.ticker,
        p.price_date,
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
    LIMIT 30;
    """

    data = pd.read_sql(query, engine, params={"ticker": ticker})

    if data.empty:
        state["data_summary"] = f"No market data found for {ticker}."
        return state

    latest = data.iloc[0]

    data_summary = f"""
Ticker: {ticker}
Latest date: {latest["price_date"]}
Latest adjusted close: {latest["adjusted_close"]}
Latest volume: {latest["volume"]}
Latest daily return: {latest["daily_return"]}
7-day moving average: {latest["moving_avg_7"]}
30-day moving average: {latest["moving_avg_30"]}
30-day volatility: {latest["volatility_30"]}
RSI 14: {latest["rsi_14"]}
MACD: {latest["macd"]}
Signal line: {latest["signal_line"]}
Bollinger upper: {latest["bollinger_upper"]}
Bollinger lower: {latest["bollinger_lower"]}
30-row average daily return: {data["daily_return"].mean()}
30-row average volatility: {data["volatility_30"].mean()}
"""

    state["data_summary"] = data_summary
    return state


def ml_agent(state: AgentState) -> AgentState:
    ticker = state["ticker"]

    direction_results_path = Path("reports/model/baseline_model_results.csv")
    return_results_path = Path("reports/model/forecasting/return_forecasting_results.csv")

    direction_summary = "Direction model results file not found."
    return_summary = "Return forecasting results file not found."

    if direction_results_path.exists():
        direction_results = pd.read_csv(direction_results_path)
        best_direction_model = direction_results.sort_values(
            "f1_score",
            ascending=False,
        ).iloc[0]

        direction_summary = f"""
Best direction model: {best_direction_model["model_name"]}
Accuracy: {best_direction_model["accuracy"]}
Precision: {best_direction_model["precision"]}
Recall: {best_direction_model["recall"]}
F1 score: {best_direction_model["f1_score"]}
"""

    if return_results_path.exists():
        return_results = pd.read_csv(return_results_path)
        best_return_model = return_results.sort_values(
            "rmse",
            ascending=True,
        ).iloc[0]

        return_summary = f"""
Best return forecasting model: {best_return_model["model_name"]}
MAE: {best_return_model["mae"]}
MSE: {best_return_model["mse"]}
RMSE: {best_return_model["rmse"]}
R2 score: {best_return_model["r2_score"]}
Directional accuracy: {best_return_model["directional_accuracy"]}
"""

    model_summary = f"""
Ticker reviewed: {ticker}

Direction classification summary:
{direction_summary}

Next-day return forecasting summary:
{return_summary}
"""

    state["model_summary"] = model_summary
    return state


def portfolio_agent(state: AgentState) -> AgentState:
    engine = get_engine()

    query = """
    SELECT
        portfolio_name,
        calculation_date,
        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        max_drawdown,
        value_at_risk_95,
        value_at_risk_99,
        risk_score,
        risk_level
    FROM portfolio_risk
    ORDER BY created_at DESC
    LIMIT 1;
    """

    data = pd.read_sql(query, engine)

    if data.empty:
        state["portfolio_summary"] = "No portfolio risk record found."
        return state

    latest = data.iloc[0]

    portfolio_summary = f"""
Portfolio name: {latest["portfolio_name"]}
Calculation date: {latest["calculation_date"]}
Annualized return: {latest["annualized_return"]}
Annualized volatility: {latest["annualized_volatility"]}
Sharpe ratio: {latest["sharpe_ratio"]}
Maximum drawdown: {latest["max_drawdown"]}
95% historical VaR: {latest["value_at_risk_95"]}
99% historical VaR: {latest["value_at_risk_99"]}
Risk score: {latest["risk_score"]}
Risk level: {latest["risk_level"]}
"""

    state["portfolio_summary"] = portfolio_summary
    return state


def analyst_agent(state: AgentState) -> AgentState:
    prompt = f"""
You are a financial analytics assistant for an educational portfolio project.

Important:
- Do not give financial advice.
- Do not say buy, sell, or hold.
- Present this as educational analytics only.
- Be concise, professional, and evidence-based.

Create an analyst-style summary using the information below.

Market data:
{state["data_summary"]}

Model summary:
{state["model_summary"]}

Portfolio risk:
{state["portfolio_summary"]}

Write:
1. Market snapshot
2. Technical interpretation
3. Model interpretation
4. Risk interpretation
5. Key caution
"""

    analyst_summary = call_llm(prompt)
    state["analyst_summary"] = analyst_summary
    return state


def report_agent(state: AgentState) -> AgentState:
    ticker = state["ticker"]

    prompt = f"""
Create a polished Markdown report for the ticker {ticker}.

Use this structure:

# FinAgentOps Daily Stock Intelligence Report

## Ticker Reviewed
## Market Snapshot
## Technical Signal Summary
## Machine Learning Model Summary
## Portfolio Risk Context
## Educational Interpretation
## Limitations
## Disclaimer

Rules:
- Do not provide financial advice.
- Do not use words like "buy", "sell", or "hold".
- Clearly state this is for educational analytics only.
- Keep it professional and portfolio-ready.

Source analysis:
{state["analyst_summary"]}
"""

    final_report = call_llm(prompt)

    report_path = REPORT_DIR / f"{ticker}_agent_stock_report.md"
    report_path.write_text(final_report, encoding="utf-8")

    state["final_report"] = final_report

    print(f"Agent report saved to {report_path}")

    return state


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("data_agent", data_agent)
    graph.add_node("ml_agent", ml_agent)
    graph.add_node("portfolio_agent", portfolio_agent)
    graph.add_node("analyst_agent", analyst_agent)
    graph.add_node("report_agent", report_agent)

    graph.set_entry_point("data_agent")

    graph.add_edge("data_agent", "ml_agent")
    graph.add_edge("ml_agent", "portfolio_agent")
    graph.add_edge("portfolio_agent", "analyst_agent")
    graph.add_edge("analyst_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()


def run_agent_report(ticker: str = "AAPL"):
    app = build_agent_graph()

    initial_state: AgentState = {
        "ticker": ticker,
        "data_summary": None,
        "model_summary": None,
        "portfolio_summary": None,
        "analyst_summary": None,
        "final_report": None,
    }

    result = app.invoke(initial_state)

    return result


if __name__ == "__main__":
    run_agent_report("AAPL")