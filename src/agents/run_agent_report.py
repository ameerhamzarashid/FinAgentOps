import argparse

from src.agents.financial_report_agent import run_agent_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate an agentic AI stock intelligence report."
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker to analyse, for example AAPL, MSFT, NVDA.",
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()

    print(f"Generating agentic AI report for {ticker}...")

    result = run_agent_report(ticker)

    print("\nReport preview:")
    print("----------------")
    print(result["final_report"][:1500])


if __name__ == "__main__":
    main()