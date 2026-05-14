from pathlib import Path

import pandas as pd

from src.database.connection import get_engine


REPORT_DIR = Path("reports/fundamentals")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_fundamentals():
    engine = get_engine()

    query = """
    WITH ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY ticker
                ORDER BY fiscal_year DESC, filed_date DESC
            ) AS row_num
        FROM company_fundamentals
        WHERE form_type IN ('10-K', '10-Q')
    )
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
    FROM ranked
    WHERE row_num = 1
    ORDER BY ticker;
    """

    return pd.read_sql(query, engine)


def build_fundamental_ratios(data):
    data = data.copy()

    numeric_columns = [
        "revenue",
        "net_income",
        "assets",
        "liabilities",
        "stockholders_equity",
        "operating_cash_flow",
        "eps_basic",
        "eps_diluted",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["net_margin"] = data["net_income"] / data["revenue"]
    data["debt_to_assets"] = data["liabilities"] / data["assets"]
    data["return_on_assets"] = data["net_income"] / data["assets"]
    data["equity_ratio"] = data["stockholders_equity"] / data["assets"]

    return data


def save_fundamental_summary(summary):
    output_path = REPORT_DIR / "latest_fundamental_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"Fundamental summary saved to {output_path}")


def main():
    print("Creating latest fundamentals summary...")

    data = load_latest_fundamentals()

    if data.empty:
        print("No fundamentals found. Run Stage 9 ingestion first.")
        return

    summary = build_fundamental_ratios(data)

    save_fundamental_summary(summary)

    print("\nLatest fundamental summary:\n")
    print(summary)


if __name__ == "__main__":
    main()