import pandas as pd

from src.database.connection import get_engine


def check_sec_fundamentals():
    engine = get_engine()

    metadata_query = """
    SELECT *
    FROM sec_company_metadata
    ORDER BY ticker;
    """

    metadata = pd.read_sql(metadata_query, engine)

    print("\nSEC company metadata:\n")
    print(metadata)

    fundamentals_query = """
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
    ORDER BY ticker, fiscal_year DESC, fiscal_period DESC
    LIMIT 40;
    """

    fundamentals = pd.read_sql(fundamentals_query, engine)

    print("\nLatest SEC fundamentals:\n")
    print(fundamentals)


if __name__ == "__main__":
    check_sec_fundamentals()