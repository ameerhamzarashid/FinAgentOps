import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text

from src.database.connection import get_engine
from src.config.settings import DEFAULT_TICKERS


load_dotenv()

SEC_BASE_URL = "https://data.sec.gov"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

RAW_DIR = Path("data/external/sec")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Ameer Hamza ameerhamzauk9@gmail.com",
)

HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


TAXONOMY = "us-gaap"

FUNDAMENTAL_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
    ],
    "assets": [
        "Assets",
    ],
    "liabilities": [
        "Liabilities",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
    ],
}


def normalise_cik(cik: int | str) -> str:
    return str(cik).zfill(10)


def sec_get_json(url: str, host: Optional[str] = None):
    headers = HEADERS.copy()

    if host:
        headers["Host"] = host

    response = requests.get(url, headers=headers, timeout=30)

    response.raise_for_status()

    time.sleep(0.2)

    return response.json()


def fetch_ticker_mapping():
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }

    response = requests.get(
        SEC_TICKER_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    mapping_json = response.json()

    rows = []

    for _, item in mapping_json.items():
        rows.append(
            {
                "ticker": item["ticker"].upper(),
                "cik": normalise_cik(item["cik_str"]),
                "company_name": item["title"],
            }
        )

    mapping = pd.DataFrame(rows)

    mapping_path = RAW_DIR / "company_tickers.csv"
    mapping.to_csv(mapping_path, index=False)

    print(f"SEC ticker mapping saved to {mapping_path}")

    return mapping


def insert_company_metadata(mapping, tickers):
    engine = get_engine()

    selected = mapping[mapping["ticker"].isin(tickers)].copy()

    query = text(
        """
        INSERT INTO sec_company_metadata (
            ticker,
            cik,
            company_name
        )
        VALUES (
            :ticker,
            :cik,
            :company_name
        )
        ON CONFLICT (ticker)
        DO UPDATE SET
            cik = EXCLUDED.cik,
            company_name = EXCLUDED.company_name;
        """
    )

    with engine.begin() as connection:
        connection.execute(query, selected.to_dict(orient="records"))

    print("SEC company metadata inserted successfully.")


def fetch_company_facts(cik: str):
    url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"

    data = sec_get_json(url)

    output_path = RAW_DIR / f"companyfacts_{cik}.json"

    pd.Series(data).to_json(output_path)

    return data


def get_fact_units(company_facts, tag):
    facts = company_facts.get("facts", {})
    taxonomy_facts = facts.get(TAXONOMY, {})
    tag_data = taxonomy_facts.get(tag, {})
    units = tag_data.get("units", {})

    return units


def select_best_usd_fact(company_facts, possible_tags):
    for tag in possible_tags:
        units = get_fact_units(company_facts, tag)

        if "USD" in units:
            return units["USD"], tag

        if "USD/shares" in units:
            return units["USD/shares"], tag

        if "shares" in units:
            return units["shares"], tag

    return [], None


def extract_latest_facts(ticker, cik, company_name, company_facts):
    base_records = {}

    for field_name, tags in FUNDAMENTAL_TAGS.items():
        facts, selected_tag = select_best_usd_fact(company_facts, tags)

        for fact in facts:
            fiscal_year = fact.get("fy")
            fiscal_period = fact.get("fp")
            form_type = fact.get("form")
            filed_date = fact.get("filed")
            value = fact.get("val")

            if fiscal_year is None or fiscal_period is None or form_type is None:
                continue

            if form_type not in ["10-K", "10-Q"]:
                continue

            key = (
                ticker,
                int(fiscal_year),
                str(fiscal_period),
                str(form_type),
            )

            if key not in base_records:
                base_records[key] = {
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": company_name,
                    "fiscal_year": int(fiscal_year),
                    "fiscal_period": str(fiscal_period),
                    "form_type": str(form_type),
                    "filed_date": filed_date,
                    "revenue": None,
                    "net_income": None,
                    "assets": None,
                    "liabilities": None,
                    "stockholders_equity": None,
                    "operating_cash_flow": None,
                    "eps_basic": None,
                    "eps_diluted": None,
                }

            current_filed = base_records[key].get("filed_date")

            if current_filed is None or str(filed_date) >= str(current_filed):
                base_records[key]["filed_date"] = filed_date
                base_records[key][field_name] = value

    records = list(base_records.values())

    if not records:
        return pd.DataFrame()

    data = pd.DataFrame(records)

    data["filed_date"] = pd.to_datetime(data["filed_date"], errors="coerce").dt.date

    data = data.sort_values(
        ["ticker", "fiscal_year", "fiscal_period", "form_type"],
        ascending=[True, False, False, True],
    )

    return data


def insert_fundamentals_to_database(fundamentals):
    if fundamentals.empty:
        print("No fundamentals to insert.")
        return

    engine = get_engine()

    query = text(
        """
        INSERT INTO company_fundamentals (
            ticker,
            cik,
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
        )
        VALUES (
            :ticker,
            :cik,
            :company_name,
            :fiscal_year,
            :fiscal_period,
            :form_type,
            :filed_date,
            :revenue,
            :net_income,
            :assets,
            :liabilities,
            :stockholders_equity,
            :operating_cash_flow,
            :eps_basic,
            :eps_diluted
        )
        ON CONFLICT (ticker, fiscal_year, fiscal_period, form_type)
        DO UPDATE SET
            cik = EXCLUDED.cik,
            company_name = EXCLUDED.company_name,
            filed_date = EXCLUDED.filed_date,
            revenue = EXCLUDED.revenue,
            net_income = EXCLUDED.net_income,
            assets = EXCLUDED.assets,
            liabilities = EXCLUDED.liabilities,
            stockholders_equity = EXCLUDED.stockholders_equity,
            operating_cash_flow = EXCLUDED.operating_cash_flow,
            eps_basic = EXCLUDED.eps_basic,
            eps_diluted = EXCLUDED.eps_diluted;
        """
    )

    records = fundamentals.where(pd.notnull(fundamentals), None).to_dict(
        orient="records"
    )

    with engine.begin() as connection:
        connection.execute(query, records)

    print(f"Inserted {len(records)} fundamentals records into PostgreSQL.")


def fetch_and_process_fundamentals(tickers):
    mapping = fetch_ticker_mapping()

    tickers = [ticker.upper() for ticker in tickers]

    insert_company_metadata(mapping, tickers)

    all_fundamentals = []

    for ticker in tickers:
        selected = mapping[mapping["ticker"] == ticker]

        if selected.empty:
            print(f"No SEC CIK found for {ticker}. Skipping.")
            continue

        row = selected.iloc[0]
        cik = row["cik"]
        company_name = row["company_name"]

        print(f"Fetching SEC company facts for {ticker} / CIK {cik}...")

        try:
            company_facts = fetch_company_facts(cik)

            fundamentals = extract_latest_facts(
                ticker=ticker,
                cik=cik,
                company_name=company_name,
                company_facts=company_facts,
            )

            if fundamentals.empty:
                print(f"No usable fundamentals found for {ticker}.")
                continue

            fundamentals_path = RAW_DIR / f"{ticker}_fundamentals.csv"
            fundamentals.to_csv(fundamentals_path, index=False)

            print(f"Saved {ticker} fundamentals to {fundamentals_path}")

            all_fundamentals.append(fundamentals)

        except Exception as error:
            print(f"Failed to fetch fundamentals for {ticker}: {error}")

    if not all_fundamentals:
        raise ValueError("No fundamentals were collected.")

    final_fundamentals = pd.concat(all_fundamentals, ignore_index=True)

    output_path = RAW_DIR / "company_fundamentals.csv"
    final_fundamentals.to_csv(output_path, index=False)

    print(f"Combined fundamentals saved to {output_path}")

    insert_fundamentals_to_database(final_fundamentals)

    return final_fundamentals


def main():
    print("Starting Stage 9 SEC fundamentals ingestion...")

    fundamentals = fetch_and_process_fundamentals(DEFAULT_TICKERS)

    print(f"Collected {len(fundamentals)} SEC fundamentals rows.")
    print("Stage 9 SEC fundamentals ingestion completed successfully.")


if __name__ == "__main__":
    main()