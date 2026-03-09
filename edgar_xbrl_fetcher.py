"""
SEC EDGAR XBRL Company Facts fetcher — 10-15+ years of regulatory-filed data.

Uses the free XBRL API at data.sec.gov (no API key required).
Imports EDGAR_SESSION, _rate_limited_get, get_cik from sec_fetcher.py.
"""

import json
import logging
import pathlib
import time
from datetime import datetime

import numpy as np
import pandas as pd

from sec_fetcher import EDGAR_SESSION, _rate_limited_get, get_cik

logger = logging.getLogger(__name__)

# ── Cache directory ──────────────────────────────────────────────────────────

_CACHE_DIR = pathlib.Path(__file__).parent / "edgar_cache"
_CACHE_TTL_SECS = 7 * 86400  # 7 days


def _ensure_cache_dir():
    """Create edgar_cache/ under stock_screener/ if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Concept maps ─────────────────────────────────────────────────────────────

# Income statement concepts (in priority order for each metric)
REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "TotalRevenuesAndOtherIncome",
    "InterestAndDividendIncomeOperating",
    "RegulatedAndUnregulatedOperatingRevenue",
]

INCOME_STMT_CONCEPTS = {
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestPaid",
    ],
    "income_tax": [
        "IncomeTaxExpenseBenefit",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "ebitda": [
        "EarningsBeforeInterestTaxesDepreciationAndAmortization",
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
}

BALANCE_SHEET_CONCEPTS = {
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "total_debt": [
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "CommercialPaper",
        "LongTermDebtCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
}

CASH_FLOW_CONCEPTS = {
    "operating_cash_flow": [
        "NetCashProvidedByOperatingActivities",
        "NetCashProvidedByOperatingActivitiesContinuingOperations",
    ],
    "capital_expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "dividends_paid": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ],
    "share_repurchases": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
    "free_cash_flow": [
        "FreeCashFlow",
    ],
    "depreciation_cf": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "debt_repayment": [
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
    ],
}


# ── Core fetching ────────────────────────────────────────────────────────────

def get_company_facts(cik: str) -> dict | None:
    """Fetch XBRL company facts from SEC EDGAR.

    Caches as JSON with 7-day TTL. Returns None on 404 or error.
    """
    if not cik:
        return None

    _ensure_cache_dir()
    cache_file = _CACHE_DIR / f"{cik}_facts.json"

    # Check cache
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < _CACHE_TTL_SECS:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    # Fetch from SEC
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = _rate_limited_get(url, timeout=30)
        if resp.status_code == 404:
            logger.info(f"No XBRL facts for CIK {cik}")
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"XBRL fetch failed for CIK {cik}: {e}")
        return None

    # Cache to disk
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
    except OSError as e:
        logger.warning(f"Cache write failed: {e}")

    return data


def extract_annual_series(
    facts: dict,
    concept: str,
    taxonomy: str = "us-gaap",
) -> pd.Series:
    """Extract annual (10-K, FY) values for a given XBRL concept.

    Filters to fp=="FY" + form=="10-K", deduplicates by fiscal year
    (keeps most recently filed). Returns pd.Series indexed by integer year.
    """
    try:
        units_dict = facts["facts"][taxonomy][concept]["units"]
    except (KeyError, TypeError):
        return pd.Series(dtype=float)

    # XBRL values can be in "USD", "USD/shares", "shares", "pure", etc.
    # Try USD first, then USD/shares, then pure, then shares
    records = []
    for unit_key in ["USD", "USD/shares", "shares", "pure"]:
        if unit_key in units_dict:
            records = units_dict[unit_key]
            break

    if not records:
        # Take whatever unit is available
        for unit_key, unit_records in units_dict.items():
            records = unit_records
            break

    if not records:
        return pd.Series(dtype=float)

    # Filter to annual filings
    annual = []
    for r in records:
        fp = r.get("fp", "")
        form = r.get("form", "")
        if fp == "FY" and form == "10-K":
            annual.append(r)

    if not annual:
        return pd.Series(dtype=float)

    # Build DataFrame for dedup
    df = pd.DataFrame(annual)
    df["fy"] = df["fy"].astype(int)

    if "filed" in df.columns:
        df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce")
        # Dedup: keep most recently filed per fiscal year
        df = df.sort_values("filed_dt", ascending=False).drop_duplicates(
            subset=["fy"], keep="first"
        )
    else:
        df = df.drop_duplicates(subset=["fy"], keep="last")

    df = df.sort_values("fy")
    series = pd.Series(df["val"].values, index=df["fy"].values, dtype=float)
    series.index.name = "year"
    return series


def _extract_best_series(facts: dict, concept_list: list[str]) -> pd.Series:
    """Try multiple concept tags, return the longest available series."""
    best = pd.Series(dtype=float)
    for concept in concept_list:
        s = extract_annual_series(facts, concept)
        if len(s) > len(best):
            best = s
    return best


def _extract_best_revenue(facts: dict) -> pd.Series:
    """Extract revenue using 9 fallback concepts, fill gaps from alternatives."""
    best = pd.Series(dtype=float)
    for concept in REVENUE_CONCEPTS:
        s = extract_annual_series(facts, concept)
        if len(s) > len(best):
            best = s

    # Fill gaps from alternative concepts
    if len(best) > 0:
        for concept in REVENUE_CONCEPTS:
            alt = extract_annual_series(facts, concept)
            if len(alt) > 0:
                for year in alt.index:
                    if year not in best.index:
                        best[year] = alt[year]
        best = best.sort_index()

    return best


# ── Main data builder ────────────────────────────────────────────────────────

def get_xbrl_history(ticker: str) -> dict:
    """Orchestrate CIK lookup → facts fetch → extract all → build DataFrame.

    Returns dict with:
        available: bool
        annual_data: pd.DataFrame (indexed by year)
        years_available: int
        ticker: str
        cik: str
        error: str or None
    """
    result = {
        "available": False,
        "annual_data": pd.DataFrame(),
        "years_available": 0,
        "ticker": ticker,
        "cik": None,
        "error": None,
    }

    cik = get_cik(ticker)
    if not cik:
        result["error"] = f"No CIK found for {ticker}"
        return result

    result["cik"] = cik

    facts = get_company_facts(cik)
    if not facts:
        result["error"] = f"No XBRL facts available for {ticker}"
        return result

    # ── Extract all series ────────────────────────────────────────
    data = {}

    # Revenue
    rev = _extract_best_revenue(facts)
    if len(rev) > 0:
        data["revenue"] = rev

    # Income statement
    for key, concepts in INCOME_STMT_CONCEPTS.items():
        s = _extract_best_series(facts, concepts)
        if len(s) > 0:
            data[key] = s

    # Balance sheet
    for key, concepts in BALANCE_SHEET_CONCEPTS.items():
        s = _extract_best_series(facts, concepts)
        if len(s) > 0:
            data[key] = s

    # Cash flow
    for key, concepts in CASH_FLOW_CONCEPTS.items():
        s = _extract_best_series(facts, concepts)
        if len(s) > 0:
            data[key] = s

    if not data:
        result["error"] = f"No extractable XBRL data for {ticker}"
        return result

    # Build DataFrame
    annual_data = pd.DataFrame(data)
    annual_data.index.name = "year"
    annual_data = annual_data.sort_index()

    # ── Derive free cash flow if not directly available ───────────
    if "free_cash_flow" not in annual_data.columns:
        if "operating_cash_flow" in annual_data.columns and "capital_expenditures" in annual_data.columns:
            annual_data["free_cash_flow"] = (
                annual_data["operating_cash_flow"] - annual_data["capital_expenditures"].abs()
            )

    # ── Calculate historical metrics ─────────────────────────────
    annual_data = calculate_historical_metrics(annual_data)

    result["available"] = True
    result["annual_data"] = annual_data
    result["years_available"] = len(annual_data)

    return result


# ── Derived metrics ──────────────────────────────────────────────────────────

def calculate_historical_metrics(annual_data: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: YoY growth rates, margins, ratios."""
    df = annual_data.copy()

    # ── YoY growth rates ─────────────────────────────────────────
    for col in ["revenue", "net_income", "operating_cash_flow", "free_cash_flow",
                "eps_diluted", "eps_basic"]:
        if col in df.columns:
            df[f"{col}_growth"] = df[col].pct_change()

    # ── Margins ──────────────────────────────────────────────────
    if "revenue" in df.columns and "gross_profit" in df.columns:
        df["gross_margin"] = df["gross_profit"] / df["revenue"]

    if "revenue" in df.columns and "net_income" in df.columns:
        df["net_margin"] = df["net_income"] / df["revenue"]

    if "revenue" in df.columns and "operating_income" in df.columns:
        df["operating_margin"] = df["operating_income"] / df["revenue"]

    if "revenue" in df.columns and "free_cash_flow" in df.columns:
        df["fcf_margin"] = df["free_cash_flow"] / df["revenue"]

    # ── Return ratios ────────────────────────────────────────────
    if "net_income" in df.columns and "stockholders_equity" in df.columns:
        df["roe"] = df["net_income"] / df["stockholders_equity"].replace(0, np.nan)

    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)

    # ── Leverage ratios ──────────────────────────────────────────
    if "total_debt" in df.columns and "stockholders_equity" in df.columns:
        df["debt_to_equity"] = df["total_debt"] / df["stockholders_equity"].replace(0, np.nan)

    # Interest coverage from XBRL
    if "operating_income" in df.columns and "interest_expense" in df.columns:
        df["interest_coverage"] = (
            df["operating_income"] / df["interest_expense"].abs().replace(0, np.nan)
        )

    # ── Earnings quality ─────────────────────────────────────────
    if "operating_cash_flow" in df.columns and "net_income" in df.columns:
        df["earnings_quality"] = (
            df["operating_cash_flow"] / df["net_income"].replace(0, np.nan)
        )

    # ── CapEx intensity ──────────────────────────────────────────
    if "capital_expenditures" in df.columns and "revenue" in df.columns:
        df["capex_intensity"] = df["capital_expenditures"].abs() / df["revenue"].replace(0, np.nan)

    return df


# ── Recession resilience ─────────────────────────────────────────────────────

def recession_resilience_score(eps_series: pd.Series) -> dict:
    """Test 2008-2009 and 2020 behavior. Returns score 0-100 + detail.

    Parameters
    ----------
    eps_series : pd.Series
        EPS or net income indexed by integer year.
    """
    result = {
        "score": 50,
        "verdict": "Insufficient Data",
        "detail": {},
        "gfc_impact": None,
        "covid_impact": None,
        "recovery_speed": None,
    }

    if eps_series is None or len(eps_series) < 3:
        return result

    score = 50  # baseline
    years = eps_series.index.tolist()

    # ── GFC 2008-2009 ────────────────────────────────────────────
    gfc_years = [2007, 2008, 2009, 2010]
    gfc_available = [y for y in gfc_years if y in years]

    if 2008 in years and 2007 in years:
        pre_gfc = eps_series[2007]
        gfc_low = eps_series[2008]
        if 2009 in years:
            gfc_low = min(eps_series[2008], eps_series[2009])

        if pre_gfc > 0:
            gfc_decline = (gfc_low - pre_gfc) / abs(pre_gfc)
            result["gfc_impact"] = round(gfc_decline * 100, 1)

            if gfc_decline > -0.10:
                score += 20  # Barely affected
            elif gfc_decline > -0.30:
                score += 10  # Moderate hit
            elif gfc_decline > -0.50:
                score += 0   # Significant hit
            else:
                score -= 15  # Severe hit

            # Recovery check
            if 2010 in years and pre_gfc > 0:
                recovery = eps_series[2010] / pre_gfc
                if recovery >= 1.0:
                    score += 10
                    result["recovery_speed"] = "fast"
                elif recovery >= 0.8:
                    score += 5
                    result["recovery_speed"] = "moderate"
                else:
                    result["recovery_speed"] = "slow"

    # ── COVID 2020 ───────────────────────────────────────────────
    if 2020 in years and 2019 in years:
        pre_covid = eps_series[2019]
        covid_val = eps_series[2020]

        if pre_covid > 0:
            covid_decline = (covid_val - pre_covid) / abs(pre_covid)
            result["covid_impact"] = round(covid_decline * 100, 1)

            if covid_decline > -0.05:
                score += 15  # COVID-proof
            elif covid_decline > -0.20:
                score += 5   # Resilient
            elif covid_decline > -0.50:
                score -= 5   # Affected
            else:
                score -= 15  # Severely affected

            # 2021 bounce-back
            if 2021 in years and pre_covid > 0:
                bounce = eps_series[2021] / pre_covid
                if bounce >= 1.1:
                    score += 5  # Strong bounce

    # ── Clamp and classify ───────────────────────────────────────
    score = max(0, min(100, score))
    result["score"] = score

    if score >= 80:
        result["verdict"] = "Highly Resilient"
    elif score >= 65:
        result["verdict"] = "Resilient"
    elif score >= 45:
        result["verdict"] = "Moderate"
    elif score >= 30:
        result["verdict"] = "Vulnerable"
    else:
        result["verdict"] = "Highly Vulnerable"

    result["detail"] = {
        "gfc_available": len(gfc_available),
        "years_of_data": len(years),
    }

    return result


# ── Data quality badge ───────────────────────────────────────────────────────

def get_data_quality_badge(ticker: str, xbrl_result: dict | None) -> dict:
    """Return a badge dict with label and color for display.

    Returns dict with: label, color, years
    """
    if xbrl_result is None or not xbrl_result.get("available"):
        return {"label": "4yr Data Only", "color": "#888888", "years": 0}

    years = xbrl_result.get("years_available", 0)

    if years >= 10:
        return {"label": f"{years}yr XBRL Data", "color": "#27ae60", "years": years}
    elif years >= 7:
        return {"label": f"{years}yr XBRL Data", "color": "#2980b9", "years": years}
    elif years >= 4:
        return {"label": f"{years}yr XBRL Data", "color": "#f39c12", "years": years}
    else:
        return {"label": "Limited Data", "color": "#e74c3c", "years": years}


# ── Batch loader ─────────────────────────────────────────────────────────────

def load_xbrl_for_tickers(
    tickers: list[str],
    progress_callback=None,
) -> dict:
    """Load XBRL data for multiple tickers with progress reporting.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    progress_callback : callable, optional
        Called with (current_index, total, ticker, status) after each ticker.

    Returns
    -------
    dict
        Keyed by ticker, each value is the result from get_xbrl_history().
    """
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            result = get_xbrl_history(ticker)
            results[ticker] = result
            status = "ok" if result["available"] else "no data"
        except Exception as e:
            logger.warning(f"XBRL fetch failed for {ticker}: {e}")
            results[ticker] = {
                "available": False,
                "annual_data": pd.DataFrame(),
                "years_available": 0,
                "ticker": ticker,
                "cik": None,
                "error": str(e),
            }
            status = "error"

        if progress_callback:
            progress_callback(i, total, ticker, status)

    return results
