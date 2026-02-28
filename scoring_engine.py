"""
Score each stock 0-100 on four pillars using sector-relative percentile ranking.

Pillars: Quality (35%), Growth (30%), Value (25%), Sentiment (10%).
All metrics ranked within sector. Missing data = 50th percentile.
Outliers clipped at 1st/99th percentile before ranking.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Default pillar weights ──────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "quality": 0.35,
    "growth": 0.30,
    "value": 0.25,
    "sentiment": 0.10,
}


# ── Metric extraction helpers ───────────────────────────────────────────────

def _safe_get(info: dict, key: str, default=None):
    """Get a value from info dict, returning default if missing/None/NaN."""
    val = info.get(key, default)
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _yoy_growth(series: pd.Series) -> float | None:
    """YoY growth from most recent vs prior year in a financials row."""
    vals = series.dropna().sort_index(ascending=False)
    if len(vals) < 2:
        return None
    recent, prior = vals.iloc[0], vals.iloc[1]
    if prior == 0:
        return None
    return (recent - prior) / abs(prior)


def _get_financial_row(df: pd.DataFrame, labels: list[str]) -> pd.Series | None:
    """Find the first matching row label in a financials DataFrame."""
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return None


def _extract_metrics(ticker: str, data: dict) -> dict:
    """Extract all raw metrics for a single stock."""
    info = data.get("info", {})
    financials = data.get("financials")
    balance_sheet = data.get("balance_sheet")
    cashflow = data.get("cashflow")

    m = {"ticker": ticker, "sector": data.get("sector", "Unknown")}

    # ── Quality metrics ─────────────────────────────────────────────
    m["roe"] = _safe_get(info, "returnOnEquity")
    m["profit_margin"] = _safe_get(info, "profitMargins")
    m["debt_to_equity"] = _safe_get(info, "debtToEquity")
    m["roa"] = _safe_get(info, "returnOnAssets")

    # Interest coverage: EBITDA / interest expense
    ebitda = _safe_get(info, "ebitda")
    interest = None
    interest_row = _get_financial_row(
        financials, ["Interest Expense", "Interest Expense Non Operating"]
    )
    if interest_row is not None:
        vals = interest_row.dropna()
        if len(vals) > 0:
            interest = abs(vals.iloc[0])
    if ebitda and interest and interest > 0:
        m["interest_coverage"] = ebitda / interest
    else:
        m["interest_coverage"] = None

    # ── Growth metrics ──────────────────────────────────────────────
    revenue_row = _get_financial_row(
        financials, ["Total Revenue", "Revenue", "Operating Revenue"]
    )
    m["revenue_growth"] = _yoy_growth(revenue_row) if revenue_row is not None else None

    ni_row = _get_financial_row(
        financials, ["Net Income", "Net Income Common Stockholders"]
    )
    m["earnings_growth"] = _yoy_growth(ni_row) if ni_row is not None else None

    # FCF growth: (operating cashflow - capex) YoY
    ocf_row = _get_financial_row(
        cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"]
    )
    capex_row = _get_financial_row(
        cashflow, ["Capital Expenditure", "Capital Expenditures"]
    )
    if ocf_row is not None and capex_row is not None:
        ocf = ocf_row.dropna().sort_index(ascending=False)
        capex = capex_row.dropna().sort_index(ascending=False)
        # Align on common dates
        common = ocf.index.intersection(capex.index)
        if len(common) >= 2:
            fcf = ocf[common] - abs(capex[common])
            fcf_sorted = fcf.sort_index(ascending=False)
            if len(fcf_sorted) >= 2 and abs(fcf_sorted.iloc[1]) > 0:
                m["fcf_growth"] = (
                    (fcf_sorted.iloc[0] - fcf_sorted.iloc[1]) / abs(fcf_sorted.iloc[1])
                )
            else:
                m["fcf_growth"] = None
        else:
            m["fcf_growth"] = None
    else:
        m["fcf_growth"] = None

    # Forward EPS growth
    fwd_eps = _safe_get(info, "forwardEps")
    trail_eps = _safe_get(info, "trailingEps")
    if fwd_eps and trail_eps and trail_eps > 0:
        m["fwd_eps_growth"] = (fwd_eps - trail_eps) / abs(trail_eps)
    else:
        m["fwd_eps_growth"] = None

    # ── Value metrics (INVERTED — lower = better) ───────────────────
    m["trailing_pe"] = _safe_get(info, "trailingPE")
    m["ev_ebitda"] = _safe_get(info, "enterpriseToEbitda")
    m["price_to_book"] = _safe_get(info, "priceToBook")

    # ── Sentiment proxy metrics ─────────────────────────────────────
    price = _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice")
    high52 = _safe_get(info, "fiftyTwoWeekHigh")
    if price and high52 and high52 > 0:
        m["pct_of_52wk_high"] = price / high52
    else:
        m["pct_of_52wk_high"] = None

    ma50 = _safe_get(info, "fiftyDayAverage")
    ma200 = _safe_get(info, "twoHundredDayAverage")
    if ma50 and ma200 and ma200 > 0:
        m["ma50_vs_ma200"] = ma50 / ma200
    else:
        m["ma50_vs_ma200"] = None

    # 3-month return (approximate from 52-week low and current price data)
    low52 = _safe_get(info, "fiftyTwoWeekLow")
    m["price_return_3m"] = None  # computed sector-relative later; use ma50 as proxy
    if price and ma200 and ma200 > 0:
        m["price_return_3m"] = (price - ma200) / ma200

    return m


# ── Percentile ranking engine ───────────────────────────────────────────────

def _percentile_rank_in_sector(
    df: pd.DataFrame,
    metric_col: str,
    inverted: bool = False,
) -> pd.Series:
    """Rank a metric within sector as 0-100 percentile.

    Missing values get 50. Clips at 1st/99th before ranking.
    inverted=True means lower raw value = higher score.
    """
    result = pd.Series(50.0, index=df.index)

    for sector, group in df.groupby("sector"):
        vals = group[metric_col].copy()
        non_null = vals.dropna()

        if len(non_null) < 2:
            continue

        # Clip at 1st/99th percentile
        lo = non_null.quantile(0.01)
        hi = non_null.quantile(0.99)
        clipped = non_null.clip(lo, hi)

        if inverted:
            clipped = -clipped

        # Percentile rank within sector (0-100)
        ranked = clipped.rank(pct=True) * 100
        result.loc[ranked.index] = ranked

    # Fill any remaining NaN with 50
    result = result.fillna(50.0)
    return result


def _pillar_score(
    df: pd.DataFrame,
    metric_configs: list[tuple[str, float, bool]],
) -> pd.Series:
    """Compute a pillar score as weighted sum of percentile-ranked metrics.

    metric_configs: list of (column_name, sub_weight, inverted)
    """
    score = pd.Series(0.0, index=df.index)
    for col, weight, inverted in metric_configs:
        pctile = _percentile_rank_in_sector(df, col, inverted)
        score += pctile * weight
    return score


# ── Main scoring function ───────────────────────────────────────────────────

def score_all_stocks(
    all_ticker_data: dict,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Score all stocks and return a DataFrame.

    Parameters
    ----------
    all_ticker_data : dict
        Keyed by ticker, from data_fetcher.
    weights : dict, optional
        Pillar weights (quality, growth, value, sentiment). Default = DEFAULT_WEIGHTS.

    Returns
    -------
    DataFrame with columns:
        ticker, sector, quality_score, growth_score, value_score,
        sentiment_score, composite_score, sector_rank, sector_total
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # Extract metrics for all stocks
    rows = []
    for ticker, data in all_ticker_data.items():
        rows.append(_extract_metrics(ticker, data))
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    # ── Quality pillar ──────────────────────────────────────────────
    df["quality_score"] = _pillar_score(df, [
        ("roe", 0.30, False),
        ("profit_margin", 0.25, False),
        ("debt_to_equity", 0.20, True),    # inverted: lower D/E is better
        ("roa", 0.15, False),
        ("interest_coverage", 0.10, False),
    ])

    # ── Growth pillar ───────────────────────────────────────────────
    df["growth_score"] = _pillar_score(df, [
        ("revenue_growth", 0.30, False),
        ("earnings_growth", 0.30, False),
        ("fcf_growth", 0.20, False),
        ("fwd_eps_growth", 0.20, False),
    ])

    # ── Value pillar (all inverted) ─────────────────────────────────
    df["value_score"] = _pillar_score(df, [
        ("trailing_pe", 0.35, True),
        ("ev_ebitda", 0.35, True),
        ("price_to_book", 0.30, True),
    ])

    # ── Sentiment pillar ────────────────────────────────────────────
    df["sentiment_score"] = _pillar_score(df, [
        ("pct_of_52wk_high", 0.40, False),
        ("ma50_vs_ma200", 0.30, False),
        ("price_return_3m", 0.30, False),
    ])

    # ── Composite score ─────────────────────────────────────────────
    df["composite_score"] = (
        df["quality_score"] * w["quality"]
        + df["growth_score"] * w["growth"]
        + df["value_score"] * w["value"]
        + df["sentiment_score"] * w["sentiment"]
    )

    # Round scores
    for col in ["quality_score", "growth_score", "value_score",
                "sentiment_score", "composite_score"]:
        df[col] = df[col].round(1)

    # ── Sector rank ─────────────────────────────────────────────────
    df["sector_rank"] = df.groupby("sector")["composite_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    df["sector_total"] = df.groupby("sector")["ticker"].transform("count")

    # Select output columns
    out_cols = [
        "ticker", "sector", "quality_score", "growth_score",
        "value_score", "sentiment_score", "composite_score",
        "sector_rank", "sector_total",
    ]
    return df[out_cols].sort_values("composite_score", ascending=False).reset_index(drop=True)
