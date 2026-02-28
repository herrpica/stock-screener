"""
Rate each stock as "Above Bar", "At Bar", or "Below Bar" based on
historical consistency over 4 years of annual financial data.

Sub-ratings:
  - Earnings Consistency (40%)
  - Debt Discipline (25%)
  - Dividend Quality (20%)
  - Buyback Discipline (15%)
"""

import numpy as np
import pandas as pd


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_row(df: pd.DataFrame, labels: list[str]) -> pd.Series | None:
    """Find first matching row label in a financials DataFrame."""
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return None


def _is_2020(date) -> bool:
    """Check if a pandas Timestamp falls in 2020 (COVID macro year)."""
    try:
        return pd.Timestamp(date).year == 2020
    except Exception:
        return False


# ── Earnings Consistency (40%) ──────────────────────────────────────────────

def _score_earnings_consistency(ticker_data: dict) -> tuple[float, dict]:
    """Score earnings consistency from annual Net Income history.

    Returns (score 0-100, detail dict).
    """
    financials = ticker_data.get("financials")
    ni = _get_row(financials, ["Net Income", "Net Income Common Stockholders"])
    if ni is None:
        return 50.0, {"years": 0, "growth_years": 0, "note": "No data"}

    vals = ni.dropna().sort_index()  # ascending by date
    if len(vals) < 2:
        return 50.0, {"years": len(vals), "growth_years": 0, "note": "Insufficient data"}

    growth_years = 0
    total_comparisons = 0

    for i in range(1, len(vals)):
        # Excuse 2020 drops
        if _is_2020(vals.index[i]) and vals.iloc[i] < vals.iloc[i - 1]:
            continue
        total_comparisons += 1
        if vals.iloc[i] > vals.iloc[i - 1]:
            growth_years += 1

    if total_comparisons == 0:
        pct = 1.0
    else:
        pct = growth_years / total_comparisons

    if pct >= 0.80:
        score = 90.0
    elif pct >= 0.60:
        score = 65.0
    else:
        score = 30.0

    detail = {
        "years": len(vals),
        "growth_years": growth_years,
        "total_comparisons": total_comparisons,
        "growth_pct": round(pct * 100, 1),
    }
    return score, detail


# ── Debt Discipline (25%) ───────────────────────────────────────────────────

def _score_debt_discipline(
    ticker_data: dict,
    sector_median_de: float | None,
) -> tuple[float, dict]:
    """Score debt discipline from D/E trend and sector comparison.

    Returns (score 0-100, detail dict).
    """
    bs = ticker_data.get("balance_sheet")
    debt_row = _get_row(bs, ["Total Debt", "Long Term Debt"])
    equity_row = _get_row(bs, [
        "Total Stockholders Equity", "Stockholders Equity",
        "Common Stock Equity",
    ])

    if debt_row is None or equity_row is None:
        return 50.0, {"note": "No balance sheet data"}

    debt = debt_row.dropna().sort_index()
    equity = equity_row.dropna().sort_index()
    common = debt.index.intersection(equity.index)

    if len(common) < 2:
        return 50.0, {"note": "Insufficient history"}

    de_series = debt[common] / equity[common].replace(0, np.nan)
    de_series = de_series.dropna().sort_index()

    if len(de_series) < 2:
        return 50.0, {"note": "D/E computation failed"}

    current_de = de_series.iloc[-1]

    # Trend: linear regression slope
    x = np.arange(len(de_series))
    slope = np.polyfit(x, de_series.values, 1)[0]

    score = 50.0

    # Trend component (is D/E declining or flat?)
    if slope < -0.05:
        score += 25  # Declining significantly
    elif slope < 0.02:
        score += 10  # Roughly flat
    else:
        score -= 15  # Increasing

    # Level component (vs sector median)
    if sector_median_de is not None and sector_median_de > 0:
        if current_de < sector_median_de * 0.8:
            score += 20
        elif current_de < sector_median_de * 1.2:
            score += 5
        else:
            score -= 10

    score = max(0, min(100, score))

    detail = {
        "current_de": round(float(current_de), 2),
        "de_trend_slope": round(float(slope), 4),
        "sector_median_de": round(float(sector_median_de), 2) if sector_median_de else None,
        "years": len(de_series),
    }
    return score, detail


# ── Dividend Quality (20%) ──────────────────────────────────────────────────

def _score_dividend_quality(ticker_data: dict) -> tuple[float, dict]:
    """Score dividend quality from payment history.

    No dividends = neutral 50. Cuts = penalty. Consecutive increases = bonus.
    """
    dividends = ticker_data.get("dividends")
    if dividends is None or len(dividends) == 0:
        return 50.0, {"note": "No dividends paid", "has_dividends": False}

    # Resample to annual sums
    annual = dividends.resample("YE").sum()
    annual = annual[annual > 0]

    if len(annual) < 2:
        return 50.0, {"note": "Insufficient dividend history", "has_dividends": True}

    # Count consecutive increases and any cuts
    consecutive_increases = 0
    max_consecutive = 0
    cuts = 0

    for i in range(1, len(annual)):
        if annual.iloc[i] > annual.iloc[i - 1]:
            consecutive_increases += 1
            max_consecutive = max(max_consecutive, consecutive_increases)
        elif annual.iloc[i] < annual.iloc[i - 1] * 0.95:  # 5% tolerance
            cuts += 1
            consecutive_increases = 0
        else:
            consecutive_increases = 0  # flat

    score = 50.0

    # Consecutive increase bonus
    if max_consecutive >= 10:
        score += 35  # Aristocrat-level
    elif max_consecutive >= 5:
        score += 20
    elif max_consecutive >= 3:
        score += 10

    # Cut penalty
    score -= cuts * 15

    score = max(0, min(100, score))

    detail = {
        "has_dividends": True,
        "years_of_data": len(annual),
        "max_consecutive_increases": max_consecutive,
        "dividend_cuts": cuts,
    }
    return score, detail


# ── Buyback Discipline (15%) ───────────────────────────────────────────────

def _score_buyback_discipline(ticker_data: dict) -> tuple[float, dict]:
    """Score buyback discipline from shares outstanding trend.

    Declining shares = positive, increasing = negative.
    """
    bs = ticker_data.get("balance_sheet")
    shares_row = _get_row(bs, [
        "Ordinary Shares Number", "Share Issued",
        "Common Stock", "Basic Average Shares",
    ])

    if shares_row is None:
        # Try info field
        info = ticker_data.get("info", {})
        shares = info.get("sharesOutstanding")
        if shares:
            return 50.0, {"note": "Only current shares available", "trend": "unknown"}
        return 50.0, {"note": "No shares data", "trend": "unknown"}

    vals = shares_row.dropna().sort_index()
    if len(vals) < 2:
        return 50.0, {"note": "Insufficient history", "trend": "unknown"}

    # Trend
    oldest = vals.iloc[0]
    newest = vals.iloc[-1]

    if oldest == 0:
        return 50.0, {"note": "Zero shares in history", "trend": "unknown"}

    change_pct = (newest - oldest) / abs(oldest)

    if change_pct < -0.05:
        score = 75.0  # Meaningful buyback
    elif change_pct < -0.01:
        score = 60.0  # Slight buyback
    elif change_pct < 0.02:
        score = 50.0  # Flat
    elif change_pct < 0.10:
        score = 35.0  # Mild dilution
    else:
        score = 20.0  # Significant dilution

    detail = {
        "shares_change_pct": round(change_pct * 100, 1),
        "trend": "declining" if change_pct < -0.01 else (
            "flat" if change_pct < 0.02 else "increasing"
        ),
        "years": len(vals),
    }
    return score, detail


# ── Main rating function ────────────────────────────────────────────────────

def rate_all_stocks(
    all_ticker_data: dict,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rate all stocks and return a DataFrame.

    Parameters
    ----------
    all_ticker_data : dict
        From data_fetcher.
    scores_df : pd.DataFrame
        From scoring_engine (used for sector medians).

    Returns
    -------
    DataFrame with columns:
        ticker, quality_rating, earnings_consistency_score,
        debt_discipline_score, dividend_quality_score, buyback_score,
        rating_detail
    """
    # Pre-compute sector median D/E for debt discipline scoring
    sector_de_medians = {}
    for ticker, data in all_ticker_data.items():
        info = data.get("info", {})
        de = info.get("debtToEquity")
        sector = data.get("sector", "Unknown")
        if de is not None:
            sector_de_medians.setdefault(sector, []).append(de)
    for sector in sector_de_medians:
        sector_de_medians[sector] = float(np.median(sector_de_medians[sector]))

    rows = []
    for ticker, data in all_ticker_data.items():
        sector = data.get("sector", "Unknown")
        sector_med_de = sector_de_medians.get(sector)

        ec_score, ec_detail = _score_earnings_consistency(data)
        dd_score, dd_detail = _score_debt_discipline(data, sector_med_de)
        dq_score, dq_detail = _score_dividend_quality(data)
        bb_score, bb_detail = _score_buyback_discipline(data)

        # Weighted total
        total = (
            ec_score * 0.40
            + dd_score * 0.25
            + dq_score * 0.20
            + bb_score * 0.15
        )

        if total >= 75:
            rating = "Above Bar"
        elif total >= 50:
            rating = "At Bar"
        else:
            rating = "Below Bar"

        rows.append({
            "ticker": ticker,
            "quality_rating": rating,
            "earnings_consistency_score": round(ec_score, 1),
            "debt_discipline_score": round(dd_score, 1),
            "dividend_quality_score": round(dq_score, 1),
            "buyback_score": round(bb_score, 1),
            "rating_total": round(total, 1),
            "rating_detail": {
                "earnings": ec_detail,
                "debt": dd_detail,
                "dividends": dq_detail,
                "buybacks": bb_detail,
            },
        })

    return pd.DataFrame(rows)
