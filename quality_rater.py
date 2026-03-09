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


# ── XBRL-Enhanced Ratings ────────────────────────────────────────────────────

def rate_earnings_consistency_xbrl(
    ticker: str,
    xbrl_result: dict,
) -> tuple[float, dict]:
    """Score earnings consistency from full XBRL history.

    Improvements over base:
    - 10-15+ years of history instead of 4
    - Crisis year excusal (2008-2009, 2020)
    - Consecutive growth streak bonus
    - Earnings quality ratio (OCF/NI)
    - Recession resilience adjustment
    """
    if not xbrl_result or not xbrl_result.get("available"):
        return 50.0, {"note": "No XBRL data", "source": "fallback"}

    annual = xbrl_result["annual_data"]
    if "net_income" not in annual.columns:
        return 50.0, {"note": "No net income in XBRL", "source": "fallback"}

    ni = annual["net_income"].dropna().sort_index()
    if len(ni) < 3:
        return 50.0, {"note": "Insufficient XBRL history", "source": "fallback"}

    # ── Count growth years (excusing crisis years) ───────────────
    crisis_years = {2008, 2009, 2020}
    growth_years = 0
    total_comparisons = 0
    consecutive = 0
    max_consecutive = 0

    for i in range(1, len(ni)):
        year = ni.index[i]
        # Excuse crisis year drops
        if year in crisis_years and ni.iloc[i] < ni.iloc[i - 1]:
            continue
        total_comparisons += 1
        if ni.iloc[i] > ni.iloc[i - 1]:
            growth_years += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    if total_comparisons == 0:
        pct = 1.0
    else:
        pct = growth_years / total_comparisons

    # ── Base score ───────────────────────────────────────────────
    if pct >= 0.80:
        score = 90.0
    elif pct >= 0.70:
        score = 78.0
    elif pct >= 0.60:
        score = 65.0
    elif pct >= 0.50:
        score = 50.0
    else:
        score = 30.0

    # ── Streak bonus ─────────────────────────────────────────────
    if max_consecutive >= 8:
        score += 8
    elif max_consecutive >= 5:
        score += 4

    # ── Earnings quality ratio (OCF / NI) ────────────────────────
    eq_bonus = 0
    if "earnings_quality" in annual.columns:
        eq = annual["earnings_quality"].dropna()
        if len(eq) >= 3:
            avg_eq = eq.median()
            if avg_eq >= 1.2:
                eq_bonus = 5  # Strong cash backing
            elif avg_eq < 0.6:
                eq_bonus = -5  # Poor cash conversion

    # ── Recession resilience adjustment ──────────────────────────
    resilience_adj = 0
    try:
        from edgar_xbrl_fetcher import recession_resilience_score
        eps_col = None
        for col in ["eps_diluted", "eps_basic"]:
            if col in annual.columns:
                eps_col = col
                break
        test_series = annual[eps_col].dropna() if eps_col else ni
        rr = recession_resilience_score(test_series)
        if rr["score"] >= 75:
            resilience_adj = 5
        elif rr["score"] <= 30:
            resilience_adj = -5
    except ImportError:
        rr = None

    score = max(0, min(100, score + eq_bonus + resilience_adj))

    detail = {
        "source": "xbrl",
        "years": len(ni),
        "growth_years": growth_years,
        "total_comparisons": total_comparisons,
        "growth_pct": round(pct * 100, 1),
        "max_consecutive_growth": max_consecutive,
        "earnings_quality_bonus": eq_bonus,
        "resilience_adj": resilience_adj,
    }
    return score, detail


def rate_debt_discipline_xbrl(
    ticker: str,
    xbrl_result: dict,
    sector_median_de: float | None = None,
) -> tuple[float, dict]:
    """Score debt discipline from full XBRL D/E history.

    Improvements over base:
    - Full 10-15yr D/E trend
    - Peak D/E analysis
    - Low-rate-era (2020-2021) debt loading detection
    - Debt growth rate tracking
    """
    if not xbrl_result or not xbrl_result.get("available"):
        return 50.0, {"note": "No XBRL data", "source": "fallback"}

    annual = xbrl_result["annual_data"]

    has_debt = "total_debt" in annual.columns
    has_equity = "stockholders_equity" in annual.columns

    if not has_debt or not has_equity:
        return 50.0, {"note": "Missing debt/equity in XBRL", "source": "fallback"}

    debt = annual["total_debt"].dropna()
    equity = annual["stockholders_equity"].dropna()
    common_years = debt.index.intersection(equity.index)

    if len(common_years) < 2:
        return 50.0, {"note": "Insufficient D/E history", "source": "fallback"}

    de_series = (debt[common_years] / equity[common_years].replace(0, np.nan)).dropna()
    de_series = de_series.sort_index()

    if len(de_series) < 2:
        return 50.0, {"note": "D/E computation failed", "source": "fallback"}

    current_de = de_series.iloc[-1]
    peak_de = de_series.max()
    peak_de_year = int(de_series.idxmax())

    # ── Trend via linear regression ──────────────────────────────
    x = np.arange(len(de_series))
    slope = np.polyfit(x, de_series.values, 1)[0]

    score = 50.0

    # Trend component
    if slope < -0.05:
        score += 25  # Declining significantly
    elif slope < 0.02:
        score += 10  # Roughly flat
    else:
        score -= 15  # Increasing

    # Level vs sector
    if sector_median_de is not None and sector_median_de > 0:
        if current_de < sector_median_de * 0.8:
            score += 20
        elif current_de < sector_median_de * 1.2:
            score += 5
        else:
            score -= 10

    # ── Low-rate era debt loading (2020-2021) ────────────────────
    low_rate_penalty = 0
    if 2019 in de_series.index and 2021 in de_series.index:
        pre_era_de = de_series[2019]
        era_de = de_series[2021]
        if pre_era_de > 0:
            de_jump = (era_de - pre_era_de) / pre_era_de
            if de_jump > 0.30:
                low_rate_penalty = -10  # Loaded up on cheap debt
            elif de_jump > 0.15:
                low_rate_penalty = -5

    # ── Debt growth rate ─────────────────────────────────────────
    debt_growth_penalty = 0
    if len(debt) >= 3:
        oldest_debt = debt.iloc[0]
        newest_debt = debt.iloc[-1]
        if oldest_debt > 0 and newest_debt > 0:
            n_years = debt.index[-1] - debt.index[0]
            if n_years > 0:
                debt_cagr = (newest_debt / oldest_debt) ** (1 / n_years) - 1
                if debt_cagr > 0.15:
                    debt_growth_penalty = -8
                elif debt_cagr > 0.08:
                    debt_growth_penalty = -4
            else:
                debt_cagr = 0
        else:
            debt_cagr = 0
    else:
        debt_cagr = 0

    score = max(0, min(100, score + low_rate_penalty + debt_growth_penalty))

    detail = {
        "source": "xbrl",
        "current_de": round(float(current_de), 3),
        "peak_de": round(float(peak_de), 3),
        "peak_de_year": peak_de_year,
        "de_trend_slope": round(float(slope), 4),
        "sector_median_de": round(float(sector_median_de), 2) if sector_median_de else None,
        "low_rate_era_penalty": low_rate_penalty,
        "debt_growth_rate": round(float(debt_cagr), 4) if isinstance(debt_cagr, (int, float)) else None,
        "debt_growth_penalty": debt_growth_penalty,
        "years": len(de_series),
    }
    return score, detail


def rate_all_stocks_xbrl(
    all_ticker_data: dict,
    scores_df: pd.DataFrame,
    xbrl_data: dict | None = None,
) -> pd.DataFrame:
    """Enhanced version of rate_all_stocks() using XBRL where available.

    Falls back per-ticker to yfinance-based scoring if XBRL unavailable.
    """
    xbrl_data = xbrl_data or {}

    # Pre-compute sector median D/E
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
        xbrl = xbrl_data.get(ticker)
        has_xbrl = xbrl and xbrl.get("available", False)

        # Earnings consistency: prefer XBRL
        if has_xbrl:
            ec_score, ec_detail = rate_earnings_consistency_xbrl(ticker, xbrl)
        else:
            ec_score, ec_detail = _score_earnings_consistency(data)

        # Debt discipline: prefer XBRL
        if has_xbrl:
            dd_score, dd_detail = rate_debt_discipline_xbrl(ticker, xbrl, sector_med_de)
        else:
            dd_score, dd_detail = _score_debt_discipline(data, sector_med_de)

        # Dividend quality (no XBRL enhancement yet)
        dq_score, dq_detail = _score_dividend_quality(data)

        # Buyback discipline (no XBRL enhancement yet)
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
