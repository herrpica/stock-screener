"""
Two-stage discounted earnings model for intrinsic value calculation.

Stage 1 (years 1-5): grow EPS at observed CAGR (capped at 25%, floored at -5%)
Stage 2 (years 6-10): grow at min(observed, 8%)
Terminal value: Year 10 EPS * 15 (conservative P/E exit)
Discount at required_rate (default 10%)
"""

import numpy as np
import pandas as pd


def _compute_earnings_growth(ticker_data: dict) -> float | None:
    """Compute CAGR from 4 years of annual net income.

    Returns growth rate or None if earnings are negative/unavailable.
    """
    financials = ticker_data.get("financials")
    if financials is None or financials.empty:
        return None

    # financials columns are dates, rows are line items
    # Get Net Income row
    net_income = None
    for label in ["Net Income", "Net Income Common Stockholders"]:
        if label in financials.index:
            net_income = financials.loc[label].dropna().sort_index()
            break

    if net_income is None or len(net_income) < 2:
        return None

    # Use oldest and most recent for CAGR
    oldest = net_income.iloc[-1]  # oldest (columns sorted ascending by date)
    newest = net_income.iloc[0]   # most recent

    if oldest <= 0 or newest <= 0:
        return None

    years = len(net_income) - 1
    if years <= 0:
        return None

    cagr = (newest / oldest) ** (1 / years) - 1

    # Cap at 25%, floor at -5%
    cagr = max(-0.05, min(0.25, cagr))
    return cagr


def calculate_intrinsic_value(
    ticker_data: dict,
    required_rate: float = 0.10,
) -> dict:
    """Calculate intrinsic value per share using two-stage DCF.

    Returns dict with:
        intrinsic_value, margin_of_safety, stage1_values, stage2_values,
        growth_rate_used, base_eps, terminal_value, current_price, error
    """
    info = ticker_data.get("info", {})
    base_eps = info.get("trailingEps")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    result = {
        "intrinsic_value": None,
        "margin_of_safety": None,
        "stage1_values": [],
        "stage2_values": [],
        "terminal_value": None,
        "growth_rate_used": None,
        "base_eps": base_eps,
        "current_price": current_price,
        "error": None,
    }

    if base_eps is None or base_eps <= 0:
        result["error"] = "N/A - Unprofitable"
        return result

    if current_price is None or current_price <= 0:
        result["error"] = "N/A - No price data"
        return result

    growth_rate = _compute_earnings_growth(ticker_data)
    if growth_rate is None:
        # Fallback: use info field
        fwd = info.get("forwardEps")
        if fwd and base_eps > 0:
            growth_rate = max(-0.05, min(0.25, (fwd / base_eps) - 1))
        else:
            growth_rate = 0.05  # conservative default

    result["growth_rate_used"] = growth_rate

    # Stage 1: years 1-5 at observed growth rate
    stage1 = []
    eps = base_eps
    for year in range(1, 6):
        eps = eps * (1 + growth_rate)
        pv = eps / (1 + required_rate) ** year
        stage1.append({"year": year, "eps": eps, "pv": pv})

    # Stage 2: years 6-10 at min(observed, 8%)
    stage2_rate = min(growth_rate, 0.08)
    stage2 = []
    for year in range(6, 11):
        eps = eps * (1 + stage2_rate)
        pv = eps / (1 + required_rate) ** year
        stage2.append({"year": year, "eps": eps, "pv": pv})

    # Terminal value: Year 10 EPS * 15, discounted back
    terminal_value = eps * 15
    terminal_pv = terminal_value / (1 + required_rate) ** 10

    # Sum all PVs
    total_pv = sum(s["pv"] for s in stage1) + sum(s["pv"] for s in stage2) + terminal_pv
    intrinsic_value = total_pv

    # Margin of safety
    margin = (intrinsic_value - current_price) / intrinsic_value * 100

    result["intrinsic_value"] = round(intrinsic_value, 2)
    result["margin_of_safety"] = round(margin, 2)
    result["stage1_values"] = stage1
    result["stage2_values"] = stage2
    result["terminal_value"] = round(terminal_pv, 2)

    return result


def get_earnings_growth_rate(
    ticker: str,
    ticker_data: dict,
    xbrl_result: dict | None = None,
) -> dict:
    """Calculate earnings growth using XBRL history when available.

    Computes CAGR at 4yr, 7yr, 10yr horizons from XBRL net income.
    Uses median of available CAGRs for robustness.
    Falls back to _compute_earnings_growth() if XBRL unavailable.

    Returns dict with:
        growth_rate: float (the recommended rate)
        cagr_4yr, cagr_7yr, cagr_10yr: float or None
        trend: "accelerating" | "decelerating" | "stable"
        profitable_pct: float (0-1)
        recession_resilience: dict (from recession_resilience_score)
        data_source: "xbrl" | "yfinance"
        horizons_used: int
    """
    result = {
        "growth_rate": None,
        "cagr_4yr": None,
        "cagr_7yr": None,
        "cagr_10yr": None,
        "trend": "stable",
        "profitable_pct": None,
        "recession_resilience": None,
        "data_source": "yfinance",
        "horizons_used": 0,
    }

    # Try XBRL first
    if xbrl_result and xbrl_result.get("available"):
        annual = xbrl_result["annual_data"]
        if "net_income" in annual.columns:
            ni = annual["net_income"].dropna()
            if len(ni) >= 3:
                result["data_source"] = "xbrl"
                years = ni.index.tolist()
                latest_year = max(years)
                latest_val = ni[latest_year]

                if latest_val <= 0:
                    # Can't compute meaningful CAGR with negative latest
                    pass
                else:
                    cagrs = []
                    # 4-year CAGR
                    target_4 = latest_year - 4
                    candidates_4 = [y for y in years if y <= target_4]
                    if candidates_4 and ni[max(candidates_4)] > 0:
                        base_year = max(candidates_4)
                        n = latest_year - base_year
                        cagr = (latest_val / ni[base_year]) ** (1 / n) - 1
                        result["cagr_4yr"] = round(max(-0.10, min(0.40, cagr)), 4)
                        cagrs.append(result["cagr_4yr"])

                    # 7-year CAGR
                    target_7 = latest_year - 7
                    candidates_7 = [y for y in years if y <= target_7]
                    if candidates_7 and ni[max(candidates_7)] > 0:
                        base_year = max(candidates_7)
                        n = latest_year - base_year
                        cagr = (latest_val / ni[base_year]) ** (1 / n) - 1
                        result["cagr_7yr"] = round(max(-0.10, min(0.40, cagr)), 4)
                        cagrs.append(result["cagr_7yr"])

                    # 10-year CAGR
                    target_10 = latest_year - 10
                    candidates_10 = [y for y in years if y <= target_10]
                    if candidates_10 and ni[max(candidates_10)] > 0:
                        base_year = max(candidates_10)
                        n = latest_year - base_year
                        cagr = (latest_val / ni[base_year]) ** (1 / n) - 1
                        result["cagr_10yr"] = round(max(-0.10, min(0.40, cagr)), 4)
                        cagrs.append(result["cagr_10yr"])

                    result["horizons_used"] = len(cagrs)

                    if cagrs:
                        # Use median of available CAGRs
                        median_cagr = float(np.median(cagrs))
                        result["growth_rate"] = round(
                            max(-0.05, min(0.25, median_cagr)), 4
                        )

                    # Trend detection
                    if result["cagr_4yr"] is not None and result["cagr_10yr"] is not None:
                        diff = result["cagr_4yr"] - result["cagr_10yr"]
                        if diff > 0.03:
                            result["trend"] = "accelerating"
                        elif diff < -0.03:
                            result["trend"] = "decelerating"

                # Profitable %
                profitable_years = (ni > 0).sum()
                result["profitable_pct"] = round(profitable_years / len(ni), 3)

                # Recession resilience (lazy import to avoid circular)
                try:
                    from edgar_xbrl_fetcher import recession_resilience_score
                    eps_col = "eps_diluted" if "eps_diluted" in annual.columns else None
                    if eps_col is None:
                        eps_col = "eps_basic" if "eps_basic" in annual.columns else None
                    if eps_col:
                        eps = annual[eps_col].dropna()
                    else:
                        eps = ni  # use net income as proxy
                    result["recession_resilience"] = recession_resilience_score(eps)
                except ImportError:
                    pass

    # Fall back to yfinance-based calculation
    if result["growth_rate"] is None:
        yf_rate = _compute_earnings_growth(ticker_data)
        if yf_rate is not None:
            result["growth_rate"] = yf_rate
            result["data_source"] = "yfinance"

    return result


def calculate_all_valuations(
    all_ticker_data: dict,
    required_rate: float = 0.10,
) -> dict:
    """Run valuation for every ticker. Returns dict keyed by ticker."""
    valuations = {}
    for ticker, data in all_ticker_data.items():
        valuations[ticker] = calculate_intrinsic_value(data, required_rate)
    return valuations
