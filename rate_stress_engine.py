"""
Rate Stress Engine — Full DCF revaluation under sustained high-rate scenarios.

For each stock, runs the same two-stage DCF from valuation_engine.py with:
  1. Stress discount rate (replacing the base 10%)
  2. Earnings base adjustment for rate-exposed companies (debt refinancing impact)

Output: stress-adjusted intrinsic value, stress margin of safety, and a
plain-English verdict for every stock.
"""

import numpy as np

from valuation_engine import _compute_earnings_growth


# ── Scenarios ───────────────────────────────────────────────────────────────

SCENARIOS = {
    "normalized": {
        "rate": 0.09,
        "label": "Mild Normalization (9%)",
        "description": (
            "Rates ease modestly but don't return to pre-2022 lows. "
            "New normal above historical average."
        ),
    },
    "sustained_high": {
        "rate": 0.12,
        "label": "Sustained High (12%)",
        "description": (
            "Rates remain structurally elevated. No normalization."
        ),
    },
    "severe": {
        "rate": 0.14,
        "label": "Severe Stress (14%)",
        "description": (
            "Dollar pressure or credit spread widening forces "
            "rates significantly higher."
        ),
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_financial_value(df, labels):
    """Get the most recent value from a financials DataFrame row."""
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            vals = df.loc[label].dropna().sort_index(ascending=False)
            if len(vals) > 0:
                return float(vals.iloc[0])
    return None


def _run_two_stage_dcf(base_eps, growth_rate, discount_rate):
    """Run the exact same two-stage DCF as valuation_engine.calculate_intrinsic_value().

    Stage 1 (years 1-5): grow at growth_rate
    Stage 2 (years 6-10): grow at min(growth_rate, 8%)
    Terminal: Year 10 EPS * 15, discounted back
    """
    eps = base_eps
    total_pv = 0.0

    # Stage 1
    for year in range(1, 6):
        eps = eps * (1 + growth_rate)
        total_pv += eps / (1 + discount_rate) ** year

    # Stage 2
    stage2_rate = min(growth_rate, 0.08)
    for year in range(6, 11):
        eps = eps * (1 + stage2_rate)
        total_pv += eps / (1 + discount_rate) ** year

    # Terminal value
    terminal_value = eps * 15
    terminal_pv = terminal_value / (1 + discount_rate) ** 10
    total_pv += terminal_pv

    return round(total_pv, 2)


# ── Core stress DCF ────────────────────────────────────────────────────────

def calculate_stress_dcf(
    ticker: str,
    ticker_data: dict,
    base_valuation: dict,
    stress_rate: float,
) -> dict:
    """Rerun full two-stage DCF with stress discount rate and earnings adjustment.

    Returns dict with stress_intrinsic_value, stress_margin_of_safety,
    stress_status, stress_base_eps, and supporting detail.
    """
    info = ticker_data.get("info", {})
    financials = ticker_data.get("financials")
    balance_sheet = ticker_data.get("balance_sheet")

    base_eps = base_valuation.get("base_eps")
    current_price = base_valuation.get("current_price")
    growth_rate = base_valuation.get("growth_rate_used")

    result = {
        "ticker": ticker,
        "stress_rate": stress_rate,
        "base_eps": base_eps,
        "stress_base_eps": None,
        "stress_intrinsic_value": None,
        "stress_margin_of_safety": None,
        "stress_status": None,
        "earnings_impact_pct": None,
        "incremental_interest": None,
        "net_cash_position": None,
        "total_debt": None,
        "current_interest_expense": None,
        "error": None,
    }

    # Guard: need valid base valuation
    if base_valuation.get("error") or base_eps is None or base_eps <= 0:
        result["error"] = base_valuation.get("error", "N/A - No base EPS")
        return result

    if current_price is None or current_price <= 0:
        result["error"] = "N/A - No price data"
        return result

    if growth_rate is None:
        growth_rate = _compute_earnings_growth(ticker_data)
        if growth_rate is None:
            growth_rate = 0.05

    # Clamp growth rate (same as valuation_engine)
    growth_rate = max(-0.05, min(0.25, growth_rate))

    # ── Adjustment 2: Earnings base for rate-exposed companies ──────

    # Get debt and cash data
    total_debt = _get_financial_value(
        balance_sheet, ["Total Debt", "Long Term Debt"]
    )
    cash = _get_financial_value(
        balance_sheet,
        ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
         "Cash Financial", "Cash And Short Term Investments"],
    )
    total_debt = total_debt or 0
    cash = cash or 0
    net_cash_position = cash - total_debt

    result["total_debt"] = total_debt
    result["net_cash_position"] = net_cash_position

    # Get current interest expense
    current_interest = _get_financial_value(
        financials,
        ["Interest Expense", "Interest Expense Non Operating"],
    )
    current_interest = abs(current_interest) if current_interest else 0
    result["current_interest_expense"] = current_interest

    # Estimate tax rate
    pretax = _get_financial_value(
        financials,
        ["Pretax Income", "Income Before Tax", "EBIT"],
    )
    net_income = _get_financial_value(
        financials,
        ["Net Income", "Net Income Common Stockholders"],
    )
    if pretax and net_income and pretax > 0 and net_income > 0:
        tax_rate = 1 - (net_income / pretax)
        tax_rate = max(0.0, min(0.50, tax_rate))  # sanity clamp
    else:
        tax_rate = 0.21  # statutory default

    shares = info.get("sharesOutstanding", None)
    if not shares or shares <= 0:
        result["error"] = "N/A - No shares outstanding"
        return result

    stress_base_eps = base_eps

    if net_cash_position < 0:
        # Net debt company — higher rates hurt
        stress_interest = total_debt * stress_rate
        incremental_interest = stress_interest - current_interest
        result["incremental_interest"] = incremental_interest

        after_tax_hit = incremental_interest * (1 - tax_rate)
        eps_reduction = after_tax_hit / shares
        stress_base_eps = base_eps - eps_reduction

        if stress_base_eps <= 0:
            result["stress_base_eps"] = round(stress_base_eps, 4)
            result["stress_status"] = "Earnings_Eliminated"
            result["earnings_impact_pct"] = -100.0
            return result

        if stress_base_eps < base_eps * 0.5:
            result["stress_status"] = "Severely_Impaired"
        elif stress_base_eps < base_eps * 0.9:
            result["stress_status"] = "Moderately_Impaired"
        else:
            result["stress_status"] = "Resilient"

    else:
        # Net cash company — higher rates help
        interest_income = net_cash_position * stress_rate
        after_tax_income = interest_income * (1 - tax_rate)
        stress_base_eps = base_eps + (after_tax_income / shares)
        result["stress_status"] = "Rate_Beneficiary"

    result["stress_base_eps"] = round(stress_base_eps, 4)
    result["earnings_impact_pct"] = round(
        ((stress_base_eps - base_eps) / base_eps) * 100, 2
    )

    # ── Run full DCF with both adjustments ──────────────────────────
    stress_iv = _run_two_stage_dcf(stress_base_eps, growth_rate, stress_rate)
    stress_mos = round(
        ((stress_iv - current_price) / stress_iv) * 100, 2
    ) if stress_iv and stress_iv > 0 else None

    result["stress_intrinsic_value"] = stress_iv
    result["stress_margin_of_safety"] = stress_mos

    return result


# ── Three-scenario runner ───────────────────────────────────────────────────

def calculate_all_stress_scenarios(
    ticker: str,
    ticker_data: dict,
    base_valuation: dict,
    stress_rate_override: float | None = None,
) -> dict:
    """Run calculate_stress_dcf for all three scenarios.

    If stress_rate_override is provided, it replaces the sustained_high rate.

    Returns full result dict with all scenarios, verdict, and summary fields.
    """
    current_price = base_valuation.get("current_price")
    base_iv = base_valuation.get("intrinsic_value")
    base_mos = base_valuation.get("margin_of_safety")

    scenarios_out = {}
    for key, scenario in SCENARIOS.items():
        rate = scenario["rate"]
        # Allow the user's slider to override the sustained_high scenario
        if key == "sustained_high" and stress_rate_override is not None:
            rate = stress_rate_override

        dcf = calculate_stress_dcf(ticker, ticker_data, base_valuation, rate)

        iv_change = None
        if base_iv and base_iv > 0 and dcf.get("stress_intrinsic_value"):
            iv_change = round(
                ((dcf["stress_intrinsic_value"] - base_iv) / base_iv) * 100, 2
            )

        scenarios_out[key] = {
            "rate": rate,
            "label": scenario["label"],
            "description": scenario["description"],
            "stress_base_eps": dcf.get("stress_base_eps"),
            "stress_intrinsic_value": dcf.get("stress_intrinsic_value"),
            "stress_margin_of_safety": dcf.get("stress_margin_of_safety"),
            "stress_status": dcf.get("stress_status"),
            "earnings_impact_pct": dcf.get("earnings_impact_pct"),
            "iv_change_from_base_pct": iv_change,
            "incremental_interest": dcf.get("incremental_interest"),
            "net_cash_position": dcf.get("net_cash_position"),
            "total_debt": dcf.get("total_debt"),
            "current_interest_expense": dcf.get("current_interest_expense"),
            "error": dcf.get("error"),
        }

    # ── Stress verdict (based on sustained_high scenario) ───────────
    sh = scenarios_out.get("sustained_high", {})
    sh_status = sh.get("stress_status")
    sh_mos = sh.get("stress_margin_of_safety")
    sh_iv = sh.get("stress_intrinsic_value")

    if sh_status in ("Earnings_Eliminated", "Severely_Impaired"):
        stress_verdict = "Earnings Impaired at Stress Rates"
    elif sh_status == "Rate_Beneficiary" and sh_iv and base_iv and sh_iv > base_iv:
        stress_verdict = "Rate Beneficiary — Enhanced Value"
    elif sh_mos is not None and sh_mos > 20:
        stress_verdict = "Undervalued at Stress Rates"
    elif sh_mos is not None and sh_mos >= 0:
        stress_verdict = "Fairly Valued at Stress Rates"
    elif sh_mos is not None:
        stress_verdict = "Overvalued at Stress Rates"
    else:
        stress_verdict = "Insufficient Data"

    # ── Rate risk rating (simple classification) ────────────────────
    if sh_status == "Rate_Beneficiary":
        rate_risk_rating = "Rate Beneficiary"
    elif sh_status == "Resilient":
        rate_risk_rating = "Resilient"
    elif sh_status == "Moderately_Impaired":
        rate_risk_rating = "Sensitive"
    elif sh_status in ("Severely_Impaired", "Earnings_Eliminated"):
        rate_risk_rating = "Vulnerable"
    else:
        rate_risk_rating = "Unknown"

    return {
        "ticker": ticker,
        "current_price": current_price,
        "base_intrinsic_value": base_iv,
        "base_margin_of_safety": base_mos,
        "scenarios": scenarios_out,
        "rate_risk_rating": rate_risk_rating,
        "stress_verdict": stress_verdict,
    }


# ── Batch runner ────────────────────────────────────────────────────────────

# ── XBRL-enhanced debt trajectory ──────────────────────────────────────────

def calculate_debt_trajectory(
    ticker: str,
    xbrl_result: dict,
) -> dict:
    """Extract historical debt and D/E trajectory from XBRL data.

    Returns dict with:
        available: bool
        debt_series: dict (year -> value)
        de_series: dict (year -> value)
        low_rate_era_flag: bool
        debt_growth_rate: float or None
        avg_interest_coverage_10yr: float or None
        peak_de: float or None
        peak_de_year: int or None
        interest_coverage_series: dict (year -> value)
    """
    result = {
        "available": False,
        "debt_series": {},
        "de_series": {},
        "low_rate_era_flag": False,
        "debt_growth_rate": None,
        "avg_interest_coverage_10yr": None,
        "peak_de": None,
        "peak_de_year": None,
        "interest_coverage_series": {},
    }

    if not xbrl_result or not xbrl_result.get("available"):
        return result

    annual = xbrl_result["annual_data"]

    # ── Debt series ──────────────────────────────────────────────
    if "total_debt" in annual.columns:
        debt = annual["total_debt"].dropna()
        if len(debt) > 0:
            result["debt_series"] = {int(y): float(v) for y, v in debt.items()}

    # ── D/E series ───────────────────────────────────────────────
    if "debt_to_equity" in annual.columns:
        de = annual["debt_to_equity"].dropna()
        if len(de) > 0:
            result["de_series"] = {int(y): float(v) for y, v in de.items()}
            result["peak_de"] = round(float(de.max()), 3)
            result["peak_de_year"] = int(de.idxmax())
    elif "total_debt" in annual.columns and "stockholders_equity" in annual.columns:
        debt = annual["total_debt"].dropna()
        equity = annual["stockholders_equity"].dropna()
        common = debt.index.intersection(equity.index)
        if len(common) > 0:
            import numpy as _np
            de = (debt[common] / equity[common].replace(0, _np.nan)).dropna()
            result["de_series"] = {int(y): float(v) for y, v in de.items()}
            if len(de) > 0:
                result["peak_de"] = round(float(de.max()), 3)
                result["peak_de_year"] = int(de.idxmax())

    # ── Low-rate era flag ────────────────────────────────────────
    de = result["de_series"]
    if 2019 in de and 2021 in de:
        if de[2019] > 0:
            jump = (de[2021] - de[2019]) / de[2019]
            if jump > 0.20:
                result["low_rate_era_flag"] = True

    # ── Debt growth rate ─────────────────────────────────────────
    debt_vals = result["debt_series"]
    if len(debt_vals) >= 3:
        years_sorted = sorted(debt_vals.keys())
        oldest = debt_vals[years_sorted[0]]
        newest = debt_vals[years_sorted[-1]]
        n = years_sorted[-1] - years_sorted[0]
        if oldest > 0 and newest > 0 and n > 0:
            result["debt_growth_rate"] = round(
                (newest / oldest) ** (1 / n) - 1, 4
            )

    # ── Interest coverage (10yr avg) ─────────────────────────────
    if "interest_coverage" in annual.columns:
        ic = annual["interest_coverage"].dropna()
        if len(ic) > 0:
            result["interest_coverage_series"] = {
                int(y): round(float(v), 2) for y, v in ic.items()
            }
            # Last 10 years
            recent = ic[ic.index >= (ic.index.max() - 10)]
            if len(recent) > 0:
                result["avg_interest_coverage_10yr"] = round(float(recent.median()), 2)

    result["available"] = bool(result["debt_series"] or result["de_series"])
    return result


def calculate_stress_dcf_xbrl(
    ticker: str,
    ticker_data: dict,
    base_valuation: dict,
    stress_rate: float,
    xbrl_result: dict | None = None,
) -> dict:
    """Wrapper around calculate_stress_dcf that adds debt trajectory data."""
    base_result = calculate_stress_dcf(ticker, ticker_data, base_valuation, stress_rate)

    if xbrl_result and xbrl_result.get("available"):
        trajectory = calculate_debt_trajectory(ticker, xbrl_result)
        base_result["debt_trajectory"] = trajectory
    else:
        base_result["debt_trajectory"] = {"available": False}

    return base_result


def calculate_all_stress_valuations(
    all_ticker_data: dict,
    valuations: dict,
    stress_rate_override: float | None = None,
) -> dict:
    """Run stress scenarios for every ticker. Returns dict keyed by ticker."""
    results = {}
    for ticker, data in all_ticker_data.items():
        base_val = valuations.get(ticker, {})
        results[ticker] = calculate_all_stress_scenarios(
            ticker, data, base_val, stress_rate_override
        )
    return results
