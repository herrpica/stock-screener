"""
Comprehensive Claude-powered deep dive analysis on a single stock.

Reconciles SEC narrative documents against quantitative metrics and produces
adjusted scores and a structured investment thesis through 4 sequential
Claude API calls:
  1. Business Quality Assessment
  2. Quantitative Metric Reconciliation
  3. Forward Looking Analysis
  4. Investment Thesis Synthesis
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import anthropic

from sec_fetcher import (
    get_recent_filings,
    get_filing_document,
    extract_section,
    _SECTION_PATTERNS,
)

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192

CACHE_DIR = Path(__file__).parent / ".cache" / "deep_dive"
CACHE_MAX_AGE = timedelta(days=7)

# Step labels for progress reporting
STEPS = [
    "Assembling documents",
    "Assessing business quality",
    "Reconciling quantitative metrics",
    "Analyzing forward outlook",
    "Synthesizing investment thesis",
    "Assembling final result",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _truncate_words(text: str, max_words: int) -> str:
    """Truncate text to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ── Cache ──────────────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_deep_dive.json"


def load_cached_deep_dive(ticker: str) -> dict | None:
    """Load cached deep dive result if it exists and is less than 7 days old."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data.get("analysis_timestamp", ""))
        if datetime.now() - ts > CACHE_MAX_AGE:
            return None
        return data
    except Exception:
        return None


def save_deep_dive_cache(ticker: str, result: dict):
    """Save deep dive result to JSON cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def estimate_deep_dive_cost() -> str:
    """Return human-readable estimate of API cost for one deep dive.

    Four Claude Sonnet calls with ~15-30K input tokens and ~2-4K output tokens each.
    """
    return (
        "Estimated cost: $0.05–$0.15 per stock "
        "(4 Claude Sonnet API calls, varies with document length)"
    )


def _call_claude(
    system: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Call Claude and parse JSON response."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()

        # Handle markdown code fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Claude response: {e}")
        return {"error": f"JSON parse error: {e}"}
    except anthropic.APIError as e:
        logger.warning(f"Anthropic API error: {e}")
        return {"error": f"API error: {e}"}
    except Exception as e:
        logger.warning(f"Claude call failed: {e}")
        return {"error": f"Analysis failed: {e}"}


# ── Additional section patterns for deep dive ─────────────────────────────

_EXTRA_SECTION_PATTERNS = {
    "liquidity": [
        r"liquidity\s*and\s*capital\s*resources",
        r"item\s*7[\.\s].*?liquidity",
    ],
    "accounting_estimates": [
        r"critical\s*accounting\s*(?:estimates|policies)",
        r"significant\s*accounting\s*policies",
    ],
    "revenue_recognition": [
        r"revenue\s*recognition",
        r"revenue\s*from\s*contracts",
    ],
    "off_balance_sheet": [
        r"off[\-\s]*balance[\-\s]*sheet",
    ],
    "compensation": [
        r"executive\s*compensation",
        r"compensation\s*discussion\s*and\s*analysis",
        r"summary\s*compensation\s*table",
    ],
    "ownership": [
        r"security\s*ownership",
        r"beneficial\s*ownership",
    ],
}


def _extract_extra_section(
    filing_text: str,
    section: str,
    max_chars: int = 30_000,
) -> str | None:
    """Extract section using extended pattern set."""
    # Try built-in patterns first
    result = extract_section(filing_text, section, max_chars)
    if result:
        return result

    # Try extra patterns
    patterns = _EXTRA_SECTION_PATTERNS.get(section, [])
    if not patterns:
        return None

    text_lower = filing_text.lower()
    start_pos = None
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            start_pos = match.start()
            break

    if start_pos is None:
        return None

    remaining = text_lower[start_pos + 100:]
    next_item = re.search(r"item\s*\d+[a-z]?[\.\s]", remaining)
    if next_item:
        end_pos = start_pos + 100 + next_item.start()
    else:
        end_pos = start_pos + max_chars

    extracted = filing_text[start_pos:end_pos].strip()
    return extracted[:max_chars]


# ── Document Assembly ─────────────────────────────────────────────────────

def fetch_all_documents(ticker: str, sec_data: dict | None = None) -> dict:
    """Assemble all available narrative documents for deep dive analysis.

    Parameters
    ----------
    ticker : str
        Stock ticker.
    sec_data : dict, optional
        Pre-fetched SEC data from sec_fetcher.get_full_sec_analysis().
        If None, fetches filings fresh.

    Returns
    -------
    dict with keys: annual_report_text, quarterly_update_text,
    earnings_call_text, proxy_text, prior_year_mda, documents_found
    """
    docs = {
        "annual_report_text": None,
        "quarterly_update_text": None,
        "earnings_call_text": None,
        "proxy_text": None,
        "prior_year_mda": None,
        "documents_found": [],
    }

    # Get all filings
    filings = get_recent_filings(ticker, count=30)
    if not filings:
        return docs

    ten_ks = [f for f in filings if f["filing_type"] == "10-K"]
    ten_qs = [f for f in filings if f["filing_type"] == "10-Q"]
    eight_ks = [f for f in filings if f["filing_type"] == "8-K"]
    proxies = [f for f in filings if f["filing_type"] == "DEF 14A"]

    # ── Annual report (10-K) ──────────────────────────────────────
    if ten_ks:
        text = get_filing_document(ten_ks[0], max_chars=200_000)
        if text:
            sections = []

            # Extract in priority order
            for section_name in ["business", "risk_factors", "mda",
                                 "liquidity", "accounting_estimates"]:
                extracted = _extract_extra_section(text, section_name)
                if extracted:
                    sections.append(f"=== {section_name.upper().replace('_', ' ')} ===\n{extracted}")

            # Revenue recognition and off-balance-sheet footnotes
            for fn_section in ["revenue_recognition", "off_balance_sheet"]:
                extracted = _extract_extra_section(text, fn_section, max_chars=5000)
                if extracted:
                    sections.append(f"=== {fn_section.upper().replace('_', ' ')} ===\n{extracted}")

            if sections:
                combined = "\n\n".join(sections)
                # Truncate to 15000 words, prioritizing MD&A and Risk Factors
                docs["annual_report_text"] = _truncate_words(combined, 15000)
                docs["documents_found"].append("10-K")
            elif text:
                # Fall back to raw truncated text
                docs["annual_report_text"] = _truncate_words(text, 15000)
                docs["documents_found"].append("10-K (raw)")

    # ── Prior year 10-K MD&A ──────────────────────────────────────
    if len(ten_ks) > 1:
        prior_text = get_filing_document(ten_ks[1], max_chars=100_000)
        if prior_text:
            mda = _extract_extra_section(prior_text, "mda")
            if mda:
                docs["prior_year_mda"] = _truncate_words(mda, 4000)
                docs["documents_found"].append("Prior 10-K MD&A")

    # ── Quarterly update (10-Q) ───────────────────────────────────
    if ten_qs:
        q_text = get_filing_document(ten_qs[0], max_chars=100_000)
        if q_text:
            mda = _extract_extra_section(q_text, "mda")
            if mda:
                docs["quarterly_update_text"] = _truncate_words(mda, 4000)
                docs["documents_found"].append("10-Q MD&A")

    # ── Earnings call from 8-K ────────────────────────────────────
    for f in eight_ks[:10]:
        ek_text = get_filing_document(f, max_chars=50_000)
        if ek_text:
            text_lower = ek_text.lower()
            if "earnings call" in text_lower or "conference call" in text_lower:
                # Try to find Q&A section
                qa_start = None
                for qa_pattern in [r"question[\-\s]*and[\-\s]*answer",
                                   r"q\s*&\s*a\s*session", r"q&a"]:
                    match = re.search(qa_pattern, text_lower)
                    if match:
                        qa_start = match.start()
                        break

                if qa_start is not None:
                    # Prioritize Q&A, append prepared remarks if space
                    qa_text = ek_text[qa_start:]
                    prepared = ek_text[:qa_start]
                    qa_truncated = _truncate_words(qa_text, 3500)
                    prepared_truncated = _truncate_words(prepared, 1500)
                    docs["earnings_call_text"] = (
                        f"=== PREPARED REMARKS (excerpt) ===\n{prepared_truncated}\n\n"
                        f"=== Q&A SESSION ===\n{qa_truncated}"
                    )
                else:
                    docs["earnings_call_text"] = _truncate_words(ek_text, 5000)

                docs["documents_found"].append("Earnings Call (8-K)")
                break

    # ── Proxy statement (DEF 14A) ─────────────────────────────────
    if proxies:
        proxy_text = get_filing_document(proxies[0], max_chars=100_000)
        if proxy_text:
            comp_section = _extract_extra_section(proxy_text, "compensation", max_chars=5000)
            own_section = _extract_extra_section(proxy_text, "ownership", max_chars=3000)

            parts = []
            if comp_section:
                parts.append(f"=== COMPENSATION ===\n{comp_section}")
            if own_section:
                parts.append(f"=== EXECUTIVE OWNERSHIP ===\n{own_section}")

            if parts:
                docs["proxy_text"] = _truncate_words("\n\n".join(parts), 3000)
            else:
                docs["proxy_text"] = _truncate_words(proxy_text, 3000)
            docs["documents_found"].append("DEF 14A")

    return docs


# ── Call 1: Business Quality Assessment ───────────────────────────────────

_CALL1_SYSTEM = """You are a senior equity analyst at a top-tier investment fund.
You have been asked to assess the fundamental quality of this business based
solely on its public filings. Be direct, skeptical, and specific. Flag anything
that concerns you.

Return ONLY valid JSON (no markdown, no code fences) in the exact structure specified."""


_CALL1_SCHEMA = """{
  "business_model_clarity": 0-100,
  "competitive_moat_assessment": {
    "moat_exists": true|false,
    "moat_type": "cost_advantage|switching_costs|network_effect|brand|scale|none",
    "moat_durability": "wide|narrow|none",
    "evidence": "specific evidence from filings"
  },
  "revenue_quality": {
    "score": 0-100,
    "recurring_revenue_pct_estimate": float or null,
    "revenue_concentration_risk": "high|medium|low",
    "notes": "explanation"
  },
  "earnings_quality": {
    "score": 0-100,
    "concerns": ["list of specific accounting or structural concerns"],
    "one_time_items_detected": true|false,
    "one_time_item_details": "description if applicable",
    "gaap_vs_reality_gap": "significant|moderate|minimal"
  },
  "management_strategy_clarity": 0-100,
  "capital_allocation_quality": {
    "score": 0-100,
    "primary_use_of_capital": "growth_investment|acquisitions|buybacks|dividends|debt_reduction",
    "assessment": "explanation"
  },
  "ai_and_technology_position": {
    "ai_as_opportunity": true|false,
    "ai_as_threat": true|false,
    "company_ai_initiatives": ["specific initiatives mentioned"],
    "disruption_risk_level": "high|medium|low|none",
    "disruption_timeline": "immediate|2-3 years|5+ years|not_applicable"
  },
  "key_business_risks": [
    {"risk": "description", "severity": "high|medium|low", "managements_response": "summary"}
  ],
  "hidden_strengths": ["things the numbers alone dont show"],
  "hidden_weaknesses": ["things the numbers alone dont show"],
  "overall_business_quality_score": 0-100
}"""


def _run_call_1(
    ticker: str,
    company_name: str,
    sector: str,
    documents: dict,
    api_key: str,
) -> dict:
    """Call 1 — Business Quality Assessment."""
    annual = documents.get("annual_report_text") or "(Not available)"
    quarterly = documents.get("quarterly_update_text") or "(Not available)"

    user_prompt = (
        f"Company: {company_name} ({ticker})\n"
        f"Sector: {sector}\n\n"
        f"=== ANNUAL REPORT (10-K) ===\n{annual}\n\n"
        f"=== QUARTERLY UPDATE (10-Q) ===\n{quarterly}\n\n"
        f"Analyze this business and return ONLY valid JSON in this structure:\n"
        f"{_CALL1_SCHEMA}"
    )

    return _call_claude(_CALL1_SYSTEM, user_prompt, api_key)


# ── Call 2: Quantitative Metric Reconciliation ────────────────────────────

_CALL2_SYSTEM = """You are a forensic financial analyst. You have been given both the
quantitative metrics calculated from a company's reported financials AND the
narrative content of their SEC filings. Your job is to identify where the numbers
are misleading — in either direction — and recommend specific adjustments.

Return ONLY valid JSON (no markdown, no code fences) in the exact structure specified."""


_CALL2_SCHEMA = """{
  "score_adjustments": {
    "quality_score": {
      "adjustment": integer between -20 and +20,
      "reasoning": "specific reason for adjustment"
    },
    "growth_score": {
      "adjustment": integer between -20 and +20,
      "reasoning": "specific reason"
    },
    "value_score": {
      "adjustment": integer between -20 and +20,
      "reasoning": "specific reason"
    },
    "sentiment_score": {
      "adjustment": integer between -10 and +10,
      "reasoning": "specific reason"
    }
  },
  "intrinsic_value_adjustments": {
    "growth_rate_adjustment": float between -0.10 and +0.10,
    "reasoning": "why the observed growth rate overstates or understates future earnings power",
    "earnings_base_adjustment_pct": float between -0.30 and +0.30,
    "earnings_base_reasoning": "if base EPS is distorted by one-time items, suggest adjustment",
    "discount_rate_adjustment": float between -0.02 and +0.02,
    "discount_rate_reasoning": "if business risk is meaningfully higher or lower than assumed"
  },
  "quality_rating_adjustment": {
    "suggested_rating": "Above Bar|At Bar|Below Bar|No Change",
    "reasoning": "specific reason if different from quantitative rating"
  },
  "management_credibility": {
    "score": 0-100,
    "prior_year_goals": ["what management said they would do"],
    "actually_delivered": ["what actually happened based on current filing"],
    "credibility_assessment": "delivers_consistently|mixed_track_record|frequently_disappoints"
  },
  "red_flags": [
    {"flag": "description", "severity": "critical|significant|minor"}
  ],
  "positive_revisions": [
    {"item": "description", "impact": "explanation"}
  ]
}"""


def _run_call_2(
    ticker: str,
    company_name: str,
    current_scores: dict,
    current_valuation: dict,
    current_quality_rating: dict,
    documents: dict,
    api_key: str,
) -> dict:
    """Call 2 — Quantitative Metric Reconciliation."""
    prior_mda = documents.get("prior_year_mda") or "(Not available)"
    annual = documents.get("annual_report_text") or "(Not available)"
    earnings = documents.get("earnings_call_text") or "(Not available)"

    # Extract MD&A from annual report if separate
    mda_section = "(See annual report above)"
    if annual and annual != "(Not available)":
        # Try to pull just MD&A
        mda_match = re.search(r"=== MDA ===\n(.*?)(?:===|$)", annual, re.DOTALL)
        if mda_match:
            mda_section = mda_match.group(1).strip()[:8000]

    user_prompt = (
        f"Company: {company_name} ({ticker})\n\n"
        f"=== CURRENT QUANTITATIVE SCORES ===\n"
        f"Quality Score: {current_scores.get('quality_score', 'N/A')}\n"
        f"Growth Score: {current_scores.get('growth_score', 'N/A')}\n"
        f"Value Score: {current_scores.get('value_score', 'N/A')}\n"
        f"Sentiment Score: {current_scores.get('sentiment_score', 'N/A')}\n"
        f"Composite Score: {current_scores.get('composite_score', 'N/A')}\n\n"
        f"=== CURRENT VALUATION ===\n"
        f"Base EPS: ${current_valuation.get('base_eps', 'N/A')}\n"
        f"Growth Rate Used: {current_valuation.get('growth_rate_used', 'N/A')}\n"
        f"Intrinsic Value: ${current_valuation.get('intrinsic_value', 'N/A')}\n"
        f"Current Price: ${current_valuation.get('current_price', 'N/A')}\n"
        f"Margin of Safety: {current_valuation.get('margin_of_safety', 'N/A')}%\n\n"
        f"=== CURRENT QUALITY RATING ===\n"
        f"Rating: {current_quality_rating.get('quality_rating', 'N/A')}\n"
        f"Earnings Consistency: {current_quality_rating.get('earnings_consistency_score', 'N/A')}\n"
        f"Debt Discipline: {current_quality_rating.get('debt_discipline_score', 'N/A')}\n"
        f"Dividend Quality: {current_quality_rating.get('dividend_quality_score', 'N/A')}\n"
        f"Buyback Score: {current_quality_rating.get('buyback_score', 'N/A')}\n\n"
        f"=== PRIOR YEAR MD&A ===\n{prior_mda}\n\n"
        f"=== CURRENT YEAR MD&A ===\n{mda_section}\n\n"
        f"=== EARNINGS CALL ===\n{earnings}\n\n"
        f"Reconcile the quantitative metrics against the narrative disclosures. "
        f"Return ONLY valid JSON in this structure:\n{_CALL2_SCHEMA}"
    )

    return _call_claude(_CALL2_SYSTEM, user_prompt, api_key)


# ── Call 3: Forward Looking Analysis ──────────────────────────────────────

_CALL3_SYSTEM = """You are a growth equity analyst focused on where a business will be
in 3-5 years. Based on the company's own disclosures, assess the trajectory of
this business and identify catalysts and risks that backward-looking metrics miss.

Return ONLY valid JSON (no markdown, no code fences) in the exact structure specified."""


_CALL3_SCHEMA = """{
  "growth_catalysts": [
    {
      "catalyst": "description",
      "timeline": "0-6 months|6-18 months|2-3 years|3-5 years",
      "magnitude": "high|medium|low",
      "management_confidence_level": "explicit|implied|speculative"
    }
  ],
  "growth_headwinds": [
    {
      "headwind": "description",
      "timeline": "0-6 months|6-18 months|2-3 years|3-5 years",
      "magnitude": "high|medium|low"
    }
  ],
  "margin_trajectory": {
    "direction": "expanding|stable|contracting",
    "reasoning": "specific evidence from filings",
    "confidence": "high|medium|low"
  },
  "revenue_trajectory": {
    "direction": "accelerating|stable|decelerating",
    "reasoning": "specific evidence",
    "confidence": "high|medium|low"
  },
  "competitive_position_trend": "strengthening|stable|weakening",
  "competitive_position_evidence": "specific evidence from filings",
  "industry_tailwinds": ["macro trends benefiting this company"],
  "industry_headwinds": ["macro trends hurting this company"],
  "three_year_earnings_power_estimate": {
    "base_case_growth_rate": 0.0,
    "bull_case_growth_rate": 0.0,
    "bear_case_growth_rate": 0.0,
    "key_assumptions": ["list of critical assumptions"]
  }
}"""


def _run_call_3(
    ticker: str,
    company_name: str,
    sector: str,
    call1_result: dict,
    documents: dict,
    api_key: str,
) -> dict:
    """Call 3 — Forward Looking Analysis."""
    annual = documents.get("annual_report_text") or "(Not available)"
    earnings = documents.get("earnings_call_text") or "(Not available)"

    # Summarize call 1 for context
    biz_quality = call1_result.get("overall_business_quality_score", "N/A")
    moat = call1_result.get("competitive_moat_assessment", {})

    user_prompt = (
        f"Company: {company_name} ({ticker})\n"
        f"Sector: {sector}\n\n"
        f"=== PRIOR BUSINESS QUALITY ASSESSMENT ===\n"
        f"Overall Quality Score: {biz_quality}/100\n"
        f"Moat: {moat.get('moat_durability', 'N/A')} ({moat.get('moat_type', 'N/A')})\n"
        f"Revenue Quality: {call1_result.get('revenue_quality', {}).get('score', 'N/A')}/100\n"
        f"Earnings Quality: {call1_result.get('earnings_quality', {}).get('score', 'N/A')}/100\n"
        f"AI Position: Opportunity={call1_result.get('ai_and_technology_position', {}).get('ai_as_opportunity', 'N/A')}, "
        f"Threat={call1_result.get('ai_and_technology_position', {}).get('ai_as_threat', 'N/A')}\n\n"
        f"Hidden Strengths: {call1_result.get('hidden_strengths', [])}\n"
        f"Hidden Weaknesses: {call1_result.get('hidden_weaknesses', [])}\n\n"
        f"=== MD&A / ANNUAL REPORT ===\n{annual[:12000]}\n\n"
        f"=== EARNINGS CALL ===\n{earnings}\n\n"
        f"Analyze the forward trajectory. Return ONLY valid JSON:\n{_CALL3_SCHEMA}"
    )

    return _call_claude(_CALL3_SYSTEM, user_prompt, api_key)


# ── Call 4: Investment Thesis Synthesis ────────────────────────────────────

_CALL4_SYSTEM = """You are a portfolio manager writing a formal investment memo.
Synthesize all prior analysis into a structured investment thesis. Be direct
about conviction level. Write as if recommending to an investment committee.

Return ONLY valid JSON (no markdown, no code fences) in the exact structure specified."""


_CALL4_SCHEMA = """{
  "recommendation": "Strong Buy|Buy|Hold|Avoid|Strong Avoid",
  "conviction_level": "high|medium|low",
  "one_line_thesis": "single sentence investment thesis",
  "bull_case": {
    "narrative": "2-3 paragraph bull case",
    "key_assumptions": ["list of things that must be true"],
    "upside_to_intrinsic_value_pct": 0.0
  },
  "base_case": {
    "narrative": "2-3 paragraph base case",
    "key_assumptions": ["list of things that must be true"],
    "expected_return_12_month_pct": 0.0
  },
  "bear_case": {
    "narrative": "2-3 paragraph bear case",
    "key_risks": ["list of things that could go wrong"],
    "downside_pct": 0.0
  },
  "what_would_change_thesis": [
    "specific things to watch that would cause re-evaluation"
  ],
  "ideal_entry_price": 0.0,
  "ideal_entry_reasoning": "why this price represents good risk/reward",
  "position_sizing_suggestion": "core|standard|speculative|avoid",
  "time_to_thesis_realization": "6 months|1 year|2-3 years|3-5 years",
  "memo_summary": "Full 4-6 paragraph investment memo in professional analyst style"
}"""


def _run_call_4(
    ticker: str,
    company_name: str,
    call1_result: dict,
    call2_result: dict,
    call3_result: dict,
    current_price: float | None,
    adjusted_iv: float | None,
    adjusted_mos: float | None,
    api_key: str,
) -> dict:
    """Call 4 — Investment Thesis Synthesis."""
    # Build concise summaries of prior calls
    price_str = f"${current_price:.2f}" if current_price else "N/A"
    iv_str = f"${adjusted_iv:.2f}" if adjusted_iv else "N/A"
    mos_str = f"{adjusted_mos:.1f}%" if adjusted_mos is not None else "N/A"

    user_prompt = (
        f"Company: {company_name} ({ticker})\n"
        f"Current Price: {price_str}\n"
        f"Adjusted Intrinsic Value: {iv_str}\n"
        f"Adjusted Margin of Safety: {mos_str}\n\n"
    )

    # Call 1 summary
    user_prompt += (
        f"=== BUSINESS QUALITY ASSESSMENT ===\n"
        f"{json.dumps(call1_result, indent=2, default=str)[:4000]}\n\n"
    )

    # Call 2 summary
    user_prompt += (
        f"=== METRIC RECONCILIATION ===\n"
        f"{json.dumps(call2_result, indent=2, default=str)[:3000]}\n\n"
    )

    # Call 3 summary
    user_prompt += (
        f"=== FORWARD ANALYSIS ===\n"
        f"{json.dumps(call3_result, indent=2, default=str)[:3000]}\n\n"
    )

    user_prompt += (
        f"Synthesize everything into an investment thesis. "
        f"Return ONLY valid JSON:\n{_CALL4_SCHEMA}"
    )

    return _call_claude(_CALL4_SYSTEM, user_prompt, api_key)


# ── DCF Recalculation ─────────────────────────────────────────────────────

def _recalculate_dcf(
    base_eps: float,
    growth_rate: float,
    discount_rate: float,
    current_price: float,
) -> dict:
    """Recalculate two-stage DCF with adjusted parameters.

    Same model as valuation_engine but with arbitrary inputs.
    """
    # Cap/floor growth rate
    growth_rate = max(-0.05, min(0.25, growth_rate))

    # Stage 1: years 1-5 at growth rate
    eps = base_eps
    total_pv = 0
    for year in range(1, 6):
        eps *= (1 + growth_rate)
        total_pv += eps / (1 + discount_rate) ** year

    # Stage 2: years 6-10 at min(growth, 8%)
    stage2_rate = min(growth_rate, 0.08)
    for year in range(6, 11):
        eps *= (1 + stage2_rate)
        total_pv += eps / (1 + discount_rate) ** year

    # Terminal value: Year 10 EPS * 15
    terminal_pv = (eps * 15) / (1 + discount_rate) ** 10
    total_pv += terminal_pv

    intrinsic_value = round(total_pv, 2)
    margin_of_safety = round(
        (intrinsic_value - current_price) / intrinsic_value * 100, 2
    ) if intrinsic_value > 0 else None

    return {
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety,
    }


# ── Result Assembly ───────────────────────────────────────────────────────

def assemble_deep_dive_result(
    ticker: str,
    call1_result: dict,
    call2_result: dict,
    call3_result: dict,
    call4_result: dict,
    original_scores: dict,
    original_valuation: dict,
    original_quality_rating: str,
    documents_found: list[str],
    discount_rate: float = 0.10,
) -> dict:
    """Apply all adjustments and assemble final deep dive result.

    Parameters
    ----------
    original_scores : dict
        Must have: quality_score, growth_score, value_score, sentiment_score,
        composite_score
    original_valuation : dict
        Must have: intrinsic_value, margin_of_safety, base_eps,
        growth_rate_used, current_price
    original_quality_rating : str
        Current quality rating (Above Bar / At Bar / Below Bar).
    discount_rate : float
        Current discount rate from user settings.
    """
    # ── Apply score adjustments ───────────────────────────────────
    adj = call2_result.get("score_adjustments", {})

    orig_q = original_scores.get("quality_score", 50)
    orig_g = original_scores.get("growth_score", 50)
    orig_v = original_scores.get("value_score", 50)
    orig_s = original_scores.get("sentiment_score", 50)

    q_adj = adj.get("quality_score", {}).get("adjustment", 0)
    g_adj = adj.get("growth_score", {}).get("adjustment", 0)
    v_adj = adj.get("value_score", {}).get("adjustment", 0)
    s_adj = adj.get("sentiment_score", {}).get("adjustment", 0)

    # Clamp adjustments to specified ranges
    q_adj = max(-20, min(20, q_adj))
    g_adj = max(-20, min(20, g_adj))
    v_adj = max(-20, min(20, v_adj))
    s_adj = max(-10, min(10, s_adj))

    adj_q = min(100, max(0, orig_q + q_adj))
    adj_g = min(100, max(0, orig_g + g_adj))
    adj_v = min(100, max(0, orig_v + v_adj))
    adj_s = min(100, max(0, orig_s + s_adj))

    # Recompute composite with default weights
    adj_composite = round(adj_q * 0.35 + adj_g * 0.30 + adj_v * 0.25 + adj_s * 0.10, 1)

    # ── Apply valuation adjustments ───────────────────────────────
    iv_adj = call2_result.get("intrinsic_value_adjustments", {})

    orig_eps = original_valuation.get("base_eps") or 0
    orig_growth = original_valuation.get("growth_rate_used") or 0.05
    orig_price = original_valuation.get("current_price") or 0
    orig_iv = original_valuation.get("intrinsic_value") or 0
    orig_mos = original_valuation.get("margin_of_safety") or 0

    eps_adj_pct = iv_adj.get("earnings_base_adjustment_pct", 0)
    eps_adj_pct = max(-0.30, min(0.30, eps_adj_pct))

    growth_adj = iv_adj.get("growth_rate_adjustment", 0)
    growth_adj = max(-0.10, min(0.10, growth_adj))

    dr_adj = iv_adj.get("discount_rate_adjustment", 0)
    dr_adj = max(-0.02, min(0.02, dr_adj))

    adj_eps = orig_eps * (1 + eps_adj_pct)
    adj_growth = orig_growth + growth_adj
    adj_dr = discount_rate + dr_adj

    # Recalculate DCF
    if adj_eps > 0 and orig_price > 0:
        dcf = _recalculate_dcf(adj_eps, adj_growth, adj_dr, orig_price)
        adj_iv = dcf["intrinsic_value"]
        adj_mos = dcf["margin_of_safety"]
    else:
        adj_iv = orig_iv
        adj_mos = orig_mos

    # Determine adjusted quality rating
    qr_adj = call2_result.get("quality_rating_adjustment", {})
    suggested = qr_adj.get("suggested_rating", "No Change")
    adjusted_quality_rating = (
        suggested if suggested != "No Change" else original_quality_rating
    )

    return {
        "ticker": ticker,
        "analysis_timestamp": datetime.now().isoformat(),
        "documents_analyzed": documents_found,
        "business_assessment": call1_result,
        "metric_reconciliation": call2_result,
        "forward_analysis": call3_result,
        "investment_thesis": call4_result,
        "adjusted_scores": {
            "quality_score": round(adj_q, 1),
            "growth_score": round(adj_g, 1),
            "value_score": round(adj_v, 1),
            "sentiment_score": round(adj_s, 1),
            "composite_score": adj_composite,
            "quality_score_delta": round(q_adj, 1),
            "growth_score_delta": round(g_adj, 1),
            "value_score_delta": round(v_adj, 1),
            "sentiment_score_delta": round(s_adj, 1),
        },
        "adjusted_valuation": {
            "intrinsic_value": adj_iv,
            "margin_of_safety": adj_mos,
            "growth_rate_used": round(adj_growth, 4),
            "base_eps_used": round(adj_eps, 2),
            "discount_rate_used": round(adj_dr, 4),
            "intrinsic_value_delta": round(adj_iv - orig_iv, 2) if adj_iv and orig_iv else 0,
            "margin_of_safety_delta": round(adj_mos - orig_mos, 2) if adj_mos and orig_mos else 0,
        },
        "adjusted_quality_rating": adjusted_quality_rating,
        "original_quality_rating": original_quality_rating,
        "original_scores": original_scores,
        "original_valuation": {
            "intrinsic_value": orig_iv,
            "margin_of_safety": orig_mos,
            "growth_rate_used": orig_growth,
            "base_eps": orig_eps,
            "discount_rate": discount_rate,
        },
    }


# ── Main Orchestrator ─────────────────────────────────────────────────────

def run_deep_dive(
    ticker: str,
    company_name: str,
    current_scores: dict,
    current_valuation: dict,
    current_quality_rating: dict,
    sector: str,
    sector_medians: dict,
    documents: dict | None = None,
    api_key: str | None = None,
    discount_rate: float = 0.10,
    on_progress: Callable[[int, str], None] | None = None,
    force_refresh: bool = False,
) -> dict:
    """Run the full 4-call deep dive analysis pipeline.

    Parameters
    ----------
    ticker : str
    company_name : str
    current_scores : dict
        quality_score, growth_score, value_score, sentiment_score, composite_score
    current_valuation : dict
        From valuation_engine: intrinsic_value, margin_of_safety, base_eps,
        growth_rate_used, current_price
    current_quality_rating : dict
        quality_rating, earnings_consistency_score, debt_discipline_score, etc.
    sector : str
    sector_medians : dict
        Sector median values for context.
    documents : dict, optional
        Pre-fetched documents from fetch_all_documents(). If None, fetches them.
    api_key : str, optional
        Falls back to ANTHROPIC_API_KEY env var.
    discount_rate : float
        Current discount rate setting.
    on_progress : callable, optional
        Called with (step_index, step_label) for progress tracking.
        step_index is 0-based, total steps = len(STEPS).
    force_refresh : bool
        If True, bypasses cache and re-runs analysis.

    Returns
    -------
    dict with full deep dive result, or dict with "error" key on failure.
    """
    # Check cache first
    if not force_refresh:
        cached = load_cached_deep_dive(ticker)
        if cached is not None:
            return cached

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"error": "ANTHROPIC_API_KEY not set."}

    quality_rating_str = current_quality_rating.get("quality_rating", "At Bar")

    def _progress(step: int):
        if on_progress:
            on_progress(step, STEPS[step])

    # Step 0: Assemble documents
    _progress(0)
    if documents is None:
        documents = fetch_all_documents(ticker)

    if not documents.get("documents_found"):
        return {"error": f"No SEC documents found for {ticker}."}

    # Step 1: Business Quality Assessment
    _progress(1)
    call1 = _run_call_1(ticker, company_name, sector, documents, key)
    if "error" in call1:
        return {"error": f"Call 1 (Business Quality) failed: {call1['error']}"}

    # Step 2: Quantitative Metric Reconciliation
    _progress(2)
    call2 = _run_call_2(
        ticker, company_name, current_scores, current_valuation,
        current_quality_rating, documents, key,
    )
    if "error" in call2:
        return {"error": f"Call 2 (Metric Reconciliation) failed: {call2['error']}"}

    # Step 3: Forward Looking Analysis
    _progress(3)
    call3 = _run_call_3(ticker, company_name, sector, call1, documents, key)
    if "error" in call3:
        return {"error": f"Call 3 (Forward Analysis) failed: {call3['error']}"}

    # Compute adjusted IV for Call 4 context
    iv_adj = call2.get("intrinsic_value_adjustments", {})
    orig_eps = current_valuation.get("base_eps") or 0
    orig_growth = current_valuation.get("growth_rate_used") or 0.05
    orig_price = current_valuation.get("current_price") or 0

    eps_adj_pct = max(-0.30, min(0.30, iv_adj.get("earnings_base_adjustment_pct", 0)))
    growth_adj = max(-0.10, min(0.10, iv_adj.get("growth_rate_adjustment", 0)))
    dr_adj = max(-0.02, min(0.02, iv_adj.get("discount_rate_adjustment", 0)))

    adj_eps = orig_eps * (1 + eps_adj_pct)
    adj_growth = orig_growth + growth_adj
    adj_dr = discount_rate + dr_adj

    if adj_eps > 0 and orig_price > 0:
        dcf = _recalculate_dcf(adj_eps, adj_growth, adj_dr, orig_price)
        preview_iv = dcf["intrinsic_value"]
        preview_mos = dcf["margin_of_safety"]
    else:
        preview_iv = current_valuation.get("intrinsic_value")
        preview_mos = current_valuation.get("margin_of_safety")

    # Step 4: Investment Thesis Synthesis
    _progress(4)
    call4 = _run_call_4(
        ticker, company_name, call1, call2, call3,
        orig_price, preview_iv, preview_mos, key,
    )
    if "error" in call4:
        return {"error": f"Call 4 (Thesis Synthesis) failed: {call4['error']}"}

    # Step 5: Assemble final result
    _progress(5)
    result = assemble_deep_dive_result(
        ticker=ticker,
        call1_result=call1,
        call2_result=call2,
        call3_result=call3,
        call4_result=call4,
        original_scores=current_scores,
        original_valuation=current_valuation,
        original_quality_rating=quality_rating_str,
        documents_found=documents.get("documents_found", []),
        discount_rate=discount_rate,
    )

    # Cache result
    save_deep_dive_cache(ticker, result)

    return result
