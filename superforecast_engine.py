"""
Superforecasting Engine — Scenario intelligence and investment implications.

Four-phase pipeline:
  1. Conversational scenario intake → Scenario Intelligence Brief
  2. Probability tree construction (Good Judgment Project methodology)
  3. Investment opportunity identification (probability-weighted)
  4. Synthesis → executive report with monitoring indicators

All Claude API calls use claude-sonnet-4-20250514.
"""

import json
import logging
import os
import pathlib
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

_CACHE_DIR = pathlib.Path(__file__).parent / "scenario_cache"


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _ensure_cache_dir():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _parse_json_response(text: str, call_name: str = "superforecast call") -> dict:
    """Robust JSON extraction with multiple fallback strategies."""
    import re as _re
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    fence_pattern = _re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", _re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract first { ... last }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 4: array responses [ ... ]
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        try:
            arr = json.loads(text[bracket_start:bracket_end + 1])
            return {"items": arr}
        except json.JSONDecodeError:
            pass

    # Strategy 5: walk backwards to find last valid closing brace
    if brace_start >= 0:
        for i in range(len(text) - 1, brace_start, -1):
            if text[i] == "}":
                try:
                    return json.loads(text[brace_start:i + 1])
                except json.JSONDecodeError:
                    continue

    logger.warning(f"{call_name}: All JSON extraction strategies failed. Raw: {text[:300]}")
    return {"error": f"JSON parse failed for {call_name}", "raw": text[:500]}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Conversational Scenario Intake
# ═══════════════════════════════════════════════════════════════════════════════

_INTAKE_SYSTEM = """\
You are a senior geopolitical and macro analyst conducting a structured \
scenario intake interview. Your job is to gather precise information to build \
a Scenario Intelligence Brief.

Ask ONE clear question at a time. Be conversational but precise. \
Do not volunteer analysis yet — focus on understanding the user's scenario.

You are building this brief template:
- trigger_event: The specific event or development
- geographic_scope: Which regions/countries are affected
- timeline_to_onset: When might this happen (immediate / 1-3 months / 3-6 months / 6-12 months / 1-3 years)
- duration_estimate: How long would effects last
- base_probability: User's estimated probability (0.0-1.0)
- probability_reasoning: Why that probability
- key_uncertainties: At least 2 critical unknowns
- update_triggers: Events that would increase or decrease probability
- sector_hypotheses: Which GICS sectors are affected and how
- excluded_instruments: Any instruments to avoid

When you have enough information (trigger_event, geographic_scope, \
timeline_to_onset, base_probability, key_uncertainties with at least 2, \
duration_estimate), confirm the brief back to the user in plain English \
before proceeding.

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "current_brief": {<updated brief fields>},
  "next_question": "<your next question or null if complete>",
  "intake_complete": true/false,
  "brief_summary": "<plain English summary when complete, empty string otherwise>"
}
"""

_REQUIRED_INTAKE_FIELDS = [
    "trigger_event", "geographic_scope", "timeline_to_onset",
    "base_probability", "key_uncertainties", "duration_estimate",
]


def conduct_intake_conversation(
    initial_input: str,
    conversation_history: list,
    current_brief: dict,
) -> dict:
    """Manage multi-turn conversation to build a Scenario Intelligence Brief.

    Parameters
    ----------
    initial_input : str
        The user's latest message.
    conversation_history : list
        List of {"role": "user"|"assistant", "content": str} dicts.
    current_brief : dict
        Partially filled brief from previous turns.

    Returns
    -------
    dict with: current_brief, next_question, intake_complete, brief_summary
    """
    client = _get_client()

    # Build messages for Claude
    messages = []
    for msg in conversation_history:
        role = msg["role"]
        content = msg["content"]
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    # Add context about current brief state
    context = (
        f"Current brief state (update and return the full brief with any new info):\n"
        f"{json.dumps(current_brief, indent=2)}\n\n"
        f"Required fields still missing: "
        f"{[f for f in _REQUIRED_INTAKE_FIELDS if not current_brief.get(f)]}\n\n"
        f"User's latest message: {initial_input}"
    )

    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": context})
    else:
        # Append context to the last user message
        messages[-1]["content"] = context

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=_INTAKE_SYSTEM,
        messages=messages,
    )

    result = _parse_json_response(response.content[0].text, call_name="intake conversation")

    # Validate and merge brief
    new_brief = result.get("current_brief", current_brief)
    if not isinstance(new_brief, dict):
        new_brief = current_brief

    # Merge — keep existing values, add new ones
    merged = {**current_brief, **new_brief}

    # Check completeness
    missing = [f for f in _REQUIRED_INTAKE_FIELDS if not merged.get(f)]
    # key_uncertainties needs at least 2
    ku = merged.get("key_uncertainties", [])
    if isinstance(ku, list) and len(ku) < 2:
        if "key_uncertainties" not in missing:
            missing.append("key_uncertainties")

    intake_complete = result.get("intake_complete", False) and len(missing) == 0
    merged["intake_complete"] = intake_complete
    merged["missing_fields"] = missing

    # Set defaults for optional fields
    merged.setdefault("title", merged.get("trigger_event", "Untitled Scenario"))
    merged.setdefault("update_triggers", {"increase": [], "decrease": []})
    merged.setdefault("interaction_scenarios", [])
    merged.setdefault("user_conviction", "moderate")
    merged.setdefault("sector_hypotheses", {})
    merged.setdefault("excluded_instruments", [])
    merged.setdefault("special_considerations", "")
    merged.setdefault("probability_reasoning", "")

    return {
        "current_brief": merged,
        "next_question": result.get("next_question"),
        "intake_complete": intake_complete,
        "brief_summary": result.get("brief_summary", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Probability Tree Construction
# ═══════════════════════════════════════════════════════════════════════════════

_TREE_SYSTEM = """\
You are a superforecaster trained in Good Judgment Project methodology.
Build an explicit probability tree for the scenario described.

RULES:
1. Assign specific numerical probabilities — avoid round numbers (use 0.37, not 0.40).
2. Branch probabilities MUST sum to exactly 1.0.
3. Reference historical base rates and analogous events.
4. Include at least 3 branches (best case, worst case, most likely, and/or variants).
5. For each branch, specify macro impacts with min/max ranges.
6. Sector impacts should reference GICS sectors.
7. Include second-order effects that most analysts would miss.

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "scenario_title": str,
  "base_event_probability": float,
  "base_probability_reasoning": str,
  "historical_analogues": [
    {"event": str, "year": int, "similarity": str, "outcome": str, "base_rate_implication": str}
  ],
  "branches": [
    {
      "id": str (e.g. "branch_1"),
      "label": str,
      "description": str,
      "probability": float,
      "triggering_conditions": [str],
      "macro_impacts": {
        "oil_price_change_pct": [min_float, max_float],
        "us_gdp_impact_pct": [min_float, max_float],
        "dollar_index_change": [min_float, max_float],
        "us_10yr_yield_change_bps": [min_float, max_float],
        "vix_level_estimate": [min_float, max_float],
        "global_trade_impact": "severe|moderate|mild|none",
        "inflation_impact": "significant_increase|moderate_increase|stable|decrease"
      },
      "sector_impacts": {
        "sector_name": {
          "direction": "strongly_positive|positive|neutral|negative|strongly_negative",
          "magnitude_pct": [min_float, max_float],
          "reasoning": str,
          "time_to_impact": "immediate|1-3 months|3-6 months|6-12 months"
        }
      },
      "second_order_effects": [str],
      "timeline": str
    }
  ],
  "probability_sum_check": float,
  "key_swing_factors": [str],
  "model_confidence": "high|medium|low",
  "confidence_reasoning": str
}
"""


def build_probability_tree(brief: dict) -> dict:
    """Build an explicit probability tree from the scenario brief.

    Returns fully structured probability tree dict.
    """
    client = _get_client()

    prompt = (
        f"Build a probability tree for this scenario:\n\n"
        f"{json.dumps(brief, indent=2)}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_TREE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    tree = _parse_json_response(response.content[0].text, call_name="probability tree")

    # Validate and renormalize branch probabilities
    branches = tree.get("branches", [])
    if branches:
        total_prob = sum(b.get("probability", 0) for b in branches)
        if abs(total_prob - 1.0) > 0.01 and total_prob > 0:
            for b in branches:
                b["probability"] = round(b["probability"] / total_prob, 4)
        tree["probability_sum_check"] = round(
            sum(b.get("probability", 0) for b in branches), 4
        )
        tree["branches"] = branches

    return tree


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Investment Opportunity Identification
# ═══════════════════════════════════════════════════════════════════════════════

_OPPORTUNITY_SYSTEM = """\
You are a portfolio strategist analyzing investment opportunities arising \
from a specific geopolitical/macroeconomic scenario.

Your task:
1. Calculate probability-weighted expected impact across ALL branches.
2. Identify long opportunities and avoid signals.
3. Be specific about causal mechanisms — explain WHY each instrument is affected.
4. Consider both direct and indirect impacts (supply chain, sentiment, regulatory).
5. Include sector ETFs and relevant commodity instruments, not just individual stocks.

Focus on instruments from the S&P 500 universe provided plus relevant ETFs.

For each instrument, compute:
- expected_impact_pct = sum(branch_probability × branch_impact for each branch)
- variance_score = standard deviation of impacts across branches (higher = more uncertain)
- conviction_score = 0-100 based on mechanism clarity and historical precedent
- combined_score = conviction × abs(expected_impact) / (1 + variance)

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "opportunities": [sorted by combined_score descending, direction=strong_long or long],
  "risks": [sorted by combined_score descending, direction=avoid or strong_avoid],
  "hedges": [instruments resilient across multiple branches],
  "winners_summary": [str],
  "losers_summary": [str]
}

Each instrument:
{
  "ticker": str,
  "name": str,
  "instrument_type": "stock|etf|adr|commodity",
  "direction": "strong_long|long|neutral|avoid|strong_avoid",
  "expected_impact_pct": float,
  "branch_impacts": {"branch_id": {"impact_pct": float, "reasoning": str}},
  "variance_score": float,
  "conviction_score": float,
  "causal_mechanism": str,
  "mechanism_type": "direct|indirect|sentiment|supply_chain|regulatory",
  "mechanism_confidence": "established|probable|speculative",
  "time_to_impact": str,
  "position_character": "core|tactical|speculative|hedge",
  "key_risks_to_thesis": [str],
  "combined_score": float
}
"""


def identify_opportunities(
    brief: dict,
    tree: dict,
    universe_tickers: list,
    existing_screener_data: dict | None = None,
) -> dict:
    """Identify investment opportunities from probability tree.

    Parameters
    ----------
    brief : dict
        Scenario Intelligence Brief.
    tree : dict
        Probability tree from build_probability_tree().
    universe_tickers : list
        Available tickers (S&P 500 + any extras).
    existing_screener_data : dict, optional
        Existing scores/ratings keyed by ticker for enrichment.

    Returns
    -------
    dict with opportunities, risks, hedges, summaries.
    """
    client = _get_client()

    # Build ticker context (sample — don't send all 500)
    ticker_context = ""
    if universe_tickers:
        ticker_sample = universe_tickers[:100]
        ticker_context = (
            f"\n\nAvailable S&P 500 tickers (sample of {len(universe_tickers)} total):\n"
            f"{', '.join(ticker_sample)}\n"
            f"You may also suggest sector ETFs (XLE, XLF, XLK, etc.), "
            f"commodity instruments, and any other relevant S&P 500 stocks."
        )

    prompt = (
        f"Scenario Brief:\n{json.dumps(brief, indent=2)}\n\n"
        f"Probability Tree:\n{json.dumps(tree, indent=2)}"
        f"{ticker_context}\n\n"
        f"Identify the top 15 long opportunities, top 10 risk/avoid names, "
        f"and 5 all-weather hedges."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_OPPORTUNITY_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    result = _parse_json_response(response.content[0].text, call_name="opportunity identification")

    # Enrich with existing screener data if available
    screener = existing_screener_data or {}
    for section in ["opportunities", "risks", "hedges"]:
        for inst in result.get(section, []):
            ticker = inst.get("ticker", "")
            if ticker in screener:
                sd = screener[ticker]
                inst["existing_quality_rating"] = sd.get("quality_rating")
                inst["existing_margin_of_safety"] = sd.get("margin_of_safety")
                inst["stress_verdict"] = sd.get("stress_verdict")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Synthesis & Report
# ═══════════════════════════════════════════════════════════════════════════════

_REPORT_SYSTEM = """\
You are a senior investment strategist writing the final synthesis report \
for a scenario analysis. Combine the probability tree and opportunity \
analysis into an actionable report.

Include:
1. Executive summary (2-3 paragraphs)
2. Probability-weighted market outlook
3. Key monitoring indicators with specific trigger levels
4. Portfolio construction notes (sizing, timing, risk management)
5. Second-order effects most analysts would miss
6. Time sensitivity assessment

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "executive_summary": str,
  "probability_weighted_outlook": str,
  "top_opportunities": [top 10 ticker+name+reasoning],
  "top_risks": [top 10 ticker+name+reasoning],
  "all_weather_hedges": [list of hedge instruments],
  "scenario_specific_plays": {"branch_id": [tickers that outperform specifically in this branch]},
  "key_monitoring_indicators": [
    {"indicator": str, "signals_branch": str, "update_action": str}
  ],
  "second_order_effects": [str],
  "portfolio_construction_notes": str,
  "time_sensitivity": str,
  "confidence_summary": str
}
"""


def generate_forecast_report(
    brief: dict,
    tree: dict,
    opportunities: dict,
) -> dict:
    """Generate final synthesis report.

    Returns structured report dict.
    """
    client = _get_client()

    # Summarize opportunities for context (avoid token limit)
    opp_summary = {
        "top_opportunities": [
            {"ticker": o["ticker"], "name": o.get("name", ""), "impact": o.get("expected_impact_pct")}
            for o in opportunities.get("opportunities", [])[:15]
        ],
        "top_risks": [
            {"ticker": r["ticker"], "name": r.get("name", ""), "impact": r.get("expected_impact_pct")}
            for r in opportunities.get("risks", [])[:10]
        ],
        "hedges": [
            {"ticker": h["ticker"], "name": h.get("name", "")}
            for h in opportunities.get("hedges", [])[:5]
        ],
        "winners_summary": opportunities.get("winners_summary", []),
        "losers_summary": opportunities.get("losers_summary", []),
    }

    prompt = (
        f"Scenario Brief:\n{json.dumps(brief, indent=2)}\n\n"
        f"Probability Tree:\n{json.dumps(tree, indent=2)}\n\n"
        f"Opportunity Analysis:\n{json.dumps(opp_summary, indent=2)}\n\n"
        f"Generate the final synthesis report."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_REPORT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    return _parse_json_response(response.content[0].text, call_name="forecast report")


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_scenario_analysis(
    brief: dict,
    screener_data: dict | None = None,
    progress_callback=None,
) -> dict:
    """Run phases 2-4 in sequence.

    Parameters
    ----------
    brief : dict
        Completed Scenario Intelligence Brief from intake.
    screener_data : dict, optional
        Existing screener scores/ratings for enrichment.
    progress_callback : callable, optional
        Called with (step, total, message) after each phase.

    Returns
    -------
    dict with: tree, opportunities, report, scenario_id, timestamp
    """
    total_steps = 4

    # Phase 2: Probability tree
    if progress_callback:
        progress_callback(1, total_steps, "Building probability tree...")
    tree = build_probability_tree(brief)

    # Phase 3: Opportunity identification
    if progress_callback:
        progress_callback(2, total_steps, "Identifying investment opportunities...")

    # Build ticker universe from screener data
    universe = []
    enrichment = {}
    if screener_data is not None:
        import pandas as pd
        if isinstance(screener_data, pd.DataFrame) and "ticker" in screener_data.columns:
            universe = screener_data["ticker"].tolist()
            for _, row in screener_data.iterrows():
                enrichment[row["ticker"]] = row.to_dict()
        elif isinstance(screener_data, dict):
            universe = list(screener_data.keys())
            enrichment = screener_data

    opportunities = identify_opportunities(brief, tree, universe, enrichment)

    # Phase 4: Synthesis
    if progress_callback:
        progress_callback(3, total_steps, "Generating synthesis report...")
    report = generate_forecast_report(brief, tree, opportunities)

    if progress_callback:
        progress_callback(4, total_steps, "Analysis complete!")

    # Cache result
    scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _ensure_cache_dir()
    cache_file = _CACHE_DIR / f"{scenario_id}.json"
    result = {
        "scenario_id": scenario_id,
        "timestamp": datetime.now().isoformat(),
        "brief": brief,
        "tree": tree,
        "opportunities": opportunities,
        "report": report,
    }
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except OSError as e:
        logger.warning(f"Cache write failed: {e}")

    return result
