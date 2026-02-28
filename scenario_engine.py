"""
Geopolitical and macro scenario analysis using Claude API.

Sends a scenario description to Claude, gets structured sector/ticker
impact multipliers, and applies them to intrinsic valuations.
"""

import os
import json

import pandas as pd
import anthropic


MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are an expert macro and geopolitical analyst. Given a scenario
description, analyze the likely impact on S&P 500 sectors and specific stocks.

Return ONLY valid JSON (no markdown, no code fences) in this exact structure:
{
  "scenario_summary": "2-3 sentence plain English summary of the scenario",
  "confidence": "high|medium|low",
  "sector_impacts": {
    "Energy": {"multiplier": 1.20, "direction": "positive", "reasoning": "..."},
    "Industrials": {"multiplier": 0.90, "direction": "negative", "reasoning": "..."}
  },
  "specific_tickers": {
    "LMT": {"multiplier": 1.30, "reasoning": "..."},
    "DAL": {"multiplier": 0.75, "reasoning": "..."}
  },
  "macro_impacts": {
    "discount_rate_adjustment": 0.01,
    "reasoning": "..."
  },
  "second_order_effects": [
    {
      "effect": "Supply chain disruption raises input costs for manufacturers",
      "affected_sectors": ["Industrials", "Consumer Discretionary"],
      "lag": "3-6 months",
      "magnitude": "moderate"
    }
  ],
  "ai_exposure": {
    "high_opportunity": ["NVDA", "MSFT", "GOOGL"],
    "high_risk": ["INTC"],
    "reasoning": "1-2 sentence explanation of AI angle if relevant"
  },
  "winners": ["LMT", "RTX", "XOM"],
  "losers": ["DAL", "UAL", "AMAT"],
  "time_horizon": "immediate|3-6 months|6-12 months"
}

Rules:
- Include ALL sectors present in the data, even if the multiplier is 1.0 (neutral).
- Multipliers represent adjustments to intrinsic value. 1.20 = 20% upward revision.
- Only include specific_tickers for stocks with outsized impact vs their sector.
- Be conservative with multipliers: most should be 0.85-1.15 range.
- discount_rate_adjustment is basis point change (e.g. 0.01 = 1% increase).
- second_order_effects: cascading consequences beyond direct impact (e.g., supply chain, currency, trade).
- ai_exposure: only include if the scenario has relevance to AI/tech positioning.
- Keep reasoning concise (1-2 sentences per entry).
"""

# ── Preset scenarios ──────────────────────────────────────────────────────

PRESET_SCENARIOS = {
    "US-China Trade Escalation": (
        "The US announces 50% tariffs on all Chinese imports. China retaliates "
        "with export controls on rare earth minerals and bans on US tech products."
    ),
    "Middle East Oil Crisis": (
        "Iran closes the Strait of Hormuz after military escalation, disrupting "
        "20% of global oil supply. Oil prices spike to $150/barrel."
    ),
    "AI Regulation Wave": (
        "The EU passes strict AI regulation requiring licensing for all foundation "
        "models. The US follows with similar legislation within 6 months."
    ),
    "Fed Emergency Rate Cut": (
        "A sudden credit market freeze forces the Federal Reserve to cut rates "
        "by 100bps in an emergency meeting. Markets fear systemic risk."
    ),
    "Global Pandemic 2.0": (
        "A novel respiratory virus emerges with high transmissibility. Early "
        "reports suggest 2-3% mortality. WHO declares a global health emergency."
    ),
}


def analyze_scenario(
    scenario_text: str,
    stocks_df: pd.DataFrame,
    api_key: str | None = None,
) -> dict | None:
    """Send a scenario to Claude and get structured impact analysis.

    Parameters
    ----------
    scenario_text : str
        User's scenario description.
    stocks_df : pd.DataFrame
        Must have 'sector' column (for listing available sectors).
    api_key : str, optional
        Anthropic API key. Falls back to env var.

    Returns
    -------
    dict with scenario analysis, or None on error.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"error": "ANTHROPIC_API_KEY not set. Add it to your .env file."}

    sectors = sorted(stocks_df["sector"].unique().tolist())
    tickers_sample = stocks_df["ticker"].head(20).tolist()

    user_prompt = (
        f"Scenario: {scenario_text}\n\n"
        f"Sectors in the S&P 500 data: {', '.join(sectors)}\n\n"
        f"Sample tickers: {', '.join(tickers_sample)}\n\n"
        f"Analyze this scenario and return the JSON impact assessment."
    )

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()

        # Handle markdown code fences if Claude wraps it
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse Claude response as JSON: {e}"}
    except anthropic.APIError as e:
        return {"error": f"Anthropic API error: {e}"}
    except Exception as e:
        return {"error": f"Scenario analysis failed: {e}"}


def apply_scenario_to_valuations(
    scenario_result: dict,
    valuations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply scenario multipliers to intrinsic values.

    Parameters
    ----------
    scenario_result : dict
        From analyze_scenario().
    valuations_df : pd.DataFrame
        Must have columns: ticker, sector, intrinsic_value, margin_of_safety,
        current_price.

    Returns
    -------
    DataFrame with added columns:
        scenario_intrinsic_value, scenario_margin_of_safety,
        scenario_impact, scenario_multiplier_applied
    """
    df = valuations_df.copy()

    sector_impacts = scenario_result.get("sector_impacts", {})
    ticker_impacts = scenario_result.get("specific_tickers", {})
    dr_adj = scenario_result.get("macro_impacts", {}).get("discount_rate_adjustment", 0)

    df["scenario_multiplier_applied"] = 1.0
    df["scenario_impact"] = "neutral"

    # Apply sector multipliers
    for _, row in df.iterrows():
        sector = row.get("sector", "")
        ticker = row.get("ticker", "")
        idx = row.name

        # Sector-level multiplier
        sector_info = sector_impacts.get(sector, {})
        multiplier = sector_info.get("multiplier", 1.0)

        # Ticker-specific override
        if ticker in ticker_impacts:
            multiplier = ticker_impacts[ticker].get("multiplier", multiplier)

        df.at[idx, "scenario_multiplier_applied"] = multiplier

        if multiplier > 1.02:
            df.at[idx, "scenario_impact"] = "positive"
        elif multiplier < 0.98:
            df.at[idx, "scenario_impact"] = "negative"
        else:
            df.at[idx, "scenario_impact"] = "neutral"

    # Apply to intrinsic values
    df["scenario_intrinsic_value"] = df["intrinsic_value"] * df["scenario_multiplier_applied"]

    # Recalculate margin of safety
    df["scenario_margin_of_safety"] = df.apply(
        lambda r: (
            round((r["scenario_intrinsic_value"] - r["current_price"])
                  / r["scenario_intrinsic_value"] * 100, 2)
            if pd.notna(r["scenario_intrinsic_value"])
            and r["scenario_intrinsic_value"] > 0
            else None
        ),
        axis=1,
    )

    return df
