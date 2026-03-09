"""
Calibration Engine — Analyze forecasting accuracy and provide improvement insights.

Uses resolved scenario data from scenario_database to:
  - Build calibration curves (predicted probability vs actual frequency)
  - Grade forecasting accuracy
  - Detect bias (overconfident / underconfident)
  - Break down accuracy by domain/tag
  - Generate Claude-powered improvement insights
"""

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Calibration analysis ─────────────────────────────────────────────────────

def analyze_calibration(calibration_df: pd.DataFrame) -> dict:
    """Group by probability buckets and calculate actual frequency.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        From scenario_database.get_calibration_data(). Must have columns:
        predicted_probability, outcome_occurred, brier_contribution.

    Returns
    -------
    dict with: buckets, overall_brier_score, calibration_grade, bias, sample_size
    """
    if calibration_df.empty:
        return {
            "buckets": [],
            "overall_brier_score": None,
            "calibration_grade": "N/A",
            "bias": "insufficient_data",
            "sample_size": 0,
        }

    df = calibration_df.copy()
    df["predicted_probability"] = pd.to_numeric(
        df["predicted_probability"], errors="coerce"
    )
    df["outcome_occurred"] = pd.to_numeric(
        df["outcome_occurred"], errors="coerce"
    )
    df = df.dropna(subset=["predicted_probability", "outcome_occurred"])

    if df.empty:
        return {
            "buckets": [],
            "overall_brier_score": None,
            "calibration_grade": "N/A",
            "bias": "insufficient_data",
            "sample_size": 0,
        }

    # ── Bucket analysis ──────────────────────────────────────────
    bucket_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bucket_labels = [
        "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
    ]

    df["bucket"] = pd.cut(
        df["predicted_probability"],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True,
    )

    buckets = []
    for label in bucket_labels:
        group = df[df["bucket"] == label]
        if len(group) == 0:
            continue

        pred_mean = group["predicted_probability"].mean()
        actual_freq = group["outcome_occurred"].mean()
        cal_error = abs(pred_mean - actual_freq)

        buckets.append({
            "range": label,
            "predicted_probability": round(float(pred_mean), 3),
            "actual_frequency": round(float(actual_freq), 3),
            "count": int(len(group)),
            "calibration_error": round(float(cal_error), 3),
        })

    # ── Overall Brier score ──────────────────────────────────────
    brier_vals = df["brier_contribution"].dropna()
    overall_brier = float(brier_vals.mean()) if len(brier_vals) > 0 else None

    # ── Calibration grade ────────────────────────────────────────
    if overall_brier is None:
        grade = "N/A"
    elif overall_brier < 0.05:
        grade = "A"
    elif overall_brier < 0.10:
        grade = "B"
    elif overall_brier < 0.15:
        grade = "C"
    elif overall_brier < 0.20:
        grade = "D"
    else:
        grade = "F"

    # ── Bias detection ───────────────────────────────────────────
    if len(buckets) < 2:
        bias = "insufficient_data"
    else:
        # Compare average predicted vs actual
        total_predicted = sum(b["predicted_probability"] * b["count"] for b in buckets)
        total_actual = sum(b["actual_frequency"] * b["count"] for b in buckets)
        total_count = sum(b["count"] for b in buckets)

        if total_count > 0:
            avg_predicted = total_predicted / total_count
            avg_actual = total_actual / total_count
            diff = avg_predicted - avg_actual

            if diff > 0.05:
                bias = "overconfident"
            elif diff < -0.05:
                bias = "underconfident"
            else:
                bias = "well_calibrated"
        else:
            bias = "insufficient_data"

    return {
        "buckets": buckets,
        "overall_brier_score": round(overall_brier, 4) if overall_brier is not None else None,
        "calibration_grade": grade,
        "bias": bias,
        "sample_size": len(df),
    }


# ── Claude-powered insights ─────────────────────────────────────────────────

def generate_calibration_insights(analysis: dict) -> str:
    """Generate plain English assessment of forecasting accuracy.

    Parameters
    ----------
    analysis : dict
        From analyze_calibration().

    Returns
    -------
    str : Plain English assessment with improvement recommendations.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _generate_fallback_insights(analysis)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            f"Analyze this forecasting calibration data and provide specific "
            f"improvement recommendations:\n\n"
            f"{json.dumps(analysis, indent=2)}\n\n"
            f"Cover: (1) Overall accuracy assessment, (2) Where the forecaster "
            f"is well-calibrated vs poorly calibrated, (3) Specific bucket-level "
            f"issues (e.g., overconfident at 80-90%), (4) Actionable recommendations "
            f"to improve. Be direct and specific. 3-4 paragraphs max."
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=(
                "You are a calibration analyst for a superforecasting team. "
                "Provide specific, actionable feedback on probability calibration. "
                "Reference Good Judgment Project benchmarks where relevant."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    except Exception as e:
        logger.warning(f"Claude calibration insights failed: {e}")
        return _generate_fallback_insights(analysis)


def _generate_fallback_insights(analysis: dict) -> str:
    """Generate basic insights without Claude API."""
    grade = analysis.get("calibration_grade", "N/A")
    brier = analysis.get("overall_brier_score")
    bias = analysis.get("bias", "unknown")
    sample = analysis.get("sample_size", 0)

    lines = [f"**Calibration Grade: {grade}**\n"]

    if brier is not None:
        if brier < 0.10:
            lines.append(f"Brier score of {brier:.3f} is good — significantly better than random (0.25).")
        elif brier < 0.20:
            lines.append(f"Brier score of {brier:.3f} shows moderate accuracy. Room for improvement.")
        else:
            lines.append(f"Brier score of {brier:.3f} indicates forecasts are not much better than guessing (0.25).")

    if bias == "overconfident":
        lines.append(
            "You tend to be **overconfident** — assigned probabilities are higher "
            "than actual outcomes. Consider reducing your probability estimates by 5-10%."
        )
    elif bias == "underconfident":
        lines.append(
            "You tend to be **underconfident** — actual outcomes happen more often "
            "than you predict. Consider raising your probability estimates."
        )
    elif bias == "well_calibrated":
        lines.append("Your overall calibration is well-balanced — no systematic bias detected.")

    if sample < 10:
        lines.append(
            f"\n*Note: Only {sample} data points — calibration analysis becomes "
            f"meaningful with 20+ resolved scenarios.*"
        )

    # Bucket-level issues
    buckets = analysis.get("buckets", [])
    for b in buckets:
        if b["calibration_error"] > 0.15 and b["count"] >= 3:
            lines.append(
                f"- **{b['range']}** bucket: predicted {b['predicted_probability']:.0%} "
                f"but actual was {b['actual_frequency']:.0%} "
                f"(error: {b['calibration_error']:.0%}, n={b['count']})"
            )

    return "\n".join(lines)


# ── Domain accuracy ──────────────────────────────────────────────────────────

def get_domain_accuracy(calibration_df: pd.DataFrame) -> dict:
    """Break accuracy down by scenario type/tags.

    Returns dict mapping domain → accuracy metrics.
    """
    if calibration_df.empty or "scenario_title" not in calibration_df.columns:
        return {}

    # Group by scenario and compute per-scenario Brier
    result = {}
    for title, group in calibration_df.groupby("scenario_title"):
        brier_vals = group["brier_contribution"].dropna()
        if len(brier_vals) > 0:
            brier = float(brier_vals.mean())
            result[title] = {
                "brier_score": round(brier, 4),
                "predictions": len(group),
                "outcomes_correct": int(
                    (group["brier_contribution"] < 0.25).sum()
                ),
            }

    return result
