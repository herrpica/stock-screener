# 1. pip install -r requirements.txt
# 2. Copy .env.example to .env and add your Anthropic API key
# 3. streamlit run app.py
# First run will take 15-25 minutes to fetch all S&P 500 data.
# Subsequent runs use cached data and load in seconds.

"""
S&P 500 Stock Screener — Main Streamlit Application (v3).

Six pages: Screener | Stock Detail | Sector Overview | Scenario Analysis
           | SEC Intelligence | Superforecasting
"""

import os
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data_fetcher import load_or_fetch_data, get_cache_timestamp
from valuation_engine import calculate_all_valuations
from scoring_engine import score_all_stocks
from quality_rater import rate_all_stocks
from scenario_engine import (
    analyze_scenario, apply_scenario_to_valuations, PRESET_SCENARIOS,
)
from sec_fetcher import get_full_sec_analysis, get_recent_filings
from deep_dive_engine import (
    run_deep_dive, fetch_all_documents, estimate_deep_dive_cost,
    load_cached_deep_dive, load_deep_dive_history, load_deep_dive_by_path,
    STEPS as DD_STEPS,
)
from rate_stress_engine import (
    calculate_all_stress_valuations, calculate_all_stress_scenarios,
    calculate_debt_trajectory, SCENARIOS as STRESS_SCENARIOS,
)
from edgar_xbrl_fetcher import (
    get_xbrl_history, load_xbrl_for_tickers, get_data_quality_badge,
)
from quality_rater import rate_all_stocks_xbrl
from scoring_engine import backtest_scoring_methodology
from superforecast_engine import (
    conduct_intake_conversation, run_full_scenario_analysis,
)
from scenario_database import (
    save_scenario as db_save_scenario,
    get_scenario as db_get_scenario,
    get_all_scenarios as db_get_all_scenarios,
    update_scenario_results as db_update_scenario_results,
    resolve_scenario as db_resolve_scenario,
    save_recommendations as db_save_recommendations,
    get_calibration_data as db_get_calibration_data,
)
from calibration_engine import analyze_calibration, generate_calibration_insights

# ── App config ──────────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="S&P 500 Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state initialization ────────────────────────────────────────────

def _init_state():
    defaults = {
        "all_data": None,
        "valuations": None,
        "scores_df": None,
        "ratings_df": None,
        "master_df": None,
        "selected_ticker": None,
        "page": "Screener",
        "nav_radio": "Screener",
        "scenario_result": None,
        "scenario_active": False,
        "scenario_df": None,
        "scenario_history": [],
        "w_quality": 35,
        "w_growth": 30,
        "w_value": 25,
        "w_sentiment": 10,
        "discount_rate": 10,
        "data_loaded": False,
        "sec_cache": {},
        "deep_dive_cache": {},
        "dd_running": False,
        "dd_progress_step": 0,
        "session_api_cost": 0.0,
        "screener_search": "",
        "selected_for_deepdive": [],
        "dd_show_inline": None,
        "stress_rate": 12,
        "stress_valuations": None,
        "xbrl_data": {},
        "xbrl_loaded": False,
        "xbrl_loading": False,
        # Superforecasting
        "sf_intake_conversation": [],
        "sf_current_brief": {},
        "sf_intake_complete": False,
        "sf_scenario_result": None,
        "sf_current_scenario_id": None,
        "sf_resolving_scenario": None,
        # Ticker popup
        "popup_ticker": None,
        "show_popup": False,
        "watchlist": [],
        "screener_page_num": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# Load watchlist from disk
_WATCHLIST_FILE = pathlib.Path(__file__).parent / "watchlist.json"
if not st.session_state.watchlist:
    if _WATCHLIST_FILE.exists():
        try:
            import json as _wl_json
            with open(_WATCHLIST_FILE) as _wf:
                st.session_state.watchlist = _wl_json.load(_wf)
        except Exception:
            pass


def _save_watchlist():
    """Persist watchlist to disk."""
    import json as _wl_json
    try:
        with open(_WATCHLIST_FILE, "w") as f:
            _wl_json.dump(st.session_state.watchlist, f)
    except Exception:
        pass


def _navigate_to(page_name: str):
    """Programmatically navigate — sets pending flag and reruns."""
    st.session_state.page = page_name
    st.session_state._pending_nav = page_name


def _dedup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate index and columns so st.dataframe / Styler never chokes."""
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def safe_dataframe(df, **kwargs):
    """Safe wrapper for st.dataframe — prevents errors from duplicate index/columns."""
    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("No data to display.")
        return None
    df = _dedup_df(df.copy()) if not hasattr(df, "data") else df
    try:
        return st.dataframe(df, **kwargs)
    except (KeyError, ValueError):
        # Fallback — strip any Styler, dedup, try again
        raw = df.data.copy() if hasattr(df, "data") else df.copy()
        return st.dataframe(_dedup_df(raw), **kwargs)


# Load any existing deep dive cache files from disk on first run
if not st.session_state.deep_dive_cache:
    from pathlib import Path as _Path
    _dd_dir = _Path(__file__).parent / ".cache" / "deep_dive"
    if _dd_dir.exists():
        import json as _json
        for _f in _dd_dir.glob("*_deep_dive.json"):
            _tk = _f.stem.replace("_deep_dive", "").upper()
            try:
                with open(_f) as _fp:
                    st.session_state.deep_dive_cache[_tk] = _json.load(_fp)
            except Exception:
                pass


# ── Data loading & computation ──────────────────────────────────────────────

def _build_master_df() -> pd.DataFrame:
    """Merge valuations, scores, and ratings into a single master DataFrame."""
    all_data = st.session_state.all_data
    valuations = st.session_state.valuations
    scores_df = st.session_state.scores_df
    ratings_df = st.session_state.ratings_df

    stress_vals = st.session_state.get("stress_valuations") or {}

    rows = []
    for ticker, data in all_data.items():
        info = data.get("info", {})
        val = valuations.get(ticker, {})
        sv = stress_vals.get(ticker, {})
        sh = sv.get("scenarios", {}).get("sustained_high", {})
        row_data = {
            "ticker": ticker,
            "company": data.get("company", info.get("shortName", "")),
            "sector": data.get("sector", info.get("sector", "Unknown")),
            "current_price": val.get("current_price"),
            "intrinsic_value": val.get("intrinsic_value"),
            "margin_of_safety": val.get("margin_of_safety"),
            "base_eps": val.get("base_eps"),
            "growth_rate_used": val.get("growth_rate_used"),
            "val_error": val.get("error"),
            # Stress columns
            "stress_iv": sh.get("stress_intrinsic_value"),
            "stress_mos": sh.get("stress_margin_of_safety"),
            "stress_verdict": sv.get("stress_verdict"),
            "rate_status": sv.get("rate_risk_rating"),
            "stress_status": sh.get("stress_status"),
        }
        rows.append(row_data)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ticker"], keep="last")

    # Dedup source DataFrames before merging
    _scores = scores_df.drop_duplicates(subset=["ticker"], keep="last")
    _ratings = ratings_df[["ticker", "quality_rating", "rating_total",
                            "earnings_consistency_score", "debt_discipline_score",
                            "dividend_quality_score", "buyback_score"]].drop_duplicates(
        subset=["ticker"], keep="last"
    )

    df = df.merge(_scores, on="ticker", how="left", suffixes=("", "_score"))
    df = df.merge(_ratings, on="ticker", how="left")

    # Use sector from scores_df if the main one is missing
    if "sector_score" in df.columns:
        df["sector"] = df["sector"].fillna(df["sector_score"])
        df.drop(columns=["sector_score"], inplace=True, errors="ignore")

    # Final dedup and clean index
    _dup_count = df["ticker"].duplicated().sum()
    if _dup_count > 0:
        print(f"[dedup] master_df had {_dup_count} duplicate tickers after merge — deduplicating")
    _dup_cols = df.columns[df.columns.duplicated()].tolist()
    if _dup_cols:
        print(f"[dedup] master_df had duplicate columns: {_dup_cols} — keeping first")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return df


def load_data(force_refresh=False):
    """Load or fetch all data and compute derived tables."""
    all_data = load_or_fetch_data(force_refresh=force_refresh)
    st.session_state.all_data = all_data

    weights = {
        "quality": st.session_state.w_quality / 100,
        "growth": st.session_state.w_growth / 100,
        "value": st.session_state.w_value / 100,
        "sentiment": st.session_state.w_sentiment / 100,
    }

    required_rate = st.session_state.discount_rate / 100

    with st.spinner("Computing valuations..."):
        st.session_state.valuations = calculate_all_valuations(
            all_data, required_rate=required_rate
        )

    with st.spinner("Computing scores..."):
        st.session_state.scores_df = score_all_stocks(all_data, weights)

    with st.spinner("Computing quality ratings..."):
        st.session_state.ratings_df = rate_all_stocks(
            all_data, st.session_state.scores_df
        )

    with st.spinner("Computing rate stress valuations..."):
        stress_override = st.session_state.stress_rate / 100
        st.session_state.stress_valuations = calculate_all_stress_valuations(
            all_data, st.session_state.valuations, stress_rate_override=stress_override
        )

    st.session_state.master_df = _build_master_df()
    st.session_state.data_loaded = True


# ── Sidebar ─────────────────────────────────────────────────────────────────

PAGES = ["Screener", "Stock Detail", "Sector Overview",
         "Scenario Analysis", "SEC Intelligence", "Superforecasting"]

# ── Handle pending navigation BEFORE any widget renders ──────────────────────
if "_pending_nav" in st.session_state:
    _pn = st.session_state._pending_nav
    del st.session_state._pending_nav
    st.session_state.nav_radio = _pn
    st.session_state.page = _pn

with st.sidebar:
    st.title("📊 Stock Screener")

    cache_ts = get_cache_timestamp()
    if cache_ts:
        st.caption(f"Data: {cache_ts.strftime('%Y-%m-%d %H:%M')}")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.data_loaded = False
        load_data(force_refresh=True)
        st.rerun()

    st.divider()

    # Stock search — available on every page
    if st.session_state.data_loaded and st.session_state.all_data:
        all_tickers = sorted(st.session_state.all_data.keys())
        # Build display list: "AAPL — Apple Inc."
        ticker_labels = []
        for t in all_tickers:
            comp = st.session_state.all_data[t].get(
                "company", st.session_state.all_data[t].get("info", {}).get("shortName", "")
            )
            ticker_labels.append(f"{t} — {comp}" if comp else t)
        search_pick = st.selectbox(
            "🔍 Search stock",
            options=range(len(all_tickers)),
            format_func=lambda i: ticker_labels[i],
            index=None,
            key="sidebar_search",
            placeholder="Ticker or company name...",
        )
        if search_pick is not None:
            st.session_state.selected_ticker = all_tickers[search_pick]
            _navigate_to("Stock Detail")
            st.rerun()

    st.divider()

    page = st.radio(
        "Navigation",
        PAGES,
        key="nav_radio",
    )
    st.session_state.page = page

    st.divider()
    st.subheader("Pillar Weights")

    w_q = st.slider("Quality %", 0, 100, st.session_state.w_quality, key="sl_q")
    w_g = st.slider("Growth %", 0, 100, st.session_state.w_growth, key="sl_g")
    w_v = st.slider("Value %", 0, 100, st.session_state.w_value, key="sl_v")
    w_s = st.slider("Sentiment %", 0, 100, st.session_state.w_sentiment, key="sl_s")

    total_w = w_q + w_g + w_v + w_s
    if total_w != 100:
        st.warning(f"Weights sum to {total_w}%, should be 100%")

    st.divider()
    st.subheader("Discount Rate")
    dr = st.slider("Required Return %", 5, 20, st.session_state.discount_rate, key="sl_dr")
    st.subheader("Stress Test Rate")
    sr = st.slider(
        "Sustained High Rate Assumption %", 10, 16,
        st.session_state.stress_rate, key="sl_sr",
        help="Assumes debt refinances at this rate permanently. Current 10yr Treasury: ~4.5%",
    )

    weights_changed = (
        w_q != st.session_state.w_quality
        or w_g != st.session_state.w_growth
        or w_v != st.session_state.w_value
        or w_s != st.session_state.w_sentiment
    )
    dr_changed = dr != st.session_state.discount_rate
    sr_changed = sr != st.session_state.stress_rate

    if weights_changed:
        st.session_state.w_quality = w_q
        st.session_state.w_growth = w_g
        st.session_state.w_value = w_v
        st.session_state.w_sentiment = w_s
        if st.session_state.data_loaded:
            weights = {
                "quality": w_q / 100, "growth": w_g / 100,
                "value": w_v / 100, "sentiment": w_s / 100,
            }
            st.session_state.scores_df = score_all_stocks(
                st.session_state.all_data, weights
            )
            st.session_state.master_df = _build_master_df()

    if dr_changed:
        st.session_state.discount_rate = dr
        if st.session_state.data_loaded:
            st.session_state.valuations = calculate_all_valuations(
                st.session_state.all_data, required_rate=dr / 100
            )
            st.session_state.stress_valuations = calculate_all_stress_valuations(
                st.session_state.all_data, st.session_state.valuations,
                stress_rate_override=st.session_state.stress_rate / 100,
            )
            st.session_state.master_df = _build_master_df()

    if sr_changed:
        st.session_state.stress_rate = sr
        if st.session_state.data_loaded and st.session_state.valuations:
            st.session_state.stress_valuations = calculate_all_stress_valuations(
                st.session_state.all_data, st.session_state.valuations,
                stress_rate_override=sr / 100,
            )
            st.session_state.master_df = _build_master_df()

    st.divider()
    if st.session_state.data_loaded and st.session_state.all_data:
        st.caption(f"{len(st.session_state.all_data)} stocks loaded")

        # ── XBRL Historical Data ─────────────────────────────────
        st.subheader("EDGAR XBRL History")
        if st.session_state.xbrl_loaded:
            xbrl_ok = sum(
                1 for v in st.session_state.xbrl_data.values()
                if v.get("available")
            )
            st.caption(f"{xbrl_ok}/{len(st.session_state.xbrl_data)} tickers with XBRL data")
            if st.button("Refresh XBRL", key="xbrl_refresh", use_container_width=True):
                st.session_state.xbrl_loaded = False
                st.session_state.xbrl_loading = True
                st.rerun()
        else:
            if st.button(
                "Load XBRL History",
                key="xbrl_load",
                use_container_width=True,
                help="Fetch 10-15+ years of SEC EDGAR XBRL data for all loaded tickers. "
                     "No API key required. Takes 2-5 minutes on first load.",
            ):
                st.session_state.xbrl_loading = True
                st.rerun()

    st.divider()
    cost = st.session_state.session_api_cost
    st.caption(f"Session API cost: ~${cost:.2f}")


# ── XBRL loading handler ────────────────────────────────────────────────────

if st.session_state.xbrl_loading and st.session_state.data_loaded:
    tickers = sorted(st.session_state.all_data.keys())
    xbrl_bar = st.progress(0, text="Loading XBRL data...")
    xbrl_status = st.empty()

    def _xbrl_progress(idx, total, ticker, status):
        frac = (idx + 1) / total
        xbrl_bar.progress(frac, text=f"XBRL: {ticker} ({idx + 1}/{total}) — {status}")
        xbrl_status.text(f"Processing {ticker}...")

    st.session_state.xbrl_data = load_xbrl_for_tickers(tickers, _xbrl_progress)
    xbrl_bar.empty()
    xbrl_status.empty()

    st.session_state.xbrl_loaded = True
    st.session_state.xbrl_loading = False

    # Recompute quality ratings with XBRL data
    with st.spinner("Recomputing quality ratings with XBRL history..."):
        st.session_state.ratings_df = rate_all_stocks_xbrl(
            st.session_state.all_data,
            st.session_state.scores_df,
            st.session_state.xbrl_data,
        )
    st.session_state.master_df = _build_master_df()
    st.rerun()


# ── Cost tracking helper ────────────────────────────────────────────────────

_API_COSTS = {
    "sec_analysis": 0.02,
    "deep_dive": 0.10,
    "scenario": 0.03,
}

def _track_cost(call_type: str, count: int = 1):
    """Add estimated API cost to the session total."""
    st.session_state.session_api_cost += _API_COSTS[call_type] * count


def _render_dd_inline(dd_result, ticker, val=None):
    """Render deep dive results inline — no tab navigation required."""
    thesis = dd_result.get("investment_thesis", {})
    biz = dd_result.get("business_assessment", {})
    recon = dd_result.get("metric_reconciliation", {})
    fwd = dd_result.get("forward_analysis", {})
    adj_scores = dd_result.get("adjusted_scores", {})
    adj_val = dd_result.get("adjusted_valuation", {})

    rec = thesis.get("recommendation", "N/A")
    conviction = thesis.get("conviction_level", "N/A")
    one_liner = thesis.get("one_line_thesis", "")

    _REC_COLORS = {
        "Strong Buy": "#1a7a3a", "Buy": "#27ae60",
        "Hold": "#f39c12", "Avoid": "#e67e22", "Strong Avoid": "#c0392b",
    }
    rec_color = _REC_COLORS.get(rec, "gray")

    # ── Recommendation ──
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin:12px 0">'
        f'<span style="background:{rec_color};color:white;padding:10px 22px;'
        f'border-radius:8px;font-size:1.4em;font-weight:bold">{rec}</span>'
        f'<span style="background:#eee;padding:8px 16px;border-radius:6px">'
        f'Conviction: {conviction.upper()}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'*{one_liner}*')
    memo = thesis.get("memo_summary", "")
    if memo:
        st.info(memo)

    # ── Score Adjustments ──
    st.subheader("Score Adjustments")
    orig_scores = dd_result.get("original_scores", {})
    change_rows = []
    for key, name in [("quality_score", "Quality"), ("growth_score", "Growth"),
                      ("value_score", "Value"), ("sentiment_score", "Sentiment"),
                      ("composite_score", "Composite")]:
        orig = orig_scores.get(key, "—")
        adj = adj_scores.get(key, orig)
        delta = adj_scores.get(f"{key}_delta", 0)
        change_rows.append({"Metric": name, "Original": orig, "Adjusted": adj,
                            "Change": f"{delta:+.1f}" if isinstance(delta, (int, float)) and delta else "—"})
    if val:
        orig_iv = val.get("intrinsic_value")
        adj_iv = adj_val.get("intrinsic_value", 0)
        iv_d = adj_val.get("intrinsic_value_delta", 0)
        change_rows.append({"Metric": "Intrinsic Value",
                            "Original": f"${orig_iv:.2f}" if orig_iv else "N/A",
                            "Adjusted": f"${adj_iv:.2f}",
                            "Change": f"${iv_d:+.2f}" if iv_d else "—"})
    safe_dataframe(pd.DataFrame(change_rows), hide_index=True, use_container_width=True)

    # ── Bull / Base / Bear ──
    st.subheader("Scenario Analysis")
    bull = thesis.get("bull_case", {})
    base_case = thesis.get("base_case", {})
    bear = thesis.get("bear_case", {})
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.markdown("#### 🟢 Bull Case")
        up = bull.get("upside_to_intrinsic_value_pct")
        if up is not None:
            st.markdown(f'<span style="font-size:1.5em;font-weight:bold;color:#27ae60">+{up:.0f}%</span>',
                        unsafe_allow_html=True)
        st.markdown(bull.get("narrative", ""))
        for a in bull.get("key_assumptions", []):
            st.markdown(f"- {a}")
    with bc2:
        st.markdown("#### 🟡 Base Case")
        ret = base_case.get("expected_return_12_month_pct")
        if ret is not None:
            c = "#27ae60" if ret > 0 else "#e74c3c"
            st.markdown(f'<span style="font-size:1.5em;font-weight:bold;color:{c}">{ret:+.0f}%</span>',
                        unsafe_allow_html=True)
        st.markdown(base_case.get("narrative", ""))
        for a in base_case.get("key_assumptions", []):
            st.markdown(f"- {a}")
    with bc3:
        st.markdown("#### 🔴 Bear Case")
        dn = bear.get("downside_pct")
        if dn is not None:
            st.markdown(f'<span style="font-size:1.5em;font-weight:bold;color:#e74c3c">-{abs(dn):.0f}%</span>',
                        unsafe_allow_html=True)
        st.markdown(bear.get("narrative", ""))
        for r in bear.get("key_risks", []):
            st.markdown(f"- {r}")

    # ── Red Flags ──
    red_flags = recon.get("red_flags", [])
    if red_flags:
        st.subheader("Red Flags")
        for rf in red_flags:
            sev = rf.get("severity", "minor")
            if sev == "critical":
                st.error(f"**CRITICAL:** {rf.get('flag', '')}")
            elif sev == "significant":
                st.warning(f"**{rf.get('flag', '')}**")
            else:
                st.info(rf.get("flag", ""))

    # ── Forward Outlook ──
    st.subheader("Forward Outlook")
    fo1, fo2, fo3 = st.columns(3)
    with fo1:
        mt = fwd.get("margin_trajectory", {})
        st.metric("Margin Trajectory", mt.get("direction", "N/A"))
    with fo2:
        rt = fwd.get("revenue_trajectory", {})
        st.metric("Revenue Trajectory", rt.get("direction", "N/A"))
    with fo3:
        st.metric("Competitive Position", fwd.get("competitive_position_trend", "N/A"))

    catalysts = fwd.get("growth_catalysts", [])
    if catalysts:
        cat_rows = [{"Catalyst": c.get("catalyst", ""), "Timeline": c.get("timeline", ""),
                     "Magnitude": c.get("magnitude", "")} for c in catalysts]
        safe_dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

    # ── Entry Analysis ──
    st.subheader("Entry Analysis")
    entry_price = thesis.get("ideal_entry_price")
    sizing = thesis.get("position_sizing_suggestion", "N/A")
    timeline = thesis.get("time_to_thesis_realization", "N/A")
    e1, e2, e3 = st.columns(3)
    with e1:
        if entry_price:
            st.metric("Ideal Entry Price", f"${entry_price:.2f}")
        else:
            st.metric("Ideal Entry Price", "N/A")
    with e2:
        st.metric("Position Sizing", sizing.replace("_", " ").title() if isinstance(sizing, str) else "N/A")
    with e3:
        st.metric("Time to Realization", timeline)


def _run_dd_for_ticker(ticker, api_key, force_refresh=False):
    """Run deep dive for a single ticker. Returns result dict or None on error."""
    t_data = st.session_state.all_data.get(ticker, {})
    t_info = t_data.get("info", {})
    t_val = st.session_state.valuations.get(ticker, {})
    m = st.session_state.master_df
    t_row = m[m["ticker"] == ticker].iloc[0] if ticker in m["ticker"].values else None
    t_scores = {}
    if t_row is not None:
        t_scores = {
            "quality_score": t_row.get("quality_score", 50),
            "growth_score": t_row.get("growth_score", 50),
            "value_score": t_row.get("value_score", 50),
            "sentiment_score": t_row.get("sentiment_score", 50),
            "composite_score": t_row.get("composite_score", 50),
        }
    t_qr = {}
    if st.session_state.ratings_df is not None:
        rr = st.session_state.ratings_df
        rr_rows = rr[rr["ticker"] == ticker]
        if not rr_rows.empty:
            t_qr = rr_rows.iloc[0].to_dict()
    try:
        result = run_deep_dive(
            ticker=ticker,
            company_name=t_data.get("company", t_info.get("shortName", ticker)),
            current_scores=t_scores,
            current_valuation=t_val,
            current_quality_rating=t_qr,
            sector=t_data.get("sector", t_info.get("sector", "")),
            sector_medians={},
            api_key=api_key,
            discount_rate=st.session_state.discount_rate / 100,
            force_refresh=force_refresh,
        )
        if "error" not in result:
            _track_cost("deep_dive")
            st.session_state.deep_dive_cache[ticker] = result
            return result
        return None
    except Exception:
        return None


# ── Load data on first run ──────────────────────────────────────────────────

if not st.session_state.data_loaded:
    load_data()

master = st.session_state.master_df
if master is None or master.empty:
    st.error("No data available. Click 'Refresh Data' in the sidebar.")
    st.stop()


# ── Color helpers ───────────────────────────────────────────────────────────

def _mos_color(val):
    if val is None or pd.isna(val):
        return "color: gray"
    if val > 20:
        return "color: #27ae60; font-weight: bold"
    if val > 0:
        return "color: #f39c12; font-weight: bold"
    return "color: #e74c3c; font-weight: bold"


def _rating_badge(rating):
    if rating == "Above Bar":
        return "🟢 Above Bar"
    if rating == "At Bar":
        return "⚪ At Bar"
    return "🔴 Below Bar"


def _sec_flag_badge(flag):
    if flag == "green":
        return "🟢"
    if flag == "yellow":
        return "🟡"
    if flag == "red":
        return "🔴"
    return "⚫"


def _rec_badge(rec):
    """Colored recommendation badge for deep dive."""
    colors = {
        "Strong Buy": ("🟢", "#27ae60"),
        "Buy": ("🟢", "#2ecc71"),
        "Hold": ("🟡", "#f39c12"),
        "Avoid": ("🔴", "#e74c3c"),
        "Strong Avoid": ("🔴", "#c0392b"),
    }
    icon, color = colors.get(rec, ("⚫", "gray"))
    return f'{icon} <span style="color:{color};font-weight:bold;font-size:1.1em">{rec}</span>'


def _delta_arrow(delta):
    """Return colored arrow for delta values."""
    if delta > 0:
        return f'<span style="color:#27ae60">+{delta:.1f} ↑</span>'
    elif delta < 0:
        return f'<span style="color:#e74c3c">{delta:.1f} ↓</span>'
    return '<span style="color:gray">0 →</span>'


# ═════════════════════════════════════════════════════════════════════════════
# TICKER POPUP
# ═════════════════════════════════════════════════════════════════════════════

def _open_popup(ticker: str):
    """Set session state to show the ticker popup."""
    st.session_state.popup_ticker = ticker
    st.session_state.show_popup = True


def render_ticker_popup(ticker: str):
    """Render a self-contained ticker analysis panel at the top of the page."""

    # ── Close button ─────────────────────────────────────────────
    close_col, title_col = st.columns([1, 10])
    with close_col:
        if st.button("Close", key="close_popup", type="secondary"):
            st.session_state.show_popup = False
            st.session_state.popup_ticker = None
            st.rerun()

    # ── Gather data ──────────────────────────────────────────────
    all_data = st.session_state.all_data or {}
    data = all_data.get(ticker)
    in_screener = data is not None

    # If not in screener, try to fetch basic info
    if not in_screener:
        try:
            import yfinance as yf
            yfobj = yf.Ticker(ticker)
            yf_info = yfobj.info or {}
            data = {
                "info": yf_info,
                "sector": yf_info.get("sector", "Unknown"),
                "company": yf_info.get("shortName", ticker),
                "financials": yfobj.financials,
                "balance_sheet": yfobj.balance_sheet,
                "cashflow": yfobj.cashflow,
                "dividends": yfobj.dividends,
            }
        except Exception:
            data = {"info": {}, "sector": "Unknown", "company": ticker}

    info = data.get("info", {})
    val = (st.session_state.valuations or {}).get(ticker, {})
    master = st.session_state.master_df
    row = None
    if master is not None and ticker in master["ticker"].values:
        row = master[master["ticker"] == ticker].iloc[0]

    company = data.get("company", info.get("shortName", ticker))
    sector = data.get("sector", info.get("sector", "Unknown"))
    price = val.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice")
    iv = val.get("intrinsic_value")
    mos = val.get("margin_of_safety")
    rating = row["quality_rating"] if row is not None and "quality_rating" in row.index else "N/A"

    stress_data = (st.session_state.stress_valuations or {}).get(ticker, {})
    sh = stress_data.get("scenarios", {}).get("sustained_high", {})
    stress_iv = sh.get("stress_intrinsic_value")
    stress_mos = sh.get("stress_margin_of_safety")
    rate_risk = stress_data.get("rate_risk_rating", "N/A")
    stress_verdict = stress_data.get("stress_verdict", "N/A")

    # ── Header Row ───────────────────────────────────────────────
    with title_col:
        st.markdown(
            f"### {company} ({ticker})"
            f'&ensp;<span style="color:gray">{sector}</span>',
            unsafe_allow_html=True,
        )

    hm = st.columns(6)
    hm[0].metric("Price", f"${price:.2f}" if price else "N/A")
    hm[1].metric("Intrinsic Value", f"${iv:.2f}" if iv else "N/A")
    hm[2].metric("Stress IV (12%)", f"${stress_iv:.2f}" if stress_iv else "N/A")
    if mos is not None:
        hm[3].metric("MoS", f"{mos:.1f}%")
    else:
        hm[3].metric("MoS", val.get("error", "N/A"))

    # Badges
    hm[4].markdown(f"**Rating:** {_rating_badge(rating)}", unsafe_allow_html=True)
    _rr_colors = {
        "Rate Beneficiary": "#27ae60", "Resilient": "#2980b9",
        "Sensitive": "#f39c12", "Vulnerable": "#e74c3c",
    }
    rr_c = _rr_colors.get(rate_risk, "#888")
    hm[5].markdown(
        f'<span style="background:{rr_c};color:white;padding:3px 10px;'
        f'border-radius:4px;font-size:0.85em">{rate_risk}</span>',
        unsafe_allow_html=True,
    )

    # XBRL badge
    xbrl_r = st.session_state.xbrl_data.get(ticker) if st.session_state.xbrl_loaded else None
    if xbrl_r:
        xb = get_data_quality_badge(ticker, xbrl_r)
        st.markdown(
            f'<span style="background:{xb["color"]};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.8em">{xb["label"]}</span>',
            unsafe_allow_html=True,
        )

    if not in_screener:
        st.warning(f"{ticker} is not in the loaded S&P 500 universe. Showing available data only.")

    # ── Three Column Summary ─────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown("**Fundamental Snapshot**")
        _fs = {
            "P/E": info.get("trailingPE"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "P/B": info.get("priceToBook"),
            "ROE": info.get("returnOnEquity"),
            "Net Margin": info.get("profitMargins"),
            "D/E": info.get("debtToEquity"),
        }
        for k, v in _fs.items():
            if v is not None:
                if k in ("ROE", "Net Margin"):
                    st.markdown(f"- **{k}:** {v:.1%}")
                elif k == "D/E":
                    st.markdown(f"- **{k}:** {v:.1f}")
                else:
                    st.markdown(f"- **{k}:** {v:.2f}")
            else:
                st.markdown(f"- **{k}:** N/A")

    with sc2:
        st.markdown("**Valuation Summary**")
        st.markdown(f"- **Base IV:** ${iv:.2f}" if iv else "- **Base IV:** N/A")
        gr = val.get("growth_rate_used")
        st.markdown(f"- **Growth Rate:** {gr:.1%}" if gr else "- **Growth Rate:** N/A")
        st.markdown(f"- **Stress IV (12%):** ${stress_iv:.2f}" if stress_iv else "- **Stress IV:** N/A")
        st.markdown(f"- **MoS (Base):** {mos:.1f}%" if mos is not None else "- **MoS (Base):** N/A")
        st.markdown(
            f"- **MoS (Stress):** {stress_mos:.1f}%"
            if stress_mos is not None else "- **MoS (Stress):** N/A"
        )
        _verd_colors = {
            "Undervalued at Stress Rates": "#27ae60",
            "Fairly Valued at Stress Rates": "#f39c12",
            "Overvalued at Stress Rates": "#e74c3c",
            "Rate Beneficiary — Enhanced Value": "#2980b9",
            "Earnings Impaired at Stress Rates": "#c0392b",
        }
        sv_c = _verd_colors.get(stress_verdict, "#888")
        st.markdown(
            f'- **Verdict:** <span style="color:{sv_c};font-weight:bold">'
            f'{stress_verdict}</span>',
            unsafe_allow_html=True,
        )

    with sc3:
        st.markdown("**Quality Rating**")
        st.markdown(f"**{_rating_badge(rating)}**", unsafe_allow_html=True)
        if row is not None:
            st.markdown(f"- Earnings Consistency: {row.get('earnings_consistency_score', 'N/A')}")
            st.markdown(f"- Debt Discipline: {row.get('debt_discipline_score', 'N/A')}")
            st.markdown(f"- Dividend Quality: {row.get('dividend_quality_score', 'N/A')}")
            st.markdown(f"- Buyback Score: {row.get('buyback_score', 'N/A')}")

    # ── EDGAR XBRL Historical Charts ─────────────────────────────
    st.divider()

    xbrl_this = xbrl_r
    has_xbrl = xbrl_this and xbrl_this.get("available")

    if has_xbrl:
        _xdf = xbrl_this["annual_data"]
        n_years = xbrl_this.get("years_available", 0)
        st.subheader(f"EDGAR XBRL History — {n_years} Years")

        def _popup_recession_shading(fig, years_list):
            for ry_start, ry_end in [(2007.5, 2009.5), (2019.5, 2020.5)]:
                if any(ry_start <= y <= ry_end for y in years_list):
                    fig.add_vrect(
                        x0=ry_start, x1=ry_end,
                        fillcolor="rgba(255, 0, 0, 0.08)", line_width=0,
                    )
            return fig

        pc = st.columns(2)

        with pc[0]:
            # EPS chart
            eps_col = None
            for _c in ["eps_diluted", "eps_basic"]:
                if _c in _xdf.columns:
                    eps_col = _c
                    break
            if eps_col:
                _eps = _xdf[eps_col].dropna()
                if len(_eps) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=_eps.index.tolist(), y=_eps.values,
                        mode="lines+markers",
                        line=dict(color="#2980b9"),
                        name="EPS",
                    ))
                    if len(_eps) >= 3:
                        z = np.polyfit(range(len(_eps)), _eps.values, 1)
                        trend = np.polyval(z, range(len(_eps)))
                        fig.add_trace(go.Scatter(
                            x=_eps.index.tolist(), y=trend,
                            mode="lines", line=dict(dash="dash", color="gray"),
                            name="Trend",
                        ))
                    # Annotate latest
                    fig.add_annotation(
                        x=_eps.index[-1], y=_eps.iloc[-1],
                        text=f"${_eps.iloc[-1]:.2f}",
                        showarrow=True, arrowhead=2,
                    )
                    fig = _popup_recession_shading(fig, _eps.index.tolist())
                    fig.update_layout(
                        title=f"Earnings Per Share — {len(_eps)} Year History",
                        xaxis_title="Year", yaxis_title="EPS ($)",
                        showlegend=False, height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with pc[1]:
            # Revenue chart
            if "revenue" in _xdf.columns:
                _rev = _xdf["revenue"].dropna() / 1e9
                if len(_rev) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=_rev.index.tolist(), y=_rev.values,
                        mode="lines+markers", line=dict(color="#3498db"),
                        name="Revenue",
                    ))
                    fig = _popup_recession_shading(fig, _rev.index.tolist())
                    # Subtitle with growth rate
                    if len(_rev) >= 2:
                        oldest = _rev.iloc[0]
                        newest = _rev.iloc[-1]
                        n_yr = _rev.index[-1] - _rev.index[0]
                        if oldest > 0 and n_yr > 0:
                            cagr = (newest / oldest) ** (1 / n_yr) - 1
                            subtitle = f"CAGR: {cagr:.1%}"
                        else:
                            subtitle = ""
                    else:
                        subtitle = ""
                    fig.update_layout(
                        title=f"Revenue ($B) — {len(_rev)} Year History"
                              + (f"<br><sub>{subtitle}</sub>" if subtitle else ""),
                        xaxis_title="Year", yaxis_title="Revenue ($B)",
                        showlegend=False, height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        pc2 = st.columns(2)

        with pc2[0]:
            # D/E chart
            if "debt_to_equity" in _xdf.columns:
                _de = _xdf["debt_to_equity"].dropna()
                if len(_de) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=_de.index.tolist(), y=_de.values,
                        mode="lines+markers", line=dict(color="#e67e22"),
                        name="D/E",
                    ))
                    # Sector median line
                    sec_de = info.get("debtToEquity")
                    if sec_de is not None:
                        fig.add_hline(
                            y=sec_de / 100, line_dash="dot", line_color="#999",
                            annotation_text=f"Current {sec_de/100:.1f}",
                        )
                    # Low-rate era shading
                    if any(2019 <= y <= 2022 for y in _de.index):
                        fig.add_vrect(
                            x0=2019.5, x1=2021.5,
                            fillcolor="rgba(52, 152, 219, 0.1)",
                            line_width=0,
                            annotation_text="Low-Rate Era",
                            annotation_position="top left",
                            annotation_font_size=10,
                        )
                    fig.update_layout(
                        title=f"Debt/Equity — {len(_de)} Year History",
                        xaxis_title="Year", yaxis_title="D/E",
                        showlegend=False, height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with pc2[1]:
            # FCF chart
            if "free_cash_flow" in _xdf.columns:
                _fcf = _xdf["free_cash_flow"].dropna() / 1e9
                if len(_fcf) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=_fcf.index.tolist(), y=_fcf.values,
                        marker_color=[
                            "#27ae60" if v >= 0 else "#e74c3c" for v in _fcf.values
                        ],
                        name="FCF",
                    ))
                    # FCF margin overlay
                    if "fcf_margin" in _xdf.columns:
                        _fcfm = _xdf["fcf_margin"].dropna()
                        common = _fcf.index.intersection(_fcfm.index)
                        if len(common) > 0:
                            fig.add_trace(go.Scatter(
                                x=common.tolist(),
                                y=(_fcfm[common] * 100).values,
                                mode="lines+markers",
                                line=dict(color="#8e44ad", dash="dot"),
                                name="FCF Margin %",
                                yaxis="y2",
                            ))
                            fig.update_layout(
                                yaxis2=dict(
                                    title="FCF Margin %", overlaying="y",
                                    side="right", showgrid=False,
                                ),
                            )
                    fig.update_layout(
                        title=f"Free Cash Flow ($B) — {len(_fcf)} Year History",
                        xaxis_title="Year", yaxis_title="FCF ($B)",
                        showlegend=False, height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ── Key Metrics Table ────────────────────────────────────
        st.subheader("Key Metrics (XBRL)")
        metric_cols = []
        for c in ["revenue", "net_income", "eps_diluted", "free_cash_flow",
                   "debt_to_equity", "roe", "net_margin"]:
            if c in _xdf.columns:
                metric_cols.append(c)

        if metric_cols:
            metrics_df = _xdf[metric_cols].copy()
            # Show up to 10 most recent years
            metrics_df = metrics_df.tail(10)

            # Format
            display_names = {
                "revenue": "Revenue",
                "net_income": "Net Income",
                "eps_diluted": "EPS",
                "free_cash_flow": "FCF",
                "debt_to_equity": "D/E",
                "roe": "ROE",
                "net_margin": "Net Margin",
            }
            fmt_df = pd.DataFrame(index=metrics_df.index)
            for c in metric_cols:
                name = display_names.get(c, c)
                if c in ("revenue", "net_income", "free_cash_flow"):
                    fmt_df[name] = metrics_df[c].apply(
                        lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "—"
                    )
                elif c == "eps_diluted":
                    fmt_df[name] = metrics_df[c].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) else "—"
                    )
                elif c in ("roe", "net_margin"):
                    fmt_df[name] = metrics_df[c].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
                    )
                else:
                    fmt_df[name] = metrics_df[c].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )

            fmt_df.index.name = "Year"
            safe_dataframe(fmt_df, use_container_width=True)

            # CAGR row
            cagr_items = []
            for c in ["revenue", "net_income", "eps_diluted", "free_cash_flow"]:
                if c in metrics_df.columns:
                    vals = metrics_df[c].dropna()
                    if len(vals) >= 2 and vals.iloc[0] > 0 and vals.iloc[-1] > 0:
                        n = vals.index[-1] - vals.index[0]
                        if n > 0:
                            cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / n) - 1
                            cagr_items.append(f"{display_names.get(c, c)} CAGR: {cagr:.1%}")
            if cagr_items:
                st.caption(" | ".join(cagr_items))

    else:
        st.info(
            "EDGAR XBRL data not available for this ticker "
            "(common for ETFs and foreign stocks). Showing yfinance data."
        )
        # Fallback: basic yfinance charts
        financials = data.get("financials")
        if financials is not None and not financials.empty:
            fc1, fc2 = st.columns(2)
            with fc1:
                for label in ["Net Income", "Net Income Common Stockholders"]:
                    if label in financials.index:
                        ni = financials.loc[label].dropna().sort_index()
                        shares = info.get("sharesOutstanding", 1)
                        if shares:
                            eps_s = ni / shares
                            fig = px.bar(
                                x=eps_s.index.strftime("%Y"), y=eps_s.values,
                                title="Annual EPS", labels={"x": "Year", "y": "EPS ($)"},
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        break
            with fc2:
                for label in ["Total Revenue", "Revenue"]:
                    if label in financials.index:
                        rev = financials.loc[label].dropna().sort_index() / 1e9
                        fig = px.bar(
                            x=rev.index.strftime("%Y"), y=rev.values,
                            title="Revenue ($B)", labels={"x": "Year", "y": "Revenue ($B)"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        break

    # ── SEC Intelligence ─────────────────────────────────────────
    st.divider()
    sec_cached = st.session_state.sec_cache.get(ticker)
    if sec_cached:
        st.subheader("SEC Intelligence (Cached)")
        a10k = sec_cached.get("analysis_10k", {}) or {}
        acred = sec_cached.get("management_credibility", {}) or {}

        s_cols = st.columns(3)
        with s_cols[0]:
            sentiment = a10k.get("overall_sentiment", "N/A")
            st.markdown(f"**10-K Sentiment:** {sentiment}")
            risks = a10k.get("key_risks", [])
            if risks:
                st.markdown("**Top Risks:**")
                for r in risks[:3]:
                    st.markdown(f"- {r}" if isinstance(r, str) else f"- {r}")
        with s_cols[1]:
            cred_score = acred.get("credibility_score", "N/A")
            st.markdown(f"**Credibility Score:** {cred_score}")
        with s_cols[2]:
            red_flags = a10k.get("red_flags", [])
            if red_flags:
                for rf in red_flags[:3]:
                    st.warning(rf if isinstance(rf, str) else str(rf))
    else:
        st.caption("SEC analysis not cached. Run from SEC Intelligence page.")

    # ── Deep Dive Results ────────────────────────────────────────
    dd_result = st.session_state.deep_dive_cache.get(ticker)
    if dd_result and "error" not in dd_result:
        st.divider()
        st.subheader("Deep Dive Results")
        thesis = dd_result.get("investment_thesis", {})
        rec = thesis.get("recommendation", "N/A")
        st.markdown(
            f'{_rec_badge(rec)}&ensp;*{thesis.get("one_line_thesis", "")}*',
            unsafe_allow_html=True,
        )

        # Bull / Base / Bear
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bull = thesis.get("bull_case", {})
            st.markdown(f"**Bull Case** — ${bull.get('target_price', 'N/A')}")
            st.caption(bull.get("thesis", ""))
        with bc2:
            base_case = thesis.get("base_case", {})
            st.markdown(f"**Base Case** — ${base_case.get('target_price', 'N/A')}")
            st.caption(base_case.get("thesis", ""))
        with bc3:
            bear = thesis.get("bear_case", {})
            st.markdown(f"**Bear Case** — ${bear.get('target_price', 'N/A')}")
            st.caption(bear.get("thesis", ""))

    # ── Rate Stress Detail ───────────────────────────────────────
    if stress_data and stress_data.get("scenarios"):
        st.divider()
        st.subheader("Rate Stress Detail")
        scenarios = stress_data["scenarios"]
        stress_cols = st.columns(4)
        for i, (key, label) in enumerate([
            ("normalized", "Normalized (9%)"),
            ("sustained_high", "Sustained High (12%)"),
            ("severe", "Severe (14%)"),
        ]):
            sc_data = scenarios.get(key, {})
            with stress_cols[i + 1]:
                sc_iv = sc_data.get("stress_intrinsic_value")
                sc_mos = sc_data.get("stress_margin_of_safety")
                st.metric(
                    label,
                    f"${sc_iv:.2f}" if sc_iv else "N/A",
                    delta=f"{sc_mos:.1f}% MoS" if sc_mos is not None else None,
                )

        with stress_cols[0]:
            st.metric(
                "Base Case (10%)",
                f"${iv:.2f}" if iv else "N/A",
                delta=f"{mos:.1f}% MoS" if mos is not None else None,
            )

    # ── Action Buttons ───────────────────────────────────────────
    st.divider()
    ab1, ab2, ab3, ab4 = st.columns(4)

    with ab1:
        on_wl = ticker in st.session_state.watchlist
        if on_wl:
            if st.button("On Watchlist", key="popup_wl_remove"):
                st.session_state.watchlist.remove(ticker)
                _save_watchlist()
                st.rerun()
        else:
            if st.button("Add to Watchlist", key="popup_wl_add"):
                st.session_state.watchlist.append(ticker)
                _save_watchlist()
                st.rerun()

    with ab2:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if dd_result and "error" not in dd_result:
            if st.button("Refresh Deep Dive", key="popup_dd_refresh", disabled=not api_key):
                st.session_state.selected_ticker = ticker
                _navigate_to("Stock Detail")
                st.session_state.show_popup = False
                st.rerun()
        else:
            if st.button("Run Deep Dive", key="popup_dd_run", disabled=not api_key):
                st.session_state.selected_ticker = ticker
                _navigate_to("Stock Detail")
                st.session_state.show_popup = False
                st.rerun()

    with ab3:
        if st.button("View Full Detail", key="popup_full_detail"):
            st.session_state.selected_ticker = ticker
            _navigate_to("Stock Detail")
            st.session_state.show_popup = False
            st.rerun()

    with ab4:
        if st.button("Close", key="popup_close_bottom"):
            st.session_state.show_popup = False
            st.session_state.popup_ticker = None
            st.rerun()

    st.divider()


# ── Popup trigger (renders at top of every page) ─────────────────────────────

if st.session_state.show_popup and st.session_state.popup_ticker:
    with st.container(border=True):
        render_ticker_popup(st.session_state.popup_ticker)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: SCREENER
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "Screener":

    # Scenario banner
    if st.session_state.scenario_active and st.session_state.scenario_result:
        summary = st.session_state.scenario_result.get("scenario_summary", "")
        conf = st.session_state.scenario_result.get("confidence", "")
        col_b1, col_b2 = st.columns([5, 1])
        with col_b1:
            st.info(f"**Active Scenario:** {summary} (Confidence: {conf})")
        with col_b2:
            if st.button("Clear Scenario"):
                st.session_state.scenario_active = False
                st.session_state.scenario_result = None
                st.session_state.scenario_df = None
                st.rerun()

    # Superforecasting active scenario banner
    try:
        _active_sf = db_get_all_scenarios(status="active")
        if _active_sf:
            _sf_names = [s["title"] for s in _active_sf[:3]]
            _sf_extra = f" (+{len(_active_sf) - 3} more)" if len(_active_sf) > 3 else ""
            st.info(
                f"**{len(_active_sf)} active scenario(s):** "
                f"{', '.join(_sf_names)}{_sf_extra}. "
                f"View in Superforecasting tab."
            )
    except Exception:
        pass  # DB not initialized yet — skip silently

    st.header("S&P 500 Stock Screener")

    # ── Scenario Intelligence bar (prominent entry to Superforecasting) ──
    _sf_bar_cols = st.columns([5, 1])
    with _sf_bar_cols[0]:
        _sf_quick = st.text_input(
            "Scenario Intelligence",
            value="",
            key="sf_quick_input",
            placeholder="What if tariffs increase 25%?  China invades Taiwan?  Fed cuts to 0%?  (or type a ticker like AAPL)",
            label_visibility="collapsed",
        )
    with _sf_bar_cols[1]:
        _sf_go = st.button("Analyze", type="primary", use_container_width=True)

    if (_sf_go or _sf_quick) and _sf_quick.strip():
        _sf_text = _sf_quick.strip()

        # Check if input is a single ticker symbol — open popup instead of scenario
        _all_tickers_set = set()
        if st.session_state.all_data:
            _all_tickers_set = set(st.session_state.all_data.keys())
        _sf_upper = _sf_text.upper().strip("$ ")
        if len(_sf_upper.split()) == 1 and _sf_upper.isalpha() and len(_sf_upper) <= 5 and _sf_upper in _all_tickers_set:
            _open_popup(_sf_upper)
            st.rerun()
        else:
            # Seed the superforecasting intake with the user's scenario text
            st.session_state.sf_intake_conversation = [
                {"role": "user", "content": _sf_text}
            ]
            st.session_state.sf_current_brief = {}
            st.session_state.sf_intake_complete = False
            st.session_state.sf_scenario_result = None
            st.session_state.sf_current_scenario_id = None
            _navigate_to("Superforecasting")
            st.rerun()

    # ── Search bar ───────────────────────────────────────────────────
    search_term = st.text_input(
        "Search by ticker or company name",
        value=st.session_state.get("screener_search_term", ""),
        key="screener_search_input",
        placeholder="e.g. AMZN or Amazon...",
    )
    st.session_state.screener_search_term = search_term

    # Quick filter buttons — row 1
    qf_cols = st.columns(6)
    with qf_cols[0]:
        show_all = st.button("All Stocks", use_container_width=True)
    with qf_cols[1]:
        show_above = st.button("Above Bar Only", use_container_width=True)
    with qf_cols[2]:
        show_buy = st.button("Buy Candidates (base)", use_container_width=True,
                             help="Above Bar + MoS > 20% (base case only)")
    with qf_cols[3]:
        show_under = st.button("Undervalued", use_container_width=True,
                               help="MoS > 15%")
    with qf_cols[4]:
        show_ai = st.button("AI Opportunity", use_container_width=True,
                            help="Tech + AI-exposed stocks with strong scores")
    with qf_cols[5]:
        show_dd_buy = st.button("DD Buy/Strong Buy", use_container_width=True,
                                help="Filter to Deep Dive Buy or Strong Buy only")

    # Quick filter buttons — row 2 (stress-focused)
    qf_cols2 = st.columns(6)
    with qf_cols2[0]:
        show_true_buy = st.button("True Buy Candidates", use_container_width=True,
                                  help="Quality + undervalued at stress rates")
    with qf_cols2[1]:
        show_stress_under = st.button("Stress Undervalued", use_container_width=True,
                                      help="Cheap even under pessimistic rate assumptions")
    with qf_cols2[2]:
        show_rate_ben = st.button("Rate Beneficiaries", use_container_width=True,
                                  help="Companies that get more valuable as rates rise")
    with qf_cols2[3]:
        show_rate_dep = st.button("Rate Dependent", use_container_width=True,
                                  help="Investment case depends on rate normalization")
    with qf_cols2[4]:
        wl_count = len(st.session_state.watchlist)
        show_watchlist = st.button(
            f"Watchlist ({wl_count})", use_container_width=True,
            help="Show only watchlisted tickers",
        )

    # Determine active filter
    if "quick_filter" not in st.session_state:
        st.session_state.quick_filter = "all"

    if show_all:
        st.session_state.quick_filter = "all"
    elif show_above:
        st.session_state.quick_filter = "above"
    elif show_buy:
        st.session_state.quick_filter = "buy"
    elif show_under:
        st.session_state.quick_filter = "under"
    elif show_ai:
        st.session_state.quick_filter = "ai"
    elif show_dd_buy:
        st.session_state.quick_filter = "dd_buy"
    elif show_true_buy:
        st.session_state.quick_filter = "true_buy"
    elif show_stress_under:
        st.session_state.quick_filter = "stress_under"
    elif show_rate_ben:
        st.session_state.quick_filter = "rate_ben"
    elif show_rate_dep:
        st.session_state.quick_filter = "rate_dep"
    elif show_watchlist:
        st.session_state.quick_filter = "watchlist"

    # Filter sidebar
    with st.expander("Filters", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            sectors = sorted(master["sector"].dropna().unique())
            sel_sectors = st.multiselect("Sector", sectors, default=sectors,
                                         key="filter_sectors")
        with fc2:
            sel_ratings = st.multiselect(
                "Quality Rating",
                ["Above Bar", "At Bar", "Below Bar"],
                default=["Above Bar", "At Bar", "Below Bar"],
                key="filter_ratings",
            )
        fc3, fc4 = st.columns(2)
        with fc3:
            min_mos = st.slider("Min Margin of Safety %", -100, 100, -100,
                                key="filter_mos")
        with fc4:
            min_score = st.slider("Min Composite Score", 0, 100, 0,
                                  key="filter_score")
        dd_only = st.checkbox("Deep Dive analyzed only", key="filter_dd_only")

    # Apply filters
    df = master.copy()

    # Attach deep dive data to df
    dd_cache = st.session_state.deep_dive_cache
    dd_recs = {}
    for t, dd in dd_cache.items():
        thesis = dd.get("investment_thesis", {})
        dd_recs[t] = thesis.get("recommendation", "")
    df["dd_rec"] = df["ticker"].map(dd_recs).fillna("")
    df["has_dd"] = df["dd_rec"] != ""

    # Use scenario-adjusted values if active
    if st.session_state.scenario_active and st.session_state.scenario_df is not None:
        sdf = st.session_state.scenario_df
        df = df.merge(
            sdf[["ticker", "scenario_intrinsic_value", "scenario_margin_of_safety",
                 "scenario_impact", "scenario_multiplier_applied"]],
            on="ticker", how="left",
        )

    df = df[df["sector"].isin(sel_sectors)]
    df = df[df["quality_rating"].isin(sel_ratings)]

    mos_col = "scenario_margin_of_safety" if (
        st.session_state.scenario_active and "scenario_margin_of_safety" in df.columns
    ) else "margin_of_safety"

    df = df[df[mos_col].fillna(-999) >= min_mos]
    df = df[df["composite_score"].fillna(0) >= min_score]
    if dd_only:
        df = df[df["has_dd"]]

    # Apply search filter
    if search_term.strip():
        s = search_term.strip().upper()
        df = df[
            df["ticker"].str.upper().str.contains(s, na=False)
            | df["company"].str.upper().str.contains(s, na=False)
        ]

    # Quick filter overrides
    qf = st.session_state.quick_filter
    if qf == "above":
        df = df[df["quality_rating"] == "Above Bar"]
    elif qf == "buy":
        df = df[(df["quality_rating"] == "Above Bar") & (df[mos_col].fillna(-999) > 20)]
    elif qf == "under":
        df = df[df[mos_col].fillna(-999) > 15]
    elif qf == "ai":
        ai_sectors = ["Information Technology", "Communication Services"]
        df = df[df["sector"].isin(ai_sectors) & (df["composite_score"].fillna(0) >= 50)]
    elif qf == "dd_buy":
        df = df[df["dd_rec"].isin(["Buy", "Strong Buy"])]
    elif qf == "true_buy":
        df = df[
            (df["quality_rating"] == "Above Bar")
            & (df[mos_col].fillna(-999) > 20)
            & (df["stress_mos"].fillna(-999) > 0)
            & (~df["stress_verdict"].isin(["Earnings Impaired at Stress Rates", "Insufficient Data"]))
        ]
    elif qf == "stress_under":
        df = df[df["stress_verdict"] == "Undervalued at Stress Rates"]
    elif qf == "rate_ben":
        df = df[df["stress_status"] == "Rate_Beneficiary"]
    elif qf == "rate_dep":
        df = df[df["stress_verdict"] == "Overvalued at Stress Rates"]
    elif qf == "watchlist":
        df = df[df["ticker"].isin(st.session_state.watchlist)]

    # Build display table
    display_cols = ["ticker", "company", "sector", "current_price"]

    if st.session_state.scenario_active and "scenario_intrinsic_value" in df.columns:
        df["display_iv"] = df["scenario_intrinsic_value"].fillna(df["intrinsic_value"])
        df["display_mos"] = df["scenario_margin_of_safety"].fillna(df["margin_of_safety"])
        display_cols += ["display_iv", "display_mos", "scenario_multiplier_applied"]
    else:
        df["display_iv"] = df["intrinsic_value"]
        df["display_mos"] = df["margin_of_safety"]
        display_cols += ["display_iv", "display_mos"]

    # Stress columns (always present after stress engine runs)
    if "stress_iv" in df.columns:
        display_cols += ["stress_iv", "stress_mos", "stress_verdict", "rate_status"]

    # Add deep dive rec column if any exist
    has_any_dd = df["has_dd"].any()
    if has_any_dd:
        display_cols += ["dd_rec"]

    display_cols += [
        "quality_rating", "composite_score", "sector_rank",
        "quality_score", "growth_score", "value_score", "sentiment_score",
    ]

    rename_map = {
        "display_iv": "Intrinsic Value",
        "display_mos": "MoS %",
        "current_price": "Price",
        "stress_iv": "Stress IV",
        "stress_mos": "Stress MoS",
        "stress_verdict": "Stress Verdict",
        "rate_status": "Rate Status",
        "quality_rating": "Rating",
        "composite_score": "Score",
        "sector_rank": "Rank",
        "quality_score": "Q",
        "growth_score": "G",
        "value_score": "V",
        "sentiment_score": "S",
        "dd_rec": "Deep Dive",
    }
    if "scenario_multiplier_applied" in display_cols:
        rename_map["scenario_multiplier_applied"] = "Scenario"

    table_df = df[display_cols].rename(columns=rename_map)
    table_df = table_df.sort_values("Score", ascending=False)

    # Format
    table_df["Price"] = table_df["Price"].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
    )
    table_df["Intrinsic Value"] = table_df["Intrinsic Value"].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
    )
    table_df["MoS %"] = table_df["MoS %"].apply(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )
    table_df["Rating"] = table_df["Rating"].apply(
        lambda x: _rating_badge(x) if pd.notna(x) else ""
    )
    if "Stress IV" in table_df.columns:
        table_df["Stress IV"] = table_df["Stress IV"].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
        table_df["Stress MoS"] = table_df["Stress MoS"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        table_df["Stress Verdict"] = table_df["Stress Verdict"].fillna("N/A")
        table_df["Rate Status"] = table_df["Rate Status"].fillna("N/A")
    if "Scenario" in table_df.columns:
        table_df["Scenario"] = table_df["Scenario"].apply(
            lambda x: f"{x:.2f}x" if pd.notna(x) else ""
        )

    table_df = _dedup_df(table_df)
    event = st.dataframe(
        table_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Navigate to Stock Detail when a row is clicked
    if event.selection.rows:
        _sel_idx = event.selection.rows[0]
        _sel_ticker = table_df.iloc[_sel_idx]["ticker"]
        st.session_state.selected_ticker = _sel_ticker
        _navigate_to("Stock Detail")
        st.rerun()

    st.caption(f"Showing {len(table_df)} of {len(master)} stocks  — click any row to view detail")

    # Single match — quick navigate
    if len(df) == 1:
        only_ticker = df["ticker"].iloc[0]
        if st.button(f"View Detail → {only_ticker}", type="primary", key="search_go_detail"):
            st.session_state.selected_ticker = only_ticker
            _navigate_to("Stock Detail")
            st.rerun()

    # ── Deep Dive Runner ─────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Build full ticker list from unfiltered master for dropdowns
    _all_tickers_sorted = sorted(master["ticker"].tolist())
    _ticker_label_map = {}
    for _t in _all_tickers_sorted:
        _tdata = st.session_state.all_data.get(_t, {})
        _tinfo = _tdata.get("info", {})
        _comp = _tdata.get("company", _tinfo.get("longName", _tinfo.get("shortName", "")))
        _ticker_label_map[_t] = f"{_t} — {_comp}" if _comp else _t
    _ticker_labels = [_ticker_label_map[_t] for _t in _all_tickers_sorted]

    with st.expander("🔬 Run Deep Dive Analysis", expanded=False):
        if master is None or master.empty:
            st.warning("Stock data not yet loaded — please wait for the screener to initialize.")
        else:
            dd_mode = st.radio(
                "Analyze:",
                ["Individual stock", "Select stocks", "Buy candidates"],
                horizontal=True,
                key="dd_batch_mode",
            )

            dd_targets = []
            if dd_mode == "Individual stock":
                dd_pick_label = st.selectbox(
                    "Select stock",
                    options=_ticker_labels,
                    index=None,
                    key="dd_individual_pick",
                    placeholder="Type ticker or company name...",
                )
                if dd_pick_label:
                    dd_pick = dd_pick_label.split(" — ")[0].strip()
                    dd_targets = [dd_pick]
            elif dd_mode == "Select stocks":
                dd_picks_labels = st.multiselect(
                    "Select stocks (up to 20)",
                    options=_ticker_labels,
                    max_selections=20,
                    key="dd_multi_picks",
                )
                dd_targets = [lbl.split(" — ")[0].strip() for lbl in dd_picks_labels]
            else:  # Buy candidates
                _q75 = master["composite_score"].quantile(0.75)
                buy_df = master[
                    (master["quality_rating"] == "Above Bar")
                    & (master["margin_of_safety"].fillna(-999) > 0)
                    & (master["composite_score"].fillna(0) >= _q75)
                ].sort_values("composite_score", ascending=False)
                dd_targets = buy_df["ticker"].tolist()
                st.caption(
                    f"{len(dd_targets)} stocks qualify as Buy Candidates "
                    f"(Above Bar quality + positive margin of safety + top 25% composite score)"
                )

            # Show which already have deep dives cached
            dd_cache = st.session_state.deep_dive_cache
            already_done = [t for t in dd_targets if t in dd_cache]
            need_run = [t for t in dd_targets if t not in dd_cache]

            if dd_targets:
                info_parts = []
                if need_run:
                    est_cost = len(need_run) * _API_COSTS["deep_dive"]
                    est_time = len(need_run) * 75
                    info_parts.append(
                        f"**{len(need_run)}** to analyze (~${est_cost:.2f}, ~{est_time // 60}m {est_time % 60}s)"
                    )
                if already_done:
                    info_parts.append(f"**{len(already_done)}** already cached")
                st.info(" | ".join(info_parts))

            # Warning for large batches
            if len(need_run) > 20:
                est_cost = len(need_run) * _API_COSTS["deep_dive"]
                st.warning(
                    f"This will analyze {len(need_run)} stocks at an estimated cost "
                    f"of ${est_cost:.2f}. Use the buttons below to confirm."
                )

            bc1, bc2 = st.columns(2)
            with bc1:
                dd_run_batch = st.button(
                    f"Run Deep Dive ({len(need_run)} stock{'s' if len(need_run) != 1 else ''})",
                    type="primary",
                    disabled=not need_run or not api_key,
                    key="dd_batch_run",
                )
            with bc2:
                dd_force_all = st.button(
                    f"Re-run All ({len(dd_targets)} stock{'s' if len(dd_targets) != 1 else ''})",
                    disabled=not dd_targets or not api_key,
                    key="dd_batch_rerun",
                )

            if dd_run_batch or dd_force_all:
                run_list = dd_targets if dd_force_all else need_run
                progress = st.progress(0, text="Starting deep dive batch...")
                succeeded = 0
                failed = 0
                for i, t in enumerate(run_list):
                    progress.progress(
                        (i + 1) / len(run_list),
                        text=f"Analyzing {t} ({i + 1}/{len(run_list)})...",
                    )
                    result = _run_dd_for_ticker(t, api_key, force_refresh=dd_force_all)
                    if result:
                        succeeded += 1
                    else:
                        failed += 1

                progress.empty()
                msg = f"Completed deep dive for {succeeded} stock{'s' if succeeded != 1 else ''}."
                if failed:
                    msg += f" {failed} failed."
                st.success(msg)
                st.session_state.selected_for_deepdive = run_list
                st.session_state.dd_show_inline = run_list[0] if len(run_list) == 1 else None
                st.rerun()

            # Debug info
            if os.environ.get("DEBUG", "").lower() == "true":
                with st.expander("🔧 Debug Info", expanded=False):
                    st.text(f"Total stocks loaded: {len(master)}")
                    st.text(f"Columns: {master.columns.tolist()}")
                    st.text(f"Quality rating values:\n{master['quality_rating'].value_counts().to_string()}")
                    mos_s = master["margin_of_safety"].dropna()
                    st.text(f"Margin of safety range: {mos_s.min():.1f} to {mos_s.max():.1f}")
                    _q75_dbg = master["composite_score"].quantile(0.75)
                    _bc_dbg = master[
                        (master["quality_rating"] == "Above Bar")
                        & (master["margin_of_safety"].fillna(-999) > 0)
                        & (master["composite_score"].fillna(0) >= _q75_dbg)
                    ]
                    st.text(f"Buy candidates found: {len(_bc_dbg)}")

    # ── Inline Deep Dive Result (single stock) ──────────────────────────
    _dd_inline_ticker = st.session_state.get("dd_show_inline")
    if _dd_inline_ticker and _dd_inline_ticker in st.session_state.deep_dive_cache:
        _dd_inline = st.session_state.deep_dive_cache[_dd_inline_ticker]
        _comp_name = st.session_state.all_data.get(_dd_inline_ticker, {}).get("company", _dd_inline_ticker)
        _t_val = st.session_state.valuations.get(_dd_inline_ticker)
        st.success(f"Deep Dive Complete — {_dd_inline_ticker} ({_comp_name})")
        _render_dd_inline(_dd_inline, _dd_inline_ticker, val=_t_val)
        vc1, vc2 = st.columns(2)
        with vc1:
            if st.button("View Full Detail →", key="dd_inline_go_detail", type="primary"):
                st.session_state.selected_ticker = _dd_inline_ticker
                _navigate_to("Stock Detail")
                st.session_state.dd_show_inline = None
                st.rerun()
        with vc2:
            if st.button("Dismiss", key="dd_inline_dismiss"):
                st.session_state.dd_show_inline = None
                st.rerun()
        st.divider()

    # ── Deep Dive Results (quick access) ────────────────────────────────
    _dd_analyzed = [t for t in df["ticker"].tolist() if t in st.session_state.deep_dive_cache]
    if _dd_analyzed:
        with st.expander(f"📊 Deep Dive Results ({len(_dd_analyzed)} analyzed)", expanded=True):
            for _t in _dd_analyzed:
                _dd = st.session_state.deep_dive_cache[_t]
                _thesis = _dd.get("investment_thesis", {})
                _rec = _thesis.get("recommendation", "N/A")
                _one_liner = _thesis.get("one_line_thesis", "")
                _comp = st.session_state.all_data.get(_t, {}).get("company", _t)
                rc1, rc2 = st.columns([4, 1])
                with rc1:
                    st.markdown(
                        f"{_rec_badge(_rec)} **{_t}** — {_comp}  \n"
                        f"*{_one_liner}*",
                        unsafe_allow_html=True,
                    )
                with rc2:
                    if st.button("View Detail →", key=f"dd_view_{_t}"):
                        st.session_state.selected_ticker = _t
                        _navigate_to("Stock Detail")
                        st.rerun()

    # Ticker selection for detail view
    sel = st.selectbox(
        "Select a stock for detail view:",
        options=df["ticker"].tolist(),
        index=None,
        key="screener_select",
        placeholder="Type or select a ticker...",
    )
    if sel:
        st.session_state.selected_ticker = sel
        _navigate_to("Stock Detail")
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: STOCK DETAIL (Tabbed)
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "Stock Detail":

    if st.button("← Back to Screener"):
        _navigate_to("Screener")
        st.rerun()

    ticker = st.session_state.selected_ticker
    if not ticker or ticker not in st.session_state.all_data:
        st.warning("Select a stock from the Screener first.")
        sel = st.selectbox(
            "Or pick one:",
            sorted(st.session_state.all_data.keys()),
            index=None,
            key="detail_select",
        )
        if sel:
            st.session_state.selected_ticker = sel
            st.rerun()
        st.stop()

    data = st.session_state.all_data[ticker]
    info = data.get("info", {})
    val = st.session_state.valuations.get(ticker, {})
    row = master[master["ticker"] == ticker].iloc[0] if ticker in master["ticker"].values else None

    # ── Header ──────────────────────────────────────────────────────
    company = data.get("company", info.get("shortName", ticker))
    sector = data.get("sector", info.get("sector", ""))
    price = val.get("current_price")
    iv = val.get("intrinsic_value")
    mos = val.get("margin_of_safety")
    rating = row["quality_rating"] if row is not None else "N/A"

    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    with h1:
        st.title(f"{company} ({ticker})")
        st.caption(sector)
    with h2:
        st.metric("Price", f"${price:.2f}" if price else "N/A")
    with h3:
        st.metric("Intrinsic Value", f"${iv:.2f}" if iv else "N/A")
    with h4:
        if mos is not None:
            delta_color = "normal" if mos > 0 else "inverse"
            st.metric("Margin of Safety", f"{mos:.1f}%",
                      delta=f"{mos:.1f}%", delta_color=delta_color)
        else:
            st.metric("Margin of Safety", val.get("error", "N/A"))

    # ── Data quality badge ──────────────────────────────────────
    xbrl_badge = None
    xbrl_this = st.session_state.xbrl_data.get(ticker) if st.session_state.xbrl_loaded else None
    if xbrl_this:
        xbrl_badge = get_data_quality_badge(ticker, xbrl_this)

    badge_html = f"**Quality Rating:** {_rating_badge(rating)}"
    if xbrl_badge:
        badge_html += (
            f' &nbsp; <span style="background:{xbrl_badge["color"]};color:white;'
            f'padding:2px 8px;border-radius:4px;font-size:0.85em">'
            f'{xbrl_badge["label"]}</span>'
        )
    st.markdown(badge_html, unsafe_allow_html=True)

    # ── Deep Dive Button ─────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    dd_result = st.session_state.deep_dive_cache.get(ticker)
    # Also check file cache
    if not dd_result:
        dd_result = load_cached_deep_dive(ticker)
        if dd_result:
            st.session_state.deep_dive_cache[ticker] = dd_result

    dd_col1, dd_col2 = st.columns([3, 2])
    with dd_col1:
        if dd_result and "error" not in dd_result:
            ts = dd_result.get("analysis_timestamp", "")
            rec = dd_result.get("investment_thesis", {}).get("recommendation", "N/A")
            st.markdown(
                f"**Deep Dive:** {_rec_badge(rec)} — Last analyzed: {ts[:10]}",
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"{estimate_deep_dive_cost()} | ~60-90 seconds")
    with dd_col2:
        if dd_result and "error" not in dd_result:
            dd_refresh = st.button("🔄 Refresh", key="dd_refresh")
        else:
            dd_refresh = False
        dd_run = st.button("🔬 Run Deep Dive Analysis", type="primary",
                           disabled=not api_key, key="dd_run")

    if dd_run or dd_refresh:
        progress_bar = st.progress(0, text="Starting deep dive analysis...")
        status_text = st.empty()

        def _dd_progress(step_idx, step_label):
            n = len(DD_STEPS)
            frac = min(1.0, max(0.0, (step_idx + 1) / n))
            label = DD_STEPS[step_idx] if step_idx < n else step_label
            try:
                progress_bar.progress(frac, text=f"Step {step_idx + 1}/{n}: {label}")
                status_text.text(f"Step {step_idx + 1}/{n}: {label}")
            except Exception:
                pass  # Progress display error should never kill the analysis

        # Build inputs for deep dive
        scores_dict = {}
        if row is not None:
            scores_dict = {
                "quality_score": row.get("quality_score", 50),
                "growth_score": row.get("growth_score", 50),
                "value_score": row.get("value_score", 50),
                "sentiment_score": row.get("sentiment_score", 50),
                "composite_score": row.get("composite_score", 50),
            }

        ratings_df = st.session_state.ratings_df
        qr_dict = {}
        if ratings_df is not None:
            rr_rows = ratings_df[ratings_df["ticker"] == ticker]
            if not rr_rows.empty:
                qr_dict = rr_rows.iloc[0].to_dict()

        dd_result = run_deep_dive(
            ticker=ticker,
            company_name=company,
            current_scores=scores_dict,
            current_valuation=val,
            current_quality_rating=qr_dict,
            sector=sector,
            sector_medians={},
            api_key=api_key,
            discount_rate=st.session_state.discount_rate / 100,
            on_progress=_dd_progress,
            force_refresh=dd_refresh,
        )
        progress_bar.empty()
        status_text.empty()

        if "error" in dd_result:
            st.error(dd_result["error"])
        else:
            _track_cost("deep_dive")
            st.session_state.deep_dive_cache[ticker] = dd_result
            st.rerun()

    # ── Deep Dive History Log ─────────────────────────────────────
    dd_history = load_deep_dive_history(ticker)
    if dd_history:
        with st.expander(f"📋 Deep Dive History ({len(dd_history)} analyses)", expanded=False):
            for i, entry in enumerate(dd_history):
                ts = entry["timestamp"][:16].replace("T", " ")
                rec = entry["recommendation"]
                thesis = entry["one_line_thesis"]
                hc1, hc2 = st.columns([5, 1])
                with hc1:
                    st.markdown(
                        f"{_rec_badge(rec)} **{ts}** — *{thesis}*",
                        unsafe_allow_html=True,
                    )
                with hc2:
                    if st.button("View", key=f"dd_hist_{i}"):
                        hist_data = load_deep_dive_by_path(entry["file_path"])
                        if hist_data:
                            st.session_state.deep_dive_cache[ticker] = hist_data
                            st.rerun()

    # ── Tabbed Detail View ──────────────────────────────────────────
    tab_names = ["Valuation", "Rate Stress", "Scores", "Quality Rating", "SEC Intelligence", "Raw Data"]
    if dd_result and "error" not in dd_result:
        tab_names.append("Deep Dive")
    tabs = st.tabs(tab_names)
    tab_val = tabs[0]
    tab_stress = tabs[1]
    tab_scores = tabs[2]
    tab_quality = tabs[3]
    tab_sec = tabs[4]
    tab_raw = tabs[5]
    tab_dd = tabs[6] if len(tabs) > 6 else None

    # ── TAB: Valuation ──────────────────────────────────────────────
    with tab_val:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Valuation Breakdown")
            if val.get("error"):
                st.warning(val["error"])
            else:
                st.markdown(f"**Base EPS:** ${val.get('base_eps', 0):.2f}")
                st.markdown(f"**Growth Rate:** {val.get('growth_rate_used', 0):.1%}")
                st.markdown(f"**Discount Rate:** {st.session_state.discount_rate}%")

                s1 = val.get("stage1_values", [])
                s2 = val.get("stage2_values", [])
                all_years = s1 + s2
                if all_years:
                    yr_df = pd.DataFrame(all_years)
                    yr_df["eps"] = yr_df["eps"].apply(lambda x: f"${x:.2f}")
                    yr_df["pv"] = yr_df["pv"].apply(lambda x: f"${x:.2f}")
                    yr_df.columns = ["Year", "Projected EPS", "Present Value"]
                    safe_dataframe(yr_df, hide_index=True, use_container_width=True)

                tv = val.get("terminal_value")
                if tv:
                    st.markdown(f"**Terminal Value (PV):** ${tv:.2f}")
                st.markdown(f"**Intrinsic Value:** ${iv:.2f}" if iv else "N/A")

        with col_right:
            st.subheader("Sensitivity Table")
            if val.get("error"):
                st.info("Sensitivity analysis requires profitable earnings.")
            else:
                base_eps = val.get("base_eps", 0)
                base_growth = val.get("growth_rate_used", 0.05)

                # Growth rates to test
                growth_rates = [
                    max(-0.05, base_growth - 0.04),
                    max(-0.05, base_growth - 0.02),
                    base_growth,
                    min(0.25, base_growth + 0.02),
                    min(0.25, base_growth + 0.04),
                ]
                # Discount rates to test
                disc_rates = [0.08, 0.10, 0.12, 0.14]

                sens_data = {}
                for dr_val in disc_rates:
                    col_vals = []
                    for gr in growth_rates:
                        eps = base_eps
                        total_pv = 0
                        for yr in range(1, 6):
                            eps *= (1 + gr)
                            total_pv += eps / (1 + dr_val) ** yr
                        s2_rate = min(gr, 0.08)
                        for yr in range(6, 11):
                            eps *= (1 + s2_rate)
                            total_pv += eps / (1 + dr_val) ** yr
                        terminal_pv = (eps * 15) / (1 + dr_val) ** 10
                        total_pv += terminal_pv
                        col_vals.append(round(total_pv, 2))
                    sens_data[f"{dr_val:.0%} DR"] = col_vals

                sens_df = pd.DataFrame(
                    sens_data,
                    index=[f"{g:.1%} Growth" for g in growth_rates],
                )
                sens_df = _dedup_df(sens_df)

                # Highlight cells where IV > current price
                def _highlight_undervalued(val_cell):
                    try:
                        if price and val_cell > price:
                            return "background-color: #d4edda"
                    except (TypeError, ValueError):
                        pass
                    return ""

                try:
                    safe_dataframe(
                        sens_df.style.map(_highlight_undervalued),
                        use_container_width=True,
                    )
                except Exception:
                    safe_dataframe(sens_df, use_container_width=True)
                st.caption("Green = intrinsic value > current price")

    # ── TAB: Rate Stress ──────────────────────────────────────────────
    with tab_stress:
        stress_data = None
        if st.session_state.stress_valuations:
            stress_data = st.session_state.stress_valuations.get(ticker)

        if stress_data is None or not stress_data.get("scenarios"):
            st.info("Rate stress data not available. Ensure data is loaded.")
        else:
            scenarios = stress_data["scenarios"]
            base_iv_stress = stress_data.get("base_intrinsic_value")
            base_mos_stress = stress_data.get("base_margin_of_safety")
            rate_risk = stress_data.get("rate_risk_rating", "Unknown")
            verdict = stress_data.get("stress_verdict", "N/A")

            # ── Section 1: Summary metric cards ───────────────────────
            st.subheader("Valuation Across Rate Environments")

            def _stress_color(mos_val):
                if mos_val is None:
                    return "#888888"
                if mos_val > 20:
                    return "#2e7d32"
                if mos_val >= 0:
                    return "#f9a825"
                return "#c62828"

            def _status_emoji(status):
                return {
                    "Rate_Beneficiary": "🟢",
                    "Resilient": "🟢",
                    "Moderately_Impaired": "🟡",
                    "Severely_Impaired": "🔴",
                    "Earnings_Eliminated": "⛔",
                }.get(status, "⚪")

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown("**Base Case (10%)**")
                st.metric("Intrinsic Value", f"${base_iv_stress:.2f}" if base_iv_stress else "N/A")
                st.metric("Margin of Safety", f"{base_mos_stress:.1f}%" if base_mos_stress is not None else "N/A")
            for col_obj, key in zip([sc2, sc3, sc4], ["normalized", "sustained_high", "severe"]):
                sc = scenarios.get(key, {})
                with col_obj:
                    st.markdown(f"**{sc.get('label', key)}**")
                    s_iv = sc.get("stress_intrinsic_value")
                    s_mos = sc.get("stress_margin_of_safety")
                    s_status = sc.get("stress_status")
                    st.metric("Intrinsic Value", f"${s_iv:.2f}" if s_iv else "N/A")
                    st.metric("Margin of Safety", f"{s_mos:.1f}%" if s_mos is not None else "N/A")
                    st.caption(f"{_status_emoji(s_status)} {(s_status or 'N/A').replace('_', ' ')}")

            # Verdict banner
            _verdict_colors = {
                "Undervalued at Stress Rates": ("#2e7d32", "#e8f5e9"),
                "Fairly Valued at Stress Rates": ("#f57f17", "#fff8e1"),
                "Overvalued at Stress Rates": ("#c62828", "#ffebee"),
                "Earnings Impaired at Stress Rates": ("#b71c1c", "#ffcdd2"),
                "Rate Beneficiary — Enhanced Value": ("#1565c0", "#e3f2fd"),
            }
            v_fg, v_bg = _verdict_colors.get(verdict, ("#333", "#f5f5f5"))
            st.markdown(
                f'<div style="padding:12px 20px;border-radius:8px;background:{v_bg};'
                f'border-left:4px solid {v_fg};margin:12px 0">'
                f'<span style="font-size:1.1em;font-weight:600;color:{v_fg}">'
                f'{verdict}</span>'
                f'<span style="margin-left:16px;color:#555">Rate Risk: {rate_risk}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # ── Section 2: Earnings + Valuation Impact ────────────────
            st.subheader("Impact Analysis")
            imp_left, imp_right = st.columns(2)

            sh = scenarios.get("sustained_high", {})
            with imp_left:
                st.markdown("**Earnings Impact (Sustained High)**")
                base_eps_val = val.get("base_eps")
                s_eps = sh.get("stress_base_eps")
                eps_impact = sh.get("earnings_impact_pct")
                total_debt = sh.get("total_debt")
                net_cash = sh.get("net_cash_position")
                cur_interest = sh.get("current_interest_expense")
                incr_interest = sh.get("incremental_interest")

                if base_eps_val and s_eps is not None:
                    st.metric("Base EPS", f"${base_eps_val:.2f}")
                    delta_str = f"{eps_impact:+.1f}%" if eps_impact is not None else "N/A"
                    st.metric("Stress-Adjusted EPS", f"${s_eps:.4f}", delta=delta_str)
                else:
                    st.info("EPS data unavailable")

                if total_debt:
                    st.markdown(f"Total Debt: **${total_debt/1e9:.1f}B**" if total_debt > 1e9
                                else f"Total Debt: **${total_debt/1e6:.0f}M**")
                if net_cash is not None:
                    label = "Net Cash" if net_cash >= 0 else "Net Debt"
                    abs_val = abs(net_cash)
                    st.markdown(f"{label}: **${abs_val/1e9:.1f}B**" if abs_val > 1e9
                                else f"{label}: **${abs_val/1e6:.0f}M**")
                if cur_interest:
                    st.markdown(f"Current Interest Expense: **${cur_interest/1e6:.0f}M**")
                if incr_interest is not None and incr_interest != 0:
                    st.markdown(f"Incremental Interest at Stress: **${incr_interest/1e6:.0f}M**")

            with imp_right:
                st.markdown("**Valuation Impact Across Scenarios**")
                impact_rows = []
                for key in ["normalized", "sustained_high", "severe"]:
                    sc = scenarios.get(key, {})
                    impact_rows.append({
                        "Scenario": sc.get("label", key),
                        "Rate": f"{sc.get('rate', 0):.0%}",
                        "Stress IV": f"${sc.get('stress_intrinsic_value', 0):.2f}" if sc.get("stress_intrinsic_value") else "N/A",
                        "IV Change": f"{sc.get('iv_change_from_base_pct', 0):+.1f}%" if sc.get("iv_change_from_base_pct") is not None else "N/A",
                        "Stress MoS": f"{sc.get('stress_margin_of_safety', 0):.1f}%" if sc.get("stress_margin_of_safety") is not None else "N/A",
                        "Status": (sc.get("stress_status") or "N/A").replace("_", " "),
                    })
                safe_dataframe(pd.DataFrame(impact_rows), hide_index=True, use_container_width=True)

            st.markdown("---")

            # ── Section 3: IV across rates chart ──────────────────────
            st.subheader("Intrinsic Value Across Discount Rates")

            base_eps_chart = val.get("base_eps")
            growth_rate_chart = val.get("growth_rate_used", 0.05)
            if growth_rate_chart is None:
                growth_rate_chart = 0.05
            growth_rate_chart = max(-0.05, min(0.25, growth_rate_chart))

            if base_eps_chart and base_eps_chart > 0:
                from rate_stress_engine import _run_two_stage_dcf

                chart_rates = [r / 100 for r in range(7, 16)]
                base_line = []
                stress_line = []

                # Get stress EPS for the sustained_high scenario to use as stress-adjusted base
                sh_eps = sh.get("stress_base_eps") or base_eps_chart

                for r in chart_rates:
                    base_line.append(_run_two_stage_dcf(base_eps_chart, growth_rate_chart, r))
                    stress_line.append(_run_two_stage_dcf(sh_eps, growth_rate_chart, r))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[r * 100 for r in chart_rates],
                    y=base_line,
                    mode="lines+markers",
                    name="Base Earnings",
                    line=dict(color="#1565c0", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=[r * 100 for r in chart_rates],
                    y=stress_line,
                    mode="lines+markers",
                    name="Stress-Adjusted Earnings",
                    line=dict(color="#c62828", width=2, dash="dash"),
                ))
                # Current price line
                if price:
                    fig.add_hline(
                        y=price, line_dash="dot", line_color="#888",
                        annotation_text=f"Current Price ${price:.2f}",
                        annotation_position="bottom right",
                    )
                    # Find breakeven rate (where stress line crosses price)
                    for i in range(len(stress_line) - 1):
                        if (stress_line[i] >= price) != (stress_line[i + 1] >= price):
                            # Linear interpolation
                            r1, r2 = chart_rates[i] * 100, chart_rates[i + 1] * 100
                            v1, v2 = stress_line[i], stress_line[i + 1]
                            breakeven = r1 + (price - v1) * (r2 - r1) / (v2 - v1)
                            fig.add_vline(
                                x=breakeven, line_dash="dot", line_color="#ff9800",
                                annotation_text=f"Breakeven {breakeven:.1f}%",
                                annotation_position="top left",
                            )
                            break

                fig.update_layout(
                    xaxis_title="Discount Rate (%)",
                    yaxis_title="Intrinsic Value ($)",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chart requires positive base EPS.")

            st.markdown("---")

            # ── Section 4: Claude narrative (on demand) ───────────────
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            cache_key = f"stress_narrative_{ticker}"

            if api_key:
                if st.button("Generate Rate Stress Narrative", key="gen_stress_narrative"):
                    with st.spinner("Generating analysis..."):
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)

                        prompt_data = {
                            "ticker": ticker,
                            "company": company,
                            "sector": sector,
                            "current_price": price,
                            "base_iv": base_iv_stress,
                            "base_mos": base_mos_stress,
                            "base_eps": val.get("base_eps"),
                            "growth_rate": val.get("growth_rate_used"),
                            "scenarios": scenarios,
                            "rate_risk_rating": rate_risk,
                            "stress_verdict": verdict,
                        }

                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=800,
                            system=(
                                "You are a value investing analyst. Write a concise rate stress "
                                "assessment for this stock. Cover: (1) how rate-sensitive the "
                                "company's earnings are, (2) how its intrinsic value holds up "
                                "under sustained high rates, (3) what discount rate would make "
                                "it fairly valued, (4) a plain-English verdict. Use specific "
                                "numbers from the data. 3-4 paragraphs max."
                            ),
                            messages=[{
                                "role": "user",
                                "content": f"Rate stress analysis data:\n{prompt_data}",
                            }],
                        )
                        narrative = response.content[0].text
                        st.session_state[cache_key] = narrative
                        st.rerun()

                if cache_key in st.session_state:
                    st.markdown(st.session_state[cache_key])
            else:
                st.info("Set ANTHROPIC_API_KEY for Claude narrative analysis.")

            # ── Section 5: XBRL Debt Trajectory ──────────────────────
            _xbrl_stress = st.session_state.xbrl_data.get(ticker) if st.session_state.xbrl_loaded else None
            if _xbrl_stress and _xbrl_stress.get("available"):
                st.markdown("---")
                st.subheader("Historical Debt Trajectory (XBRL)")
                traj = calculate_debt_trajectory(ticker, _xbrl_stress)

                if traj.get("available"):
                    traj_cols = st.columns(4)
                    with traj_cols[0]:
                        dgr = traj.get("debt_growth_rate")
                        st.metric("Debt Growth Rate",
                                  f"{dgr:.1%}" if dgr is not None else "N/A")
                    with traj_cols[1]:
                        peak = traj.get("peak_de")
                        pk_yr = traj.get("peak_de_year")
                        st.metric("Peak D/E",
                                  f"{peak:.2f} ({pk_yr})" if peak is not None else "N/A")
                    with traj_cols[2]:
                        avg_ic = traj.get("avg_interest_coverage_10yr")
                        st.metric("10yr Avg Interest Coverage",
                                  f"{avg_ic:.1f}x" if avg_ic is not None else "N/A")
                    with traj_cols[3]:
                        if traj.get("low_rate_era_flag"):
                            st.warning("Low-Rate Era Debt Loading Detected")
                        else:
                            st.success("No Low-Rate Era Debt Spike")

                    traj_chart_cols = st.columns(2)

                    with traj_chart_cols[0]:
                        # Total Debt chart
                        ds = traj.get("debt_series", {})
                        if ds:
                            d_years = sorted(ds.keys())
                            d_vals = [ds[y] / 1e9 for y in d_years]
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=d_years, y=d_vals,
                                marker_color="#e74c3c", name="Total Debt",
                            ))
                            fig.update_layout(
                                title="Total Debt History",
                                xaxis_title="Year", yaxis_title="Total Debt ($B)",
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with traj_chart_cols[1]:
                        # D/E chart
                        de_s = traj.get("de_series", {})
                        if de_s:
                            de_years = sorted(de_s.keys())
                            de_vals = [de_s[y] for y in de_years]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=de_years, y=de_vals,
                                mode="lines+markers",
                                line=dict(color="#e67e22"),
                                name="D/E",
                            ))
                            # Low-rate era shading
                            if any(2019 <= y <= 2022 for y in de_years):
                                fig.add_vrect(
                                    x0=2019.5, x1=2021.5,
                                    fillcolor="rgba(52, 152, 219, 0.1)",
                                    line_width=0,
                                )
                            fig.update_layout(
                                title="Debt/Equity Ratio History",
                                xaxis_title="Year", yaxis_title="D/E Ratio",
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Interest coverage chart
                    ic_s = traj.get("interest_coverage_series", {})
                    if ic_s:
                        ic_years = sorted(ic_s.keys())
                        ic_vals = [ic_s[y] for y in ic_years]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=ic_years, y=ic_vals,
                            mode="lines+markers",
                            line=dict(color="#27ae60"),
                            name="Interest Coverage",
                        ))
                        fig.add_hline(y=3.0, line_dash="dash", line_color="red",
                                      annotation_text="3x Threshold")
                        fig.update_layout(
                            title="Interest Coverage Ratio History",
                            xaxis_title="Year", yaxis_title="Coverage (x)",
                            showlegend=False, height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # ── Philosophical note ────────────────────────────────────
            with st.expander("About This Analysis", expanded=False):
                st.markdown("""
This stress test assumes rates stay high for the full DCF horizon — not because
that's the most likely outcome, but because it's the most dangerous one for
long-duration equity.

The question isn't "will rates come down?" (they probably will). The question is:
**"If I'm wrong about rates, am I also wrong about this stock?"**

A stock that's undervalued at stress rates is a stock where you're being paid to
take rate risk. A stock that's only cheap assuming rate normalization is a rate bet
disguised as a value investment.

*This is the difference between structural value and cyclical value — and it matters
more than most screens will ever tell you.*
""")

    # ── TAB: Scores ─────────────────────────────────────────────────
    with tab_scores:
        if row is not None:
            col_radar, col_detail = st.columns(2)

            with col_radar:
                categories = ["Quality", "Growth", "Value", "Sentiment"]
                values = [
                    row.get("quality_score", 50),
                    row.get("growth_score", 50),
                    row.get("value_score", 50),
                    row.get("sentiment_score", 50),
                ]
                fig = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    line=dict(color="#2196F3"),
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Pillar Scores",
                    height=350,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_detail:
                st.subheader("Score Details")
                st.markdown(f"**Composite Score:** {row.get('composite_score', 'N/A')}")
                st.markdown(f"**Sector Rank:** {row.get('sector_rank', 'N/A')} of {row.get('sector_total', 'N/A')}")

                with st.expander("Quality Metrics"):
                    qm = {
                        "ROE": info.get("returnOnEquity"),
                        "Net Profit Margin": info.get("profitMargins"),
                        "Debt/Equity": info.get("debtToEquity"),
                        "ROA": info.get("returnOnAssets"),
                    }
                    for k, v in qm.items():
                        if v is not None:
                            st.markdown(f"- **{k}:** {v:.2%}" if abs(v) < 10 else f"- **{k}:** {v:.1f}")
                        else:
                            st.markdown(f"- **{k}:** N/A")

                with st.expander("Growth Metrics"):
                    fwd = info.get("forwardEps")
                    trail = info.get("trailingEps")
                    st.markdown(f"- **Trailing EPS:** ${trail:.2f}" if trail else "- **Trailing EPS:** N/A")
                    st.markdown(f"- **Forward EPS:** ${fwd:.2f}" if fwd else "- **Forward EPS:** N/A")

                with st.expander("Value Metrics"):
                    vm = {
                        "Trailing P/E": info.get("trailingPE"),
                        "EV/EBITDA": info.get("enterpriseToEbitda"),
                        "Price/Book": info.get("priceToBook"),
                    }
                    for k, v in vm.items():
                        st.markdown(f"- **{k}:** {v:.1f}" if v else f"- **{k}:** N/A")

                with st.expander("Sentiment Metrics"):
                    pct_52 = (
                        (info.get("currentPrice", 0) or 0)
                        / (info.get("fiftyTwoWeekHigh", 1) or 1)
                    )
                    ma_ratio = (
                        (info.get("fiftyDayAverage", 0) or 0)
                        / (info.get("twoHundredDayAverage", 1) or 1)
                    )
                    st.markdown(f"- **% of 52-wk High:** {pct_52:.2%}")
                    st.markdown(f"- **50MA / 200MA:** {ma_ratio:.2%}")
        else:
            st.info("No score data available for this ticker.")

    # ── TAB: Quality Rating ─────────────────────────────────────────
    with tab_quality:
        ratings_row = st.session_state.ratings_df
        rr = None
        if ratings_row is not None:
            rr_df = ratings_row[ratings_row["ticker"] == ticker]
            if not rr_df.empty:
                rr = rr_df.iloc[0]

        if rr is not None:
            st.subheader("Quality Rating Breakdown")
            qr_data = {
                "Component": ["Earnings Consistency", "Debt Discipline",
                              "Dividend Quality", "Buyback Discipline"],
                "Weight": ["40%", "25%", "20%", "15%"],
                "Score": [
                    rr.get("earnings_consistency_score", "N/A"),
                    rr.get("debt_discipline_score", "N/A"),
                    rr.get("dividend_quality_score", "N/A"),
                    rr.get("buyback_score", "N/A"),
                ],
            }
            st.table(pd.DataFrame(qr_data))
            st.markdown(f"**Weighted Total:** {rr.get('rating_total', 'N/A')} → **{_rating_badge(rating)}**")

        # Historical charts — XBRL-enhanced when available
        _xbrl_ticker = st.session_state.xbrl_data.get(ticker) if st.session_state.xbrl_loaded else None
        _has_xbrl = _xbrl_ticker and _xbrl_ticker.get("available")

        if _has_xbrl:
            _xdf = _xbrl_ticker["annual_data"]
            _n_years = _xbrl_ticker.get("years_available", 0)
            st.subheader(f"Historical Data — {_n_years} Year XBRL History")

            # Recession shading helper
            def _add_recession_shading(fig, years_list):
                """Add light red shading for recession years."""
                for ry_start, ry_end in [(2007.5, 2009.5), (2019.5, 2020.5)]:
                    if any(ry_start <= y <= ry_end for y in years_list):
                        fig.add_vrect(
                            x0=ry_start, x1=ry_end,
                            fillcolor="rgba(255, 0, 0, 0.08)",
                            line_width=0,
                            annotation_text="",
                        )
                return fig

            chart_cols = st.columns(2)

            with chart_cols[0]:
                # EPS chart from XBRL
                eps_col = None
                for _c in ["eps_diluted", "eps_basic"]:
                    if _c in _xdf.columns:
                        eps_col = _c
                        break
                if eps_col:
                    _eps = _xdf[eps_col].dropna()
                    if len(_eps) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=_eps.index.tolist(),
                            y=_eps.values,
                            marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in _eps.values],
                            name="EPS",
                        ))
                        # Trend line
                        if len(_eps) >= 3:
                            z = np.polyfit(range(len(_eps)), _eps.values, 1)
                            trend = np.polyval(z, range(len(_eps)))
                            fig.add_trace(go.Scatter(
                                x=_eps.index.tolist(), y=trend,
                                mode="lines", line=dict(dash="dash", color="gray"),
                                name="Trend",
                            ))
                        fig = _add_recession_shading(fig, _eps.index.tolist())
                        fig.update_layout(
                            title=f"Earnings Per Share — {len(_eps)} Year History",
                            xaxis_title="Year", yaxis_title="EPS ($)",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with chart_cols[1]:
                # Revenue chart from XBRL
                if "revenue" in _xdf.columns:
                    _rev = _xdf["revenue"].dropna() / 1e9
                    if len(_rev) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=_rev.index.tolist(), y=_rev.values,
                            marker_color="#3498db", name="Revenue",
                        ))
                        fig = _add_recession_shading(fig, _rev.index.tolist())
                        fig.update_layout(
                            title=f"Revenue — {len(_rev)} Year History",
                            xaxis_title="Year", yaxis_title="Revenue ($B)",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            chart_cols2 = st.columns(2)

            with chart_cols2[0]:
                # D/E from XBRL
                if "debt_to_equity" in _xdf.columns:
                    _de = _xdf["debt_to_equity"].dropna()
                    if len(_de) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=_de.index.tolist(), y=_de.values,
                            mode="lines+markers", line=dict(color="#e67e22"),
                            name="D/E",
                        ))
                        # Low-rate era shading (2020-2021)
                        if any(2019 <= y <= 2022 for y in _de.index):
                            fig.add_vrect(
                                x0=2019.5, x1=2021.5,
                                fillcolor="rgba(52, 152, 219, 0.1)",
                                line_width=0,
                                annotation_text="Low-Rate Era",
                                annotation_position="top left",
                                annotation_font_size=10,
                            )
                        fig.update_layout(
                            title=f"Debt/Equity Ratio — {len(_de)} Year History",
                            xaxis_title="Year", yaxis_title="D/E",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with chart_cols2[1]:
                # FCF from XBRL
                if "free_cash_flow" in _xdf.columns:
                    _fcf = _xdf["free_cash_flow"].dropna() / 1e9
                    if len(_fcf) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=_fcf.index.tolist(), y=_fcf.values,
                            marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in _fcf.values],
                            name="FCF",
                        ))
                        fig.update_layout(
                            title=f"Free Cash Flow — {len(_fcf)} Year History",
                            xaxis_title="Year", yaxis_title="FCF ($B)",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

        else:
            # Fallback: existing 4-year yfinance charts
            st.subheader("Historical Data")

            financials = data.get("financials")
            balance_sheet = data.get("balance_sheet")
            cashflow = data.get("cashflow")
            dividends = data.get("dividends")

            chart_cols = st.columns(2)

            with chart_cols[0]:
                if financials is not None and not financials.empty:
                    ni_row_ch = None
                    for label in ["Net Income", "Net Income Common Stockholders"]:
                        if label in financials.index:
                            ni_row_ch = financials.loc[label]
                            break
                    shares = info.get("sharesOutstanding", 1)
                    if ni_row_ch is not None and shares:
                        eps_series = (ni_row_ch.dropna() / shares).sort_index()
                        fig = px.bar(
                            x=eps_series.index.strftime("%Y"),
                            y=eps_series.values,
                            title="Annual EPS",
                            labels={"x": "Year", "y": "EPS ($)"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with chart_cols[1]:
                if financials is not None and not financials.empty:
                    rev_row = None
                    for label in ["Total Revenue", "Revenue", "Operating Revenue"]:
                        if label in financials.index:
                            rev_row = financials.loc[label]
                            break
                    if rev_row is not None:
                        rev = rev_row.dropna().sort_index() / 1e9
                        fig = px.bar(
                            x=rev.index.strftime("%Y"),
                            y=rev.values,
                            title="Annual Revenue",
                            labels={"x": "Year", "y": "Revenue ($B)"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

            chart_cols2 = st.columns(2)

            with chart_cols2[0]:
                if balance_sheet is not None and not balance_sheet.empty:
                    debt_row = None
                    eq_row = None
                    for label in ["Total Debt", "Long Term Debt"]:
                        if label in balance_sheet.index:
                            debt_row = balance_sheet.loc[label]
                            break
                    for label in ["Total Stockholders Equity", "Stockholders Equity",
                                  "Common Stock Equity"]:
                        if label in balance_sheet.index:
                            eq_row = balance_sheet.loc[label]
                            break
                    if debt_row is not None and eq_row is not None:
                        common_idx = debt_row.dropna().index.intersection(eq_row.dropna().index)
                        if len(common_idx) > 0:
                            de = (debt_row[common_idx] / eq_row[common_idx]).sort_index()
                            fig = px.line(
                                x=de.index.strftime("%Y"),
                                y=de.values,
                                title="Debt/Equity Ratio",
                                labels={"x": "Year", "y": "D/E"},
                                markers=True,
                            )
                            st.plotly_chart(fig, use_container_width=True)

            with chart_cols2[1]:
                if dividends is not None and len(dividends) > 0:
                    annual_div = dividends.resample("YE").sum()
                    annual_div = annual_div[annual_div > 0]
                    if len(annual_div) > 0:
                        fig = px.bar(
                            x=annual_div.index.strftime("%Y"),
                            y=annual_div.values,
                            title="Annual Dividends/Share",
                            labels={"x": "Year", "y": "Dividends ($)"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: SEC Intelligence ───────────────────────────────────────
    with tab_sec:
        if not api_key:
            st.warning("Set ANTHROPIC_API_KEY in .env to enable SEC analysis.")
        else:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.subheader(f"SEC Filing Analysis — {ticker}")
            with col_b:
                run_sec = st.button("Run SEC Analysis", type="primary",
                                    key=f"run_sec_{ticker}")

            if run_sec:
                with st.spinner(f"Fetching SEC filings and running AI analysis for {ticker}..."):
                    sec_result = get_full_sec_analysis(ticker, api_key)
                    st.session_state.sec_cache[ticker] = sec_result
                    _track_cost("sec_analysis")

            sec_data = st.session_state.sec_cache.get(ticker)
            if sec_data:
                # Overall flag
                flag = sec_data.get("overall_sec_flag", "green")
                st.markdown(f"**Overall SEC Flag:** {_sec_flag_badge(flag)} {flag.upper()}")

                if sec_data.get("errors"):
                    for err in sec_data["errors"]:
                        st.warning(err)

                # Filings found
                filings_found = sec_data.get("filings_found", [])
                if filings_found:
                    with st.expander(f"Recent Filings ({len(filings_found)})"):
                        safe_dataframe(
                            pd.DataFrame(filings_found),
                            hide_index=True,
                            use_container_width=True,
                        )

                # 10-K Analysis
                analysis_10k = sec_data.get("analysis_10k")
                if analysis_10k and "error" not in analysis_10k:
                    with st.expander("10-K Narrative Analysis", expanded=True):
                        st.markdown(f"**Sentiment:** {analysis_10k.get('overall_sentiment', 'N/A')}")
                        st.markdown(f"**SEC Flag:** {_sec_flag_badge(analysis_10k.get('sec_flag', 'green'))}")

                        # Red flags
                        red_flags = analysis_10k.get("red_flags", [])
                        if red_flags:
                            st.markdown("**Red Flags:**")
                            for rf in red_flags:
                                sev = rf.get("severity", "low")
                                icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "⚪"
                                st.markdown(f"- {icon} {rf.get('flag', '')} *(Section: {rf.get('section', 'N/A')})*")

                        # Key risks
                        key_risks = analysis_10k.get("key_risks", [])
                        if key_risks:
                            st.markdown("**Key Risks:**")
                            for kr in key_risks:
                                sev = kr.get("severity", "low")
                                icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "⚪"
                                st.markdown(f"- {icon} [{kr.get('category', '')}] {kr.get('risk', '')}")

                        # Management tone
                        tone = analysis_10k.get("management_tone", {})
                        if tone:
                            st.markdown(f"**Management Confidence:** {tone.get('confidence_level', 'N/A')}")
                            hedging = tone.get("hedging_language_pct")
                            if hedging is not None:
                                st.markdown(f"**Hedging Language:** {hedging:.0%}")

                        # Forward guidance
                        guidance = analysis_10k.get("forward_guidance", {})
                        if guidance:
                            st.markdown(f"**Outlook:** {guidance.get('outlook', 'N/A')}")
                            st.markdown(f"**CapEx Signal:** {guidance.get('capex_signals', 'N/A')}")
                            inits = guidance.get("key_initiatives", [])
                            if inits:
                                for init in inits:
                                    st.markdown(f"- {init}")

                # Proxy analysis
                analysis_proxy = sec_data.get("analysis_proxy")
                if analysis_proxy and "error" not in analysis_proxy:
                    with st.expander("Proxy Statement (Governance)"):
                        gov = analysis_proxy.get("governance_quality", {})
                        st.markdown(f"**Governance Grade:** {gov.get('overall_grade', 'N/A')}")
                        st.markdown(f"**Board Independence:** {gov.get('board_independence_pct', 'N/A')}")
                        st.markdown(f"**Dual-Class Shares:** {'Yes' if gov.get('dual_class_shares') else 'No'}")

                        comp = analysis_proxy.get("executive_compensation", {})
                        if comp:
                            st.markdown(f"**Pay vs Performance:** {comp.get('pay_vs_performance_alignment', 'N/A')}")

                # 8-K analysis
                analysis_8k = sec_data.get("analysis_8k")
                if analysis_8k and "error" not in analysis_8k:
                    with st.expander("Recent 8-K Events"):
                        st.markdown(f"**Overall Signal:** {analysis_8k.get('overall_signal', 'N/A')}")
                        st.markdown(analysis_8k.get("event_summary", ""))
                        events = analysis_8k.get("material_events", [])
                        if events:
                            for ev in events:
                                impact = ev.get("impact", "neutral")
                                icon = "🟢" if impact == "positive" else "🔴" if impact == "negative" else "⚪"
                                st.markdown(
                                    f"- {icon} **{ev.get('date', '')}** [{ev.get('event_type', '')}]: "
                                    f"{ev.get('description', '')}"
                                )

                # Management credibility
                credibility = sec_data.get("management_credibility")
                if credibility and "error" not in credibility:
                    with st.expander("Management Credibility"):
                        score = credibility.get("credibility_score", "N/A")
                        st.markdown(f"**Credibility Score:** {score}/100")
                        st.markdown(f"**Flag:** {_sec_flag_badge(credibility.get('credibility_flag', 'green'))}")
                        st.markdown(credibility.get("summary", ""))

                        kept = credibility.get("promises_kept", [])
                        if kept:
                            st.markdown("**Promises Kept:**")
                            for p in kept:
                                st.markdown(f"- ✅ {p.get('promise', '')} → {p.get('outcome', '')} (Grade: {p.get('grade', 'N/A')})")

                        broken = credibility.get("promises_broken", [])
                        if broken:
                            st.markdown("**Promises Broken:**")
                            for p in broken:
                                st.markdown(f"- ❌ {p.get('promise', '')} → {p.get('outcome', '')}")

            elif not run_sec:
                st.info("Click 'Run SEC Analysis' to fetch and analyze SEC filings for this stock.")

    # ── TAB: Raw Data ───────────────────────────────────────────────
    with tab_raw:
        with st.expander("Info (yfinance)", expanded=False):
            st.json(info)
        if financials is not None and not financials.empty:
            with st.expander("Financials"):
                safe_dataframe(financials, use_container_width=True)
        if balance_sheet is not None and not balance_sheet.empty:
            with st.expander("Balance Sheet"):
                safe_dataframe(balance_sheet, use_container_width=True)
        if cashflow is not None and not cashflow.empty:
            with st.expander("Cash Flow"):
                safe_dataframe(cashflow, use_container_width=True)

    # ── TAB: Deep Dive (only if results exist) ────────────────────
    if tab_dd is not None and dd_result and "error" not in dd_result:
        with tab_dd:
            thesis = dd_result.get("investment_thesis", {})
            biz = dd_result.get("business_assessment", {})
            recon = dd_result.get("metric_reconciliation", {})
            fwd = dd_result.get("forward_analysis", {})
            adj_scores = dd_result.get("adjusted_scores", {})
            adj_val = dd_result.get("adjusted_valuation", {})

            # ── Section 1: Investment Thesis ───────────────────────
            st.markdown("---")
            rec = thesis.get("recommendation", "N/A")
            conviction = thesis.get("conviction_level", "N/A")
            one_liner = thesis.get("one_line_thesis", "")

            _REC_COLORS = {
                "Strong Buy": "#1a7a3a", "Buy": "#27ae60",
                "Hold": "#f39c12", "Avoid": "#e67e22", "Strong Avoid": "#c0392b",
            }
            rec_color = _REC_COLORS.get(rec, "gray")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
                f'<span style="background:{rec_color};color:white;padding:8px 18px;'
                f'border-radius:8px;font-size:1.3em;font-weight:bold">{rec}</span>'
                f'<span style="background:#eee;padding:6px 14px;border-radius:6px;'
                f'font-size:0.95em">Conviction: {conviction.upper()}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p style="font-size:1.15em;font-style:italic;color:#555">{one_liner}</p>',
                        unsafe_allow_html=True)

            memo = thesis.get("memo_summary", "")
            if memo:
                st.container(border=True).markdown(memo)

            # ── Section 2: Bull | Base | Bear ─────────────────────
            st.markdown("---")
            st.subheader("Scenario Analysis")
            bull_col, base_col, bear_col = st.columns(3)

            bull = thesis.get("bull_case", {})
            base_case = thesis.get("base_case", {})
            bear = thesis.get("bear_case", {})

            with bull_col:
                st.markdown("#### 🟢 Bull Case")
                upside = bull.get("upside_to_intrinsic_value_pct")
                if upside is not None:
                    color = "#27ae60" if upside > 0 else "#e74c3c"
                    st.markdown(f'<p style="font-size:1.8em;font-weight:bold;color:{color}">+{upside:.0f}%</p>',
                                unsafe_allow_html=True)
                st.markdown(bull.get("narrative", ""))
                assumptions = bull.get("key_assumptions", [])
                if assumptions:
                    st.markdown("**Key Assumptions:**")
                    for a in assumptions:
                        st.markdown(f"- {a}")

            with base_col:
                st.markdown("#### 🟡 Base Case")
                exp_ret = base_case.get("expected_return_12_month_pct")
                if exp_ret is not None:
                    color = "#27ae60" if exp_ret > 0 else "#e74c3c"
                    st.markdown(f'<p style="font-size:1.8em;font-weight:bold;color:{color}">{exp_ret:+.0f}%</p>',
                                unsafe_allow_html=True)
                st.markdown(base_case.get("narrative", ""))
                assumptions = base_case.get("key_assumptions", [])
                if assumptions:
                    st.markdown("**Key Assumptions:**")
                    for a in assumptions:
                        st.markdown(f"- {a}")

            with bear_col:
                st.markdown("#### 🔴 Bear Case")
                downside = bear.get("downside_pct")
                if downside is not None:
                    st.markdown(f'<p style="font-size:1.8em;font-weight:bold;color:#e74c3c">-{abs(downside):.0f}%</p>',
                                unsafe_allow_html=True)
                st.markdown(bear.get("narrative", ""))
                risks = bear.get("key_risks", [])
                if risks:
                    st.markdown("**Key Risks:**")
                    for r in risks:
                        st.markdown(f"- {r}")

            # ── Section 3: What Changed ──────────────────────────
            st.markdown("---")
            st.subheader("What Changed (Score Adjustments)")

            # Full comparison table
            orig_scores = dd_result.get("original_scores", {})
            orig_rating = dd_result.get("original_quality_rating", "N/A")
            adj_rating = dd_result.get("adjusted_quality_rating", orig_rating)
            adj_detail = recon.get("score_adjustments", {})
            iv_detail = recon.get("intrinsic_value_adjustments", {})

            def _delta_cell(delta_val, fmt=".1f"):
                """Return colored delta string."""
                if isinstance(delta_val, str):
                    return delta_val
                if delta_val > 0:
                    return f"+{delta_val:{fmt}}"
                if delta_val < 0:
                    return f"{delta_val:{fmt}}"
                return "0"

            change_rows = []
            for pillar_key, pillar_name in [
                ("quality_score", "Quality Score"),
                ("growth_score", "Growth Score"),
                ("value_score", "Value Score"),
                ("sentiment_score", "Sentiment Score"),
                ("composite_score", "Composite Score"),
            ]:
                orig = orig_scores.get(pillar_key, "—")
                adj = adj_scores.get(pillar_key, orig)
                delta = adj_scores.get(f"{pillar_key}_delta", 0)
                reason = adj_detail.get(pillar_key, {}).get("reasoning", "—")
                change_rows.append({
                    "Metric": pillar_name,
                    "Original": f"{orig}" if isinstance(orig, (int, float)) else orig,
                    "Adjusted": f"{adj}" if isinstance(adj, (int, float)) else adj,
                    "Delta": _delta_cell(delta),
                    "Reason": reason,
                })

            # Intrinsic Value
            orig_iv = val.get("intrinsic_value")
            adj_iv_val = adj_val.get("intrinsic_value", 0)
            iv_delta = adj_val.get("intrinsic_value_delta", 0)
            change_rows.append({
                "Metric": "Intrinsic Value",
                "Original": f"${orig_iv:.2f}" if orig_iv else "N/A",
                "Adjusted": f"${adj_iv_val:.2f}",
                "Delta": f"${iv_delta:+.2f}" if iv_delta else "0",
                "Reason": iv_detail.get("reasoning", "—"),
            })

            # Margin of Safety
            orig_mos = val.get("margin_of_safety")
            adj_mos_val = adj_val.get("margin_of_safety", 0)
            mos_delta = adj_val.get("margin_of_safety_delta", 0)
            change_rows.append({
                "Metric": "Margin of Safety",
                "Original": f"{orig_mos:.1f}%" if orig_mos is not None else "N/A",
                "Adjusted": f"{adj_mos_val:.1f}%",
                "Delta": f"{mos_delta:+.1f}%" if mos_delta else "0",
                "Reason": "",
            })

            # Quality Rating
            rating_changed = adj_rating != orig_rating
            change_rows.append({
                "Metric": "Quality Rating",
                "Original": orig_rating,
                "Adjusted": adj_rating,
                "Delta": "Changed" if rating_changed else "—",
                "Reason": "",
            })

            change_df = pd.DataFrame(change_rows)

            def _color_delta(v):
                if not isinstance(v, str):
                    return ""
                v_stripped = v.replace("$", "").replace("%", "").strip()
                if v_stripped.startswith("+"):
                    return "color: #27ae60; font-weight: bold"
                if v_stripped.startswith("-"):
                    return "color: #e74c3c; font-weight: bold"
                return "color: gray"

            change_df = _dedup_df(change_df)
            try:
                safe_dataframe(
                    change_df.style.map(_color_delta, subset=["Delta"]),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception:
                safe_dataframe(change_df, hide_index=True, use_container_width=True)

            # ── Section 4: Business Quality ───────────────────────
            st.markdown("---")
            st.subheader("Business Quality Deep Dive")

            bq1, bq2, bq3 = st.columns(3)
            with bq1:
                moat = biz.get("competitive_moat_assessment", {})
                st.container(border=True).markdown(
                    f"**Competitive Moat**\n\n"
                    f"Exists: {'Yes' if moat.get('moat_exists') else 'No'}\n\n"
                    f"Type: {moat.get('moat_type', 'N/A')}\n\n"
                    f"Durability: {moat.get('moat_durability', 'N/A')}\n\n"
                    f"{moat.get('evidence', '')}"
                )
            with bq2:
                rev_q = biz.get("revenue_quality", {})
                rec_pct = rev_q.get("recurring_revenue_pct_estimate")
                st.container(border=True).markdown(
                    f"**Revenue Quality: {rev_q.get('score', 'N/A')}/100**\n\n"
                    f"Recurring Revenue: {f'{rec_pct:.0%}' if rec_pct else 'N/A'}\n\n"
                    f"Concentration Risk: {rev_q.get('revenue_concentration_risk', 'N/A')}\n\n"
                    f"{rev_q.get('notes', '')}"
                )
            with bq3:
                earn_q = biz.get("earnings_quality", {})
                st.container(border=True).markdown(
                    f"**Earnings Quality: {earn_q.get('score', 'N/A')}/100**\n\n"
                    f"GAAP vs Reality: {earn_q.get('gaap_vs_reality_gap', 'N/A')}\n\n"
                    f"One-time Items: {'Yes' if earn_q.get('one_time_items_detected') else 'No'}\n\n"
                    f"{earn_q.get('one_time_item_details', '')}"
                )

            bq4, bq5 = st.columns(2)
            with bq4:
                cap_alloc = biz.get("capital_allocation_quality", {})
                st.container(border=True).markdown(
                    f"**Capital Allocation: {cap_alloc.get('score', 'N/A')}/100**\n\n"
                    f"Primary Use: {cap_alloc.get('primary_use_of_capital', 'N/A')}\n\n"
                    f"{cap_alloc.get('assessment', '')}"
                )
            with bq5:
                ai_pos = biz.get("ai_and_technology_position", {})
                opp = "Yes" if ai_pos.get("ai_as_opportunity") else "No"
                threat = "Yes" if ai_pos.get("ai_as_threat") else "No"
                initiatives = ai_pos.get("company_ai_initiatives", [])
                st.container(border=True).markdown(
                    f"**AI & Technology Position**\n\n"
                    f"AI Opportunity: {opp} | AI Threat: {threat}\n\n"
                    f"Disruption Risk: {ai_pos.get('disruption_risk_level', 'N/A')} "
                    f"({ai_pos.get('disruption_timeline', 'N/A')})\n\n"
                    + ("\n".join(f"- {i}" for i in initiatives) if initiatives else "No specific initiatives noted")
                )

            # Hidden strengths/weaknesses
            strengths = biz.get("hidden_strengths", [])
            weaknesses = biz.get("hidden_weaknesses", [])
            if strengths or weaknesses:
                sw1, sw2 = st.columns(2)
                with sw1:
                    if strengths:
                        st.markdown("**Hidden Strengths:**")
                        for s in strengths:
                            st.markdown(f"- 🟢 {s}")
                with sw2:
                    if weaknesses:
                        st.markdown("**Hidden Weaknesses:**")
                        for w in weaknesses:
                            st.markdown(f"- 🔴 {w}")

            # ── Section 5: Forward Analysis ───────────────────────
            st.markdown("---")
            st.subheader("Forward Analysis")

            fa1, fa2, fa3 = st.columns(3)
            with fa1:
                margin_t = fwd.get("margin_trajectory", {})
                st.metric("Margin Trajectory", margin_t.get("direction", "N/A"))
                st.caption(margin_t.get("reasoning", ""))
            with fa2:
                rev_t = fwd.get("revenue_trajectory", {})
                st.metric("Revenue Trajectory", rev_t.get("direction", "N/A"))
                st.caption(rev_t.get("reasoning", ""))
            with fa3:
                st.metric("Competitive Position", fwd.get("competitive_position_trend", "N/A"))
                st.caption(fwd.get("competitive_position_evidence", ""))

            # Three-year earnings power
            ep = fwd.get("three_year_earnings_power_estimate", {})
            if ep:
                st.markdown("**Three-Year Earnings Power:**")
                ep1, ep2, ep3 = st.columns(3)
                with ep1:
                    base_gr = ep.get("base_case_growth_rate")
                    st.metric("Base Case Growth", f"{base_gr:.1%}" if base_gr is not None else "N/A")
                with ep2:
                    bull_gr = ep.get("bull_case_growth_rate")
                    st.metric("Bull Case Growth", f"{bull_gr:.1%}" if bull_gr is not None else "N/A")
                with ep3:
                    bear_gr = ep.get("bear_case_growth_rate")
                    st.metric("Bear Case Growth", f"{bear_gr:.1%}" if bear_gr is not None else "N/A")

            # Catalysts and headwinds
            catalysts = fwd.get("growth_catalysts", [])
            headwinds = fwd.get("growth_headwinds", [])

            cat_hw = st.columns(2)
            with cat_hw[0]:
                if catalysts:
                    st.markdown("**Growth Catalysts:**")
                    cat_rows = []
                    for c in catalysts:
                        cat_rows.append({
                            "Catalyst": c.get("catalyst", ""),
                            "Timeline": c.get("timeline", ""),
                            "Magnitude": c.get("magnitude", ""),
                            "Confidence": c.get("management_confidence_level", ""),
                        })
                    safe_dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

            with cat_hw[1]:
                if headwinds:
                    st.markdown("**Growth Headwinds:**")
                    hw_rows = []
                    for h in headwinds:
                        hw_rows.append({
                            "Headwind": h.get("headwind", ""),
                            "Timeline": h.get("timeline", ""),
                            "Magnitude": h.get("magnitude", ""),
                        })
                    safe_dataframe(pd.DataFrame(hw_rows), hide_index=True, use_container_width=True)

            # Industry tailwinds/headwinds
            tail = fwd.get("industry_tailwinds", [])
            head = fwd.get("industry_headwinds", [])
            if tail or head:
                iw1, iw2 = st.columns(2)
                with iw1:
                    if tail:
                        st.markdown("**Industry Tailwinds:**")
                        for t in tail:
                            st.markdown(f"- 🌊 {t}")
                with iw2:
                    if head:
                        st.markdown("**Industry Headwinds:**")
                        for h in head:
                            st.markdown(f"- 🧱 {h}")

            # ── Section 6: Red Flags & Positive Revisions ─────────
            st.markdown("---")
            st.subheader("Red Flags & Positive Revisions")

            flags_col, pos_col = st.columns(2)
            red_flags = recon.get("red_flags", [])
            pos_revisions = recon.get("positive_revisions", [])

            with flags_col:
                if red_flags:
                    for rf in red_flags:
                        sev = rf.get("severity", "minor")
                        if sev == "critical":
                            st.error(f"**CRITICAL:** {rf.get('flag', '')}")
                        elif sev == "significant":
                            st.warning(f"**{rf.get('flag', '')}**")
                        else:
                            st.info(rf.get("flag", ""))
                else:
                    st.success("No red flags identified.")

            with pos_col:
                if pos_revisions:
                    for pr in pos_revisions:
                        st.success(f"**{pr.get('item', '')}** — {pr.get('impact', '')}")
                else:
                    st.info("No positive revisions identified.")

            # What would change thesis
            change_items = thesis.get("what_would_change_thesis", [])
            if change_items:
                st.markdown("**What Would Change This Thesis:**")
                for item in change_items:
                    st.markdown(f"- ☐ {item}")

            # ── Section 7: Ideal Entry ────────────────────────────
            st.markdown("---")
            st.subheader("Entry & Position Sizing")

            entry_price = thesis.get("ideal_entry_price")
            sizing = thesis.get("position_sizing_suggestion", "N/A")
            timeline = thesis.get("time_to_thesis_realization", "N/A")

            en1, en2, en3 = st.columns(3)
            with en1:
                if entry_price and price:
                    pct_diff = ((price - entry_price) / entry_price) * 100
                    st.metric("Ideal Entry Price", f"${entry_price:.2f}",
                              delta=f"{pct_diff:+.1f}% vs current",
                              delta_color="inverse")
                elif entry_price:
                    st.metric("Ideal Entry Price", f"${entry_price:.2f}")
                else:
                    st.metric("Ideal Entry Price", "N/A")
            with en2:
                st.metric("Position Sizing", sizing.replace("_", " ").title())
            with en3:
                st.metric("Time to Realization", timeline)

            entry_reason = thesis.get("ideal_entry_reasoning", "")
            if entry_reason:
                st.caption(entry_reason)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: SECTOR OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "Sector Overview":

    st.header("Sector Overview")

    # Average composite score by sector
    avg_scores = master.groupby("sector")["composite_score"].mean().sort_values(ascending=False)
    fig1 = px.bar(
        x=avg_scores.index,
        y=avg_scores.values,
        title="Average Composite Score by Sector",
        labels={"x": "Sector", "y": "Avg Score"},
        color=avg_scores.values,
        color_continuous_scale="Blues",
    )
    fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

    # Buy candidates count by sector
    buy_df = master[
        (master["quality_rating"] == "Above Bar")
        & (master["margin_of_safety"].fillna(-999) > 20)
    ]
    buy_counts = buy_df.groupby("sector").size().sort_values(ascending=False)
    if not buy_counts.empty:
        fig2 = px.bar(
            x=buy_counts.index,
            y=buy_counts.values,
            title="Buy Candidates by Sector (Above Bar + MoS > 20%)",
            labels={"x": "Sector", "y": "Count"},
            color=buy_counts.values,
            color_continuous_scale="Greens",
        )
        fig2.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No buy candidates found with current filters.")

    # Average margin of safety by sector
    avg_mos = master.groupby("sector")["margin_of_safety"].mean().dropna().sort_values(ascending=False)
    fig3 = px.bar(
        x=avg_mos.index,
        y=avg_mos.values,
        title="Average Margin of Safety by Sector",
        labels={"x": "Sector", "y": "Avg MoS %"},
        color=avg_mos.values,
        color_continuous_scale="RdYlGn",
    )
    fig3.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    # Top 5 per sector
    st.subheader("Top 5 Stocks per Sector")
    for sector in sorted(master["sector"].dropna().unique()):
        sector_df = master[master["sector"] == sector].nlargest(5, "composite_score")
        if sector_df.empty:
            continue
        st.markdown(f"**{sector}**")
        # Render with clickable tickers
        for _, srow in sector_df.iterrows():
            _sr1, _sr2, _sr3, _sr4, _sr5 = st.columns([1.2, 2.5, 1, 1, 1.5])
            with _sr1:
                if st.button(
                    srow["ticker"],
                    key=f"sec_top5_{sector}_{srow['ticker']}",
                    help=f"Analyze {srow['ticker']}",
                ):
                    _open_popup(srow["ticker"])
                    st.rerun()
            _sr2.write(srow.get("company", ""))
            _sr3.write(f"{srow.get('composite_score', 0):.1f}")
            _mos_v = srow.get("margin_of_safety")
            _sr4.write(f"{_mos_v:.1f}%" if pd.notna(_mos_v) else "N/A")
            _sr5.markdown(_rating_badge(srow.get("quality_rating", "")), unsafe_allow_html=True)
        st.divider()

    # ── Methodology Validation (requires XBRL data) ──────────────
    if st.session_state.xbrl_loaded and st.session_state.xbrl_data:
        with st.expander("Methodology Validation (XBRL Backtest)", expanded=False):
            st.markdown(
                "Tests whether high-scoring stocks historically delivered better "
                "3-year forward earnings growth. Uses XBRL data from 2015-2022."
            )

            if st.button("Run Backtest", key="run_backtest"):
                with st.spinner("Running historical backtest..."):
                    tickers_list = list(st.session_state.xbrl_data.keys())
                    bt_df = backtest_scoring_methodology(
                        tickers_list,
                        st.session_state.all_data,
                        st.session_state.xbrl_data,
                        start_year=2015,
                        end_year=2022,
                    )
                    st.session_state["backtest_result"] = bt_df

            bt_df = st.session_state.get("backtest_result")
            if bt_df is not None and not bt_df.empty:
                # Average forward growth by score quartile
                valid = bt_df.dropna(subset=["forward_eps_growth_3yr", "score_quartile"])
                if not valid.empty:
                    avg_by_q = (
                        valid.groupby("score_quartile")["forward_eps_growth_3yr"]
                        .mean()
                        .sort_index()
                    )
                    q_labels = {1: "Q1 (Top)", 2: "Q2", 3: "Q3", 4: "Q4 (Bottom)"}
                    fig = px.bar(
                        x=[q_labels.get(int(q), f"Q{int(q)}") for q in avg_by_q.index],
                        y=(avg_by_q.values * 100),
                        title="Avg 3-Year Forward EPS Growth by Score Quartile",
                        labels={"x": "Score Quartile", "y": "Avg Forward Growth (%)"},
                        color_discrete_sequence=["#27ae60"],
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Based on {len(valid)} ticker-year observations. "
                        f"Higher-scoring quartiles should show higher forward returns."
                    )
                else:
                    st.info("Insufficient data for backtest visualization.")
            elif bt_df is not None:
                st.info("Backtest returned no results — XBRL data may be too sparse.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: SCENARIO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "Scenario Analysis":

    st.header("Scenario Analysis")
    st.markdown(
        "Describe a geopolitical or macroeconomic scenario and Claude will "
        "analyze sector and stock-level impacts."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not found.** Add it to your `.env` file:\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```"
        )

    # Preset scenarios dropdown
    preset_choice = st.selectbox(
        "Quick scenario presets",
        ["Custom"] + list(PRESET_SCENARIOS.keys()),
        key="preset_scenario",
    )

    if preset_choice != "Custom":
        default_text = PRESET_SCENARIOS[preset_choice]
    else:
        default_text = ""

    scenario_text = st.text_area(
        "Scenario description",
        value=default_text,
        placeholder="e.g., The US launches airstrikes on Iranian nuclear facilities",
        height=100,
        key="scenario_input",
    )

    if st.button("Analyze Scenario", type="primary",
                 disabled=not scenario_text.strip() or not api_key):
        with st.spinner("Claude is analyzing the scenario..."):
            result = analyze_scenario(scenario_text, master, api_key)

        if result and "error" not in result:
            _track_cost("scenario")
            st.session_state.scenario_result = result
            # Save to history
            st.session_state.scenario_history.append({
                "text": scenario_text,
                "result": result,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            })
        elif result:
            st.error(result["error"])

    # Past scenarios
    if st.session_state.scenario_history:
        with st.expander(f"Past Scenarios ({len(st.session_state.scenario_history)})"):
            for i, hist in enumerate(reversed(st.session_state.scenario_history)):
                if st.button(
                    f"{hist['timestamp']}: {hist['text'][:60]}...",
                    key=f"hist_{i}",
                ):
                    st.session_state.scenario_result = hist["result"]
                    st.rerun()

    # Display results
    result = st.session_state.scenario_result
    if result and "error" not in result:
        st.success(f"**Summary:** {result.get('scenario_summary', '')}")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Confidence", result.get("confidence", "N/A"))
        with mc2:
            st.metric("Time Horizon", result.get("time_horizon", "N/A"))

        # Winners and losers
        wl_cols = st.columns(2)
        with wl_cols[0]:
            st.markdown("**Winners**")
            for _wi, w in enumerate(result.get("winners", [])):
                _wc1, _wc2 = st.columns([1, 4])
                # Try to extract ticker from string like "AAPL - Apple"
                _w_parts = str(w).split(" ")
                _w_ticker = _w_parts[0].strip("- ").upper() if _w_parts else ""
                if _w_ticker.isalpha() and len(_w_ticker) <= 5:
                    with _wc1:
                        if st.button(_w_ticker, key=f"win_{_wi}"):
                            _open_popup(_w_ticker)
                            st.rerun()
                    _wc2.markdown(f"🟢 {w}")
                else:
                    st.markdown(f"- 🟢 {w}")
        with wl_cols[1]:
            st.markdown("**Losers**")
            for _li, l_ticker in enumerate(result.get("losers", [])):
                _lc1, _lc2 = st.columns([1, 4])
                _l_parts = str(l_ticker).split(" ")
                _l_tk = _l_parts[0].strip("- ").upper() if _l_parts else ""
                if _l_tk.isalpha() and len(_l_tk) <= 5:
                    with _lc1:
                        if st.button(_l_tk, key=f"lose_{_li}"):
                            _open_popup(_l_tk)
                            st.rerun()
                    _lc2.markdown(f"🔴 {l_ticker}")
                else:
                    st.markdown(f"- 🔴 {l_ticker}")

        # Sector impact table
        st.subheader("Sector Impacts")
        sector_impacts = result.get("sector_impacts", {})
        if sector_impacts:
            si_rows = []
            for s_name, info_dict in sector_impacts.items():
                si_rows.append({
                    "Sector": s_name,
                    "Direction": info_dict.get("direction", "neutral"),
                    "Multiplier": f"{info_dict.get('multiplier', 1.0):.2f}x",
                    "Reasoning": info_dict.get("reasoning", ""),
                })
            safe_dataframe(pd.DataFrame(si_rows), hide_index=True,
                         use_container_width=True)

        # Specific ticker impacts (enriched with stress data when available)
        ticker_impacts = result.get("specific_tickers", {})
        if ticker_impacts:
            st.subheader("Specific Stock Impacts")
            stress_vals = st.session_state.stress_valuations or {}
            for _ti_idx, (tk, info_dict) in enumerate(ticker_impacts.items()):
                _ti_c1, _ti_c2, _ti_c3, _ti_c4 = st.columns([1, 1, 1, 4])
                with _ti_c1:
                    if st.button(tk, key=f"ti_{_ti_idx}"):
                        _open_popup(tk)
                        st.rerun()
                with _ti_c2:
                    st.markdown(f"**{info_dict.get('multiplier', 1.0):.2f}x**")
                sv = stress_vals.get(tk, {})
                sh_sc = sv.get("scenarios", {}).get("sustained_high", {})
                with _ti_c3:
                    if sh_sc.get("stress_intrinsic_value"):
                        s_mos = sh_sc.get("stress_margin_of_safety")
                        mult = info_dict.get("multiplier", 1.0)
                        s_status = sh_sc.get("stress_status")
                        if mult > 1.0 and s_status in ("Severely_Impaired", "Earnings_Eliminated"):
                            st.markdown("⚠️ Conflicted")
                        elif mult < 1.0 and s_status == "Rate_Beneficiary":
                            st.markdown("⚠️ Conflicted")
                        elif mult > 1.0 and s_mos is not None and s_mos > 20:
                            st.markdown("✅ Confirmed")
                with _ti_c4:
                    st.caption(info_dict.get("reasoning", "")[:100])

        # Second-order effects
        second_order = result.get("second_order_effects", [])
        if second_order:
            st.subheader("Second-Order Effects")
            for effect in second_order:
                sectors_affected = ", ".join(effect.get("affected_sectors", []))
                lag = effect.get("lag", "unknown")
                mag = effect.get("magnitude", "unknown")
                st.markdown(
                    f"- **{effect.get('effect', '')}**\n"
                    f"  Sectors: {sectors_affected} | Lag: {lag} | Magnitude: {mag}"
                )

        # AI exposure
        ai_exp = result.get("ai_exposure", {})
        if ai_exp and (ai_exp.get("high_opportunity") or ai_exp.get("high_risk")):
            st.subheader("AI Exposure")
            ae_cols = st.columns(2)
            with ae_cols[0]:
                st.markdown("**High Opportunity**")
                for t in ai_exp.get("high_opportunity", []):
                    st.markdown(f"- 🟢 {t}")
            with ae_cols[1]:
                st.markdown("**High Risk**")
                for t in ai_exp.get("high_risk", []):
                    st.markdown(f"- 🔴 {t}")
            if ai_exp.get("reasoning"):
                st.caption(ai_exp["reasoning"])

        # Macro impacts
        macro = result.get("macro_impacts", {})
        if macro:
            dr_adj = macro.get("discount_rate_adjustment", 0)
            st.markdown(f"**Discount Rate Adjustment:** {dr_adj:+.1%}")
            st.caption(macro.get("reasoning", ""))

        # Apply to screener button
        st.divider()
        if st.button("Apply to Screener", type="primary"):
            val_rows = []
            for _, m_row in master.iterrows():
                val_rows.append({
                    "ticker": m_row["ticker"],
                    "sector": m_row["sector"],
                    "intrinsic_value": m_row["intrinsic_value"],
                    "margin_of_safety": m_row["margin_of_safety"],
                    "current_price": m_row["current_price"],
                })
            val_df = pd.DataFrame(val_rows)
            scenario_df = apply_scenario_to_valuations(result, val_df)
            st.session_state.scenario_df = scenario_df
            st.session_state.scenario_active = True
            _navigate_to("Screener")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: SEC INTELLIGENCE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "SEC Intelligence":

    st.header("SEC Intelligence Dashboard")
    st.markdown(
        "Analyze SEC filings (10-K, 10-Q, 8-K, DEF 14A) for any S&P 500 stock. "
        "Claude reads the filings and identifies red flags, management credibility, "
        "and governance quality."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not found.** Add it to your `.env` file:\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```"
        )
        st.stop()

    # Ticker selection
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        sec_ticker = st.selectbox(
            "Select ticker for SEC analysis",
            sorted(st.session_state.all_data.keys()),
            index=None,
            key="sec_dashboard_ticker",
            placeholder="Type or select a ticker...",
        )
    with col_btn:
        run_full = st.button("Run Full Analysis", type="primary",
                             disabled=not sec_ticker, key="sec_dash_run")

    if run_full and sec_ticker:
        with st.spinner(f"Running full SEC analysis for {sec_ticker}... (this may take 1-2 minutes)"):
            sec_result = get_full_sec_analysis(sec_ticker, api_key)
            st.session_state.sec_cache[sec_ticker] = sec_result
            _track_cost("sec_analysis")

    # Show cached results
    sec_data = st.session_state.sec_cache.get(sec_ticker) if sec_ticker else None

    if sec_data:
        # Summary header
        flag = sec_data.get("overall_sec_flag", "green")
        st.markdown(f"### {sec_ticker} — Overall SEC Flag: {_sec_flag_badge(flag)} {flag.upper()}")

        if sec_data.get("errors"):
            for err in sec_data["errors"]:
                st.warning(err)

        # Summary metrics
        sm1, sm2, sm3, sm4 = st.columns(4)

        analysis_10k = sec_data.get("analysis_10k", {})
        analysis_proxy = sec_data.get("analysis_proxy", {})
        credibility = sec_data.get("management_credibility", {})
        analysis_8k = sec_data.get("analysis_8k", {})

        with sm1:
            sentiment = analysis_10k.get("overall_sentiment", "N/A") if analysis_10k else "N/A"
            st.metric("10-K Sentiment", sentiment)
        with sm2:
            gov_grade = (analysis_proxy.get("governance_quality", {}).get("overall_grade", "N/A")
                         if analysis_proxy else "N/A")
            st.metric("Governance Grade", gov_grade)
        with sm3:
            cred_score = credibility.get("credibility_score", "N/A") if credibility else "N/A"
            st.metric("Mgmt Credibility", f"{cred_score}/100" if isinstance(cred_score, (int, float)) else cred_score)
        with sm4:
            event_signal = analysis_8k.get("overall_signal", "N/A") if analysis_8k else "N/A"
            st.metric("8-K Signal", event_signal)

        # Detailed sections
        tab_10k, tab_proxy, tab_8k, tab_cred, tab_filings = st.tabs([
            "10-K Analysis", "Governance", "8-K Events", "Management Credibility", "Filing History"
        ])

        with tab_10k:
            if analysis_10k and "error" not in analysis_10k:
                st.markdown(f"**SEC Flag:** {_sec_flag_badge(analysis_10k.get('sec_flag', 'green'))}")

                red_flags = analysis_10k.get("red_flags", [])
                if red_flags:
                    st.subheader("Red Flags")
                    for rf in red_flags:
                        sev = rf.get("severity", "low")
                        icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "⚪"
                        st.markdown(f"{icon} **{rf.get('flag', '')}** *(Section: {rf.get('section', 'N/A')})*")

                key_risks = analysis_10k.get("key_risks", [])
                if key_risks:
                    st.subheader("Key Risks")
                    risk_df = pd.DataFrame(key_risks)
                    safe_dataframe(risk_df, hide_index=True, use_container_width=True)

                tone = analysis_10k.get("management_tone", {})
                if tone:
                    st.subheader("Management Tone")
                    st.markdown(f"**Confidence:** {tone.get('confidence_level', 'N/A')}")
                    hedging = tone.get("hedging_language_pct")
                    if hedging is not None:
                        st.progress(min(hedging, 1.0), text=f"Hedging Language: {hedging:.0%}")
                    phrases = tone.get("notable_phrases", [])
                    if phrases:
                        st.markdown("Notable phrases: " + ", ".join(f'"{p}"' for p in phrases))

                guidance = analysis_10k.get("forward_guidance", {})
                if guidance:
                    st.subheader("Forward Guidance")
                    st.markdown(f"**Outlook:** {guidance.get('outlook', 'N/A')}")
                    st.markdown(f"**CapEx Signals:** {guidance.get('capex_signals', 'N/A')}")
                    inits = guidance.get("key_initiatives", [])
                    if inits:
                        for init in inits:
                            st.markdown(f"- {init}")

                changes = analysis_10k.get("notable_changes", [])
                if changes:
                    st.subheader("Notable Changes")
                    for ch in changes:
                        sig = ch.get("significance", "low")
                        icon = "🔴" if sig == "high" else "🟡" if sig == "medium" else "⚪"
                        st.markdown(f"{icon} {ch.get('change', '')}")
            else:
                st.info("No 10-K analysis available. Run analysis to populate.")

        with tab_proxy:
            if analysis_proxy and "error" not in analysis_proxy:
                gov = analysis_proxy.get("governance_quality", {})
                st.subheader("Governance Quality")
                gov_cols = st.columns(4)
                with gov_cols[0]:
                    st.metric("Overall Grade", gov.get("overall_grade", "N/A"))
                with gov_cols[1]:
                    indep = gov.get("board_independence_pct")
                    st.metric("Board Independence",
                              f"{indep:.0%}" if isinstance(indep, (int, float)) else "N/A")
                with gov_cols[2]:
                    st.metric("Dual-Class", "Yes" if gov.get("dual_class_shares") else "No")
                with gov_cols[3]:
                    st.metric("Poison Pill", "Yes" if gov.get("poison_pill") else "No")

                comp = analysis_proxy.get("executive_compensation", {})
                if comp:
                    st.subheader("Executive Compensation")
                    st.markdown(f"**CEO Total Comp:** {comp.get('ceo_total_comp', 'N/A')}")
                    st.markdown(f"**Pay vs Performance:** {comp.get('pay_vs_performance_alignment', 'N/A')}")
                    stock_pct = comp.get("stock_based_pct")
                    if stock_pct is not None:
                        st.markdown(f"**Stock-Based %:** {stock_pct:.0%}")
                    concerns = comp.get("concerns", [])
                    if concerns:
                        for c in concerns:
                            st.markdown(f"- ⚠️ {c}")

                sh_concerns = analysis_proxy.get("shareholder_concerns", [])
                if sh_concerns:
                    st.subheader("Shareholder Concerns")
                    for sc in sh_concerns:
                        sev = sc.get("severity", "low")
                        icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "⚪"
                        st.markdown(f"{icon} {sc.get('concern', '')}")

                rp_flags = analysis_proxy.get("related_party_flags", [])
                if rp_flags:
                    st.subheader("Related Party Flags")
                    for rp in rp_flags:
                        sev = rp.get("severity", "low")
                        icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "⚪"
                        st.markdown(f"{icon} {rp.get('flag', '')}")
            else:
                st.info("No proxy analysis available. Run analysis to populate.")

        with tab_8k:
            if analysis_8k and "error" not in analysis_8k:
                st.markdown(f"**Overall Signal:** {analysis_8k.get('overall_signal', 'N/A')}")
                st.markdown(analysis_8k.get("event_summary", ""))

                events = analysis_8k.get("material_events", [])
                if events:
                    st.subheader("Material Events Timeline")
                    for ev in events:
                        impact = ev.get("impact", "neutral")
                        icon = "🟢" if impact == "positive" else "🔴" if impact == "negative" else "⚪"
                        sig = ev.get("significance", "low")
                        st.markdown(
                            f"{icon} **{ev.get('date', '')}** "
                            f"[{ev.get('event_type', '')}] "
                            f"({sig} significance)\n\n"
                            f"  {ev.get('description', '')}"
                        )
            else:
                st.info("No 8-K analysis available. Run analysis to populate.")

        with tab_cred:
            if credibility and "error" not in credibility:
                cred_score = credibility.get("credibility_score", 0)
                st.markdown(f"**Credibility Score:** {cred_score}/100")
                st.progress(min(cred_score / 100, 1.0))
                st.markdown(f"**Flag:** {_sec_flag_badge(credibility.get('credibility_flag', 'green'))}")
                st.markdown(credibility.get("summary", ""))

                # Goal tracking
                gt = credibility.get("goal_tracking", {})
                if gt:
                    st.subheader("Goal Tracking")
                    gt_cols = st.columns(4)
                    with gt_cols[0]:
                        st.metric("Revenue Targets", gt.get("revenue_targets", "N/A"))
                    with gt_cols[1]:
                        st.metric("Margin Targets", gt.get("margin_targets", "N/A"))
                    with gt_cols[2]:
                        st.metric("CapEx Plans", gt.get("capex_plans", "N/A"))
                    with gt_cols[3]:
                        st.metric("Strategic Initiatives", gt.get("strategic_initiatives", "N/A"))

                kept = credibility.get("promises_kept", [])
                if kept:
                    st.subheader("Promises Kept")
                    for p in kept:
                        st.markdown(
                            f"✅ **{p.get('promise', '')}** → {p.get('outcome', '')} "
                            f"(Grade: {p.get('grade', 'N/A')})"
                        )

                broken = credibility.get("promises_broken", [])
                if broken:
                    st.subheader("Promises Broken")
                    for p in broken:
                        sev = p.get("severity", "low")
                        icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "⚪"
                        st.markdown(f"{icon} **{p.get('promise', '')}** → {p.get('outcome', '')}")
            else:
                st.info("No credibility analysis available. Run analysis to populate.")

        with tab_filings:
            filings_found = sec_data.get("filings_found", [])
            if filings_found:
                safe_dataframe(
                    pd.DataFrame(filings_found),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No filings found.")

    elif not sec_ticker:
        st.info("Select a ticker above to begin SEC analysis.")

    # Batch analysis section
    st.divider()
    st.subheader("Batch SEC Screening")
    st.markdown(
        "Run SEC analysis on multiple stocks at once. "
        "Results are cached for the session."
    )

    batch_options = st.multiselect(
        "Select tickers for batch analysis",
        sorted(st.session_state.all_data.keys()),
        max_selections=10,
        key="sec_batch_tickers",
    )

    if st.button("Run Batch Analysis", disabled=not batch_options or not api_key,
                 key="sec_batch_run"):
        progress = st.progress(0, text="Starting batch SEC analysis...")
        for i, t in enumerate(batch_options):
            if t not in st.session_state.sec_cache:
                progress.progress(
                    (i + 1) / len(batch_options),
                    text=f"Analyzing {t} ({i + 1}/{len(batch_options)})...",
                )
                sec_result = get_full_sec_analysis(t, api_key)
                st.session_state.sec_cache[t] = sec_result
                _track_cost("sec_analysis")
            else:
                progress.progress(
                    (i + 1) / len(batch_options),
                    text=f"Skipping {t} (cached)...",
                )
        progress.empty()
        st.success(f"Completed batch analysis for {len(batch_options)} stocks.")
        st.rerun()

    # ── Deep Dive: Top Buy Candidates ─────────────────────────────────
    st.divider()
    st.subheader("🔬 Deep Dive — Top Buy Candidates")
    st.markdown(
        "Run comprehensive AI deep dive analysis on top-ranked stocks. "
        "Buy candidates: Above Bar quality + positive margin of safety + top quartile score."
    )

    dd_top_n = st.selectbox(
        "Analyze top:",
        ["Top 10", "Top 25", "Top 50"],
        index=1,
        key="dd_top_n_select",
    )
    n_limit = int(dd_top_n.split()[1])

    # Build buy candidates list
    _buy_cands = master[
        (master["quality_rating"] == "Above Bar")
        & (master["margin_of_safety"].fillna(-999) > 0)
        & (master["composite_score"].fillna(0) >= master["composite_score"].quantile(0.75))
    ].nlargest(n_limit, "composite_score")
    _buy_tickers = _buy_cands["ticker"].tolist()

    _dd_cache = st.session_state.deep_dive_cache
    _dd_need = [t for t in _buy_tickers if t not in _dd_cache]
    _dd_have = [t for t in _buy_tickers if t in _dd_cache]

    with st.expander(f"Preview: {len(_buy_tickers)} candidates", expanded=False):
        if not _buy_tickers:
            st.info("No stocks match buy candidate criteria with current settings.")
        else:
            preview_df = _buy_cands[["ticker", "company", "composite_score",
                                     "margin_of_safety", "quality_rating"]].copy()
            preview_df.columns = ["Ticker", "Company", "Score", "MoS %", "Rating"]
            safe_dataframe(preview_df, hide_index=True, use_container_width=True)

    if _buy_tickers:
        est_cost = len(_dd_need) * _API_COSTS["deep_dive"]
        est_time = len(_dd_need) * 75
        info_parts = []
        if _dd_need:
            info_parts.append(f"**{len(_dd_need)}** to analyze (~${est_cost:.2f}, ~{est_time // 60}m {est_time % 60}s)")
        if _dd_have:
            info_parts.append(f"**{len(_dd_have)}** already cached")
        st.info(" | ".join(info_parts))

        dd_confirm = st.button(
            f"Confirm — Analyze {len(_dd_need)} stocks (~${est_cost:.2f})",
            type="primary",
            disabled=not _dd_need or not api_key,
            key="dd_buy_confirm",
        )

        if dd_confirm:
            progress = st.progress(0, text="Starting deep dive batch...")
            succeeded = 0
            failed = 0
            for i, t in enumerate(_dd_need):
                progress.progress(
                    (i + 1) / len(_dd_need),
                    text=f"Analyzing {t} ({i + 1}/{len(_dd_need)})...",
                )
                result = _run_dd_for_ticker(t, api_key)
                if result:
                    succeeded += 1
                else:
                    failed += 1
            progress.empty()
            msg = f"Completed deep dive for {succeeded} stock{'s' if succeeded != 1 else ''}."
            if failed:
                msg += f" {failed} failed."
            st.success(msg)
            st.rerun()

    # Deep dive results summary (on SEC Intelligence page)
    if st.session_state.deep_dive_cache:
        st.divider()
        st.subheader("Deep Dive Results")
        _rec_order = {"Strong Buy": 0, "Buy": 1, "Hold": 2, "Avoid": 3, "Strong Avoid": 4}
        _dd_sorted = sorted(
            st.session_state.deep_dive_cache.items(),
            key=lambda x: _rec_order.get(
                x[1].get("investment_thesis", {}).get("recommendation", "N/A"), 5
            ),
        )
        for _dd_i, (t, dd) in enumerate(_dd_sorted):
            thesis = dd.get("investment_thesis", {})
            rec = thesis.get("recommendation", "N/A")
            _ddc1, _ddc2, _ddc3, _ddc4 = st.columns([1, 2, 1, 4])
            with _ddc1:
                if st.button(t, key=f"sec_dd_{_dd_i}"):
                    _open_popup(t)
                    st.rerun()
            with _ddc2:
                st.markdown(f"**{rec}**")
            with _ddc3:
                st.caption(thesis.get("conviction_level", "N/A"))
            with _ddc4:
                st.caption(thesis.get("one_line_thesis", ""))

    # Show summary of all cached SEC analyses
    if st.session_state.sec_cache:
        st.subheader("Cached SEC Analysis Summary")
        for _sec_i, (t, sd) in enumerate(st.session_state.sec_cache.items()):
            a10k = sd.get("analysis_10k", {}) or {}
            aproxy = sd.get("analysis_proxy", {}) or {}
            acred = sd.get("management_credibility", {}) or {}
            _sc1, _sc2, _sc3, _sc4, _sc5 = st.columns([1, 2, 2, 2, 2])
            with _sc1:
                if st.button(t, key=f"sec_sum_{_sec_i}"):
                    _open_popup(t)
                    st.rerun()
            with _sc2:
                st.markdown(_sec_flag_badge(sd.get("overall_sec_flag", "green")), unsafe_allow_html=True)
            with _sc3:
                st.caption(f"10-K: {a10k.get('overall_sentiment', 'N/A')}")
            with _sc4:
                st.caption(f"Gov: {(aproxy.get('governance_quality', {}) or {}).get('overall_grade', 'N/A')}")
            with _sc5:
                st.caption(f"Cred: {acred.get('credibility_score', 'N/A')}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6: SUPERFORECASTING LAB
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "Superforecasting":

    st.title("Superforecasting Lab")
    st.caption("Scenario Intelligence & Investment Implications")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not found.** Add it to your `.env` file to enable "
            "scenario analysis.\n\n```\nANTHROPIC_API_KEY=sk-ant-...\n```"
        )

    sf_tab1, sf_tab2, sf_tab3, sf_tab4 = st.tabs([
        "New Scenario",
        "Active Scenarios",
        "Calibration Dashboard",
        "Scenario History",
    ])

    # ── TAB 1: New Scenario ──────────────────────────────────────
    with sf_tab1:
        st.subheader("Describe Your Scenario")
        st.caption(
            "Speak naturally — the system will ask follow-up questions "
            "to build a complete analysis brief."
        )

        # Auto-fire intake if seeded from screener bar (1 user msg, no assistant reply)
        _sf_conv = st.session_state.sf_intake_conversation
        _sf_needs_autofire = (
            len(_sf_conv) == 1
            and _sf_conv[0]["role"] == "user"
            and api_key
            and not st.session_state.sf_intake_complete
        )
        if _sf_needs_autofire:
            _sf_seed = _sf_conv[0]["content"]
            with st.chat_message("user"):
                st.write(_sf_seed)
            with st.spinner("Analyzing your scenario..."):
                intake_result = conduct_intake_conversation(
                    initial_input=_sf_seed,
                    conversation_history=_sf_conv,
                    current_brief=st.session_state.sf_current_brief,
                )
            st.session_state.sf_current_brief = intake_result["current_brief"]
            st.session_state.sf_intake_complete = intake_result["intake_complete"]
            if intake_result["intake_complete"]:
                _sf_resp = (
                    f"I have enough information to proceed.\n\n"
                    f"**Scenario Brief:**\n{intake_result['brief_summary']}\n\n"
                    f"Ready to run the full analysis."
                )
            else:
                _sf_resp = intake_result["next_question"] or "Could you elaborate?"
            st.session_state.sf_intake_conversation.append({
                "role": "assistant", "content": _sf_resp,
            })
            st.rerun()

        # Display conversation history as chat bubbles
        for msg in st.session_state.sf_intake_conversation:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # If intake not complete, show input
        if not st.session_state.sf_intake_complete:
            user_input = st.chat_input(
                "Describe a geopolitical or macroeconomic scenario..."
            )

            if user_input and api_key:
                # Add user message to history
                st.session_state.sf_intake_conversation.append({
                    "role": "user",
                    "content": user_input,
                })

                # Call intake engine
                with st.spinner("Analyzing..."):
                    intake_result = conduct_intake_conversation(
                        initial_input=user_input,
                        conversation_history=st.session_state.sf_intake_conversation,
                        current_brief=st.session_state.sf_current_brief,
                    )

                st.session_state.sf_current_brief = intake_result["current_brief"]
                st.session_state.sf_intake_complete = intake_result["intake_complete"]

                # Build assistant response
                if intake_result["intake_complete"]:
                    response = (
                        f"I have enough information to proceed.\n\n"
                        f"**Scenario Brief:**\n{intake_result['brief_summary']}\n\n"
                        f"Ready to run the full analysis."
                    )
                else:
                    response = intake_result["next_question"] or "Could you elaborate?"

                st.session_state.sf_intake_conversation.append({
                    "role": "assistant",
                    "content": response,
                })

                st.rerun()

        # When intake complete, show action buttons
        if st.session_state.sf_intake_complete:
            sf_a1, sf_a2, sf_a3 = st.columns(3)

            with sf_a1:
                if st.button("Run Full Analysis", type="primary", key="sf_run"):
                    # Save scenario to database
                    import json as _json
                    scenario_id = db_save_scenario({
                        "title": st.session_state.sf_current_brief.get(
                            "title", "Untitled Scenario"
                        ),
                        "raw_input": (
                            st.session_state.sf_intake_conversation[0]["content"]
                            if st.session_state.sf_intake_conversation
                            else ""
                        ),
                        "brief_json": _json.dumps(
                            st.session_state.sf_current_brief, default=str
                        ),
                    })

                    # Progress tracking
                    sf_progress = st.progress(0)
                    sf_status = st.empty()

                    def _sf_progress(step, total, message):
                        sf_progress.progress(step / total)
                        sf_status.text(message)

                    # Run analysis
                    screener_input = (
                        st.session_state.scores_df
                        if st.session_state.scores_df is not None
                        else {}
                    )
                    sf_result = run_full_scenario_analysis(
                        brief=st.session_state.sf_current_brief,
                        screener_data=screener_input,
                        progress_callback=_sf_progress,
                    )

                    # Save results
                    db_update_scenario_results(
                        scenario_id,
                        sf_result.get("tree", {}),
                        sf_result.get("opportunities", {}),
                        sf_result.get("report", {}),
                    )
                    db_save_recommendations(
                        scenario_id,
                        sf_result.get("opportunities", {}).get("opportunities", []),
                    )

                    st.session_state.sf_scenario_result = sf_result
                    st.session_state.sf_current_scenario_id = scenario_id
                    sf_progress.progress(1.0)
                    sf_status.text("Analysis complete!")
                    _track_cost("scenario")
                    st.rerun()

            with sf_a2:
                if st.button("Edit Brief", key="sf_edit"):
                    st.session_state.sf_intake_complete = False
                    st.rerun()

            with sf_a3:
                if st.button("Start New Scenario", key="sf_new"):
                    st.session_state.sf_intake_conversation = []
                    st.session_state.sf_current_brief = {}
                    st.session_state.sf_intake_complete = False
                    st.session_state.sf_scenario_result = None
                    st.session_state.sf_current_scenario_id = None
                    st.rerun()

        # ── Display results if analysis has been run ─────────────
        if st.session_state.sf_scenario_result:
            sf_res = st.session_state.sf_scenario_result
            sf_tree = sf_res.get("tree", {})
            sf_report = sf_res.get("report", {})
            sf_opps = sf_res.get("opportunities", {})

            st.divider()

            # Scenario overview
            st.subheader("Scenario Overview")
            sfo1, sfo2, sfo3 = st.columns(3)
            sfo1.metric(
                "Base Probability",
                f"{sf_tree.get('base_event_probability', 0) * 100:.0f}%",
            )
            sfo2.metric(
                "Model Confidence",
                sf_tree.get("model_confidence", "N/A").title(),
            )
            sfo3.metric(
                "Branches Modeled",
                len(sf_tree.get("branches", [])),
            )

            # Executive summary
            exec_summary = sf_report.get("executive_summary", "")
            if exec_summary:
                st.info(exec_summary)

            # Probability tree as table
            st.subheader("Probability Tree")
            sf_branches = sf_tree.get("branches", [])
            if sf_branches:
                branch_df = pd.DataFrame([{
                    "Branch": b["label"],
                    "Probability": f"{b['probability'] * 100:.1f}%",
                    "Timeline": b.get("timeline", "N/A"),
                    "Description": b["description"][:120] + ("..." if len(b["description"]) > 120 else ""),
                } for b in sf_branches])
                safe_dataframe(branch_df, hide_index=True, use_container_width=True)

            # Historical analogues
            sf_analogues = sf_tree.get("historical_analogues", [])
            if sf_analogues:
                with st.expander("Historical Analogues"):
                    for a in sf_analogues:
                        st.markdown(
                            f"**{a.get('event', 'N/A')} ({a.get('year', 'N/A')})** — "
                            f"{a.get('similarity', '')} "
                            f"Outcome: {a.get('outcome', '')}"
                        )

            # Second order effects
            sf_effects = sf_report.get("second_order_effects", [])
            if sf_effects:
                with st.expander("Second Order Effects"):
                    for e in sf_effects:
                        st.markdown(f"- {e}")

            # Investment implications
            st.subheader("Investment Implications")
            opp_tab_a, opp_tab_b, opp_tab_c = st.tabs([
                "Opportunities", "Risks / Avoid", "Hedges",
            ])

            def _render_sf_instruments(instruments, direction_label):
                if not instruments:
                    st.info(f"No {direction_label} identified.")
                    return
                for idx, inst in enumerate(instruments[:20]):
                    ic0, ic1, ic2, ic3, ic4 = st.columns([1, 2, 1, 1, 3])
                    _sf_tk = inst.get("ticker", "N/A")
                    with ic0:
                        if _sf_tk != "N/A" and _sf_tk.isalpha() and len(_sf_tk) <= 5:
                            if st.button(_sf_tk, key=f"sf_{direction_label}_{idx}"):
                                _open_popup(_sf_tk)
                                st.rerun()
                        else:
                            st.markdown(f"**{_sf_tk}**")
                    with ic1:
                        st.caption(inst.get("name", "")[:30])
                    with ic2:
                        impact = inst.get("expected_impact_pct", 0)
                        color = "green" if impact > 0 else "red"
                        st.markdown(f":{color}[{impact:+.1f}%]")
                    with ic3:
                        st.caption(f"Conviction: {inst.get('conviction_score', 0):.0f}/100")
                    with ic4:
                        st.caption(inst.get("causal_mechanism", "")[:80])

            with opp_tab_a:
                _render_sf_instruments(sf_opps.get("opportunities", []), "opportunities")

            with opp_tab_b:
                _render_sf_instruments(sf_opps.get("risks", []), "risks")

            with opp_tab_c:
                _render_sf_instruments(sf_opps.get("hedges", []), "hedges")

            # Monitoring indicators
            sf_indicators = sf_report.get("key_monitoring_indicators", [])
            if sf_indicators:
                st.subheader("Key Monitoring Indicators")
                st.caption("Watch these signals to determine which branch is materializing")
                for ind in sf_indicators:
                    with st.expander(f"{ind.get('indicator', 'N/A')}"):
                        st.write(f"**Signals:** {ind.get('signals_branch', 'N/A')}")
                        st.write(f"**Action:** {ind.get('update_action', 'N/A')}")

            # Portfolio notes
            pf_notes = sf_report.get("portfolio_construction_notes", "")
            if pf_notes:
                with st.expander("Portfolio Construction Notes"):
                    st.markdown(pf_notes)

    # ── TAB 2: Active Scenarios ──────────────────────────────────
    with sf_tab2:
        st.subheader("Active Scenarios")

        active_scenarios = db_get_all_scenarios(status="active")

        if not active_scenarios:
            st.info("No active scenarios. Create one in the New Scenario tab.")
        else:
            for sc in active_scenarios:
                import json as _json
                sc_brief = _json.loads(sc.get("brief_json", "{}"))

                with st.container():
                    sc_c1, sc_c2, sc_c3 = st.columns([3, 1, 2])

                    with sc_c1:
                        st.markdown(f"### {sc['title']}")
                        st.caption(f"Created: {str(sc['created_at'])[:10]}")
                        sc_prob = sc_brief.get("base_probability", 0)
                        if isinstance(sc_prob, (int, float)):
                            st.metric("Base Probability", f"{sc_prob * 100:.0f}%")

                    with sc_c2:
                        try:
                            days_active = (
                                pd.Timestamp.now() - pd.Timestamp(sc["created_at"])
                            ).days
                            st.metric("Days Active", days_active)
                        except Exception:
                            st.metric("Days Active", "N/A")

                    with sc_c3:
                        if st.button(
                            "Resolve", key=f"resolve_{sc['id']}"
                        ):
                            st.session_state.sf_resolving_scenario = sc["id"]

                        if st.button(
                            "View Results", key=f"view_{sc['id']}"
                        ):
                            sc_report = _json.loads(sc.get("report_json", "{}"))
                            if sc_report:
                                st.session_state.sf_scenario_result = {
                                    "tree": _json.loads(sc.get("tree_json", "{}")),
                                    "opportunities": _json.loads(
                                        sc.get("opportunities_json", "{}")
                                    ),
                                    "report": sc_report,
                                }
                                _navigate_to("Superforecasting")
                                st.rerun()

                    # Resolution UI
                    if st.session_state.sf_resolving_scenario == sc["id"]:
                        sc_tree = _json.loads(sc.get("tree_json", "{}"))
                        sc_branches = sc_tree.get("branches", [])
                        sc_branch_labels = [b.get("label", b.get("id", "?")) for b in sc_branches]

                        if sc_branch_labels:
                            actual_branch = st.selectbox(
                                "Which branch materialized?",
                                sc_branch_labels,
                                key=f"branch_sel_{sc['id']}",
                            )
                            res_notes = st.text_area(
                                "Resolution notes",
                                key=f"notes_{sc['id']}",
                            )

                            rc1, rc2 = st.columns(2)
                            with rc1:
                                if st.button(
                                    "Confirm Resolution",
                                    key=f"confirm_{sc['id']}",
                                    type="primary",
                                ):
                                    sel_idx = sc_branch_labels.index(actual_branch)
                                    branch_id = sc_branches[sel_idx].get("id", "")
                                    db_resolve_scenario(sc["id"], branch_id, res_notes)
                                    st.session_state.sf_resolving_scenario = None
                                    st.success("Scenario resolved! Check Calibration tab.")
                                    st.rerun()
                            with rc2:
                                if st.button("Cancel", key=f"cancel_{sc['id']}"):
                                    st.session_state.sf_resolving_scenario = None
                                    st.rerun()
                        else:
                            st.warning("No branches found — run analysis first.")

                    st.divider()

    # ── TAB 3: Calibration Dashboard ─────────────────────────────
    with sf_tab3:
        st.subheader("Forecasting Calibration")

        cal_data = db_get_calibration_data()

        if cal_data.empty:
            st.info(
                "Calibration data appears after you resolve scenarios. "
                "Resolve at least 3 scenarios to see meaningful calibration."
            )
        else:
            cal_analysis = analyze_calibration(cal_data)

            # Summary metrics
            cal_c1, cal_c2, cal_c3, cal_c4 = st.columns(4)
            brier_val = cal_analysis.get("overall_brier_score")
            cal_c1.metric(
                "Brier Score",
                f"{brier_val:.3f}" if brier_val is not None else "N/A",
                help="Lower is better. 0.25 = random. 0.0 = perfect.",
            )
            cal_c2.metric("Grade", cal_analysis.get("calibration_grade", "N/A"))
            cal_c3.metric(
                "Bias",
                cal_analysis.get("bias", "N/A").replace("_", " ").title(),
            )
            cal_c4.metric("Sample Size", cal_analysis.get("sample_size", 0))

            # Calibration curve chart
            cal_buckets = cal_analysis.get("buckets", [])
            if cal_buckets:
                fig_cal = go.Figure()

                # Perfect calibration line
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(dash="dash", color="gray"),
                ))

                # Actual calibration
                fig_cal.add_trace(go.Scatter(
                    x=[b["predicted_probability"] for b in cal_buckets],
                    y=[b["actual_frequency"] for b in cal_buckets],
                    mode="lines+markers",
                    name="Your Calibration",
                    marker=dict(size=10),
                ))

                fig_cal.update_layout(
                    title="Calibration Curve",
                    xaxis_title="Predicted Probability",
                    yaxis_title="Actual Frequency",
                    height=400,
                )
                st.plotly_chart(fig_cal, use_container_width=True)

            # AI insights button
            if api_key and st.button("Generate Calibration Insights", key="cal_insights"):
                with st.spinner("Analyzing your forecasting patterns..."):
                    insights = generate_calibration_insights(cal_analysis)
                st.info(insights)

    # ── TAB 4: Scenario History ──────────────────────────────────
    with sf_tab4:
        st.subheader("Scenario History")

        all_sf_scenarios = db_get_all_scenarios()

        if not all_sf_scenarios:
            st.info("No scenarios yet. Create your first in the New Scenario tab.")
        else:
            import json as _json

            total_sf = len(all_sf_scenarios)
            resolved_sf = len([s for s in all_sf_scenarios if s["status"] == "resolved"])
            active_sf = len([s for s in all_sf_scenarios if s["status"] == "active"])

            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Total Scenarios", total_sf)
            hc2.metric("Active", active_sf)
            hc3.metric("Resolved", resolved_sf)

            # History table
            history_data = []
            for s in all_sf_scenarios:
                s_brief = _json.loads(s.get("brief_json", "{}"))
                history_data.append({
                    "Title": s["title"],
                    "Status": s["status"].title(),
                    "Created": str(s["created_at"])[:10],
                    "Resolved": str(s.get("resolved_at") or "N/A")[:10],
                    "Base Probability": f"{s_brief.get('base_probability', 0) * 100:.0f}%"
                        if isinstance(s_brief.get("base_probability"), (int, float)) else "N/A",
                    "Brier Score": f"{s['brier_score']:.3f}" if s.get("brier_score") else "Pending",
                })

            hist_df = pd.DataFrame(history_data)
            safe_dataframe(hist_df, hide_index=True, use_container_width=True)
