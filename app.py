# 1. pip install -r requirements.txt
# 2. Copy .env.example to .env and add your Anthropic API key
# 3. streamlit run app.py
# First run will take 15-25 minutes to fetch all S&P 500 data.
# Subsequent runs use cached data and load in seconds.

"""
S&P 500 Stock Screener — Main Streamlit Application (v2).

Five pages: Screener | Stock Detail | Sector Overview | Scenario Analysis | SEC Intelligence
"""

import os

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
    load_cached_deep_dive, STEPS as DD_STEPS,
)

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Data loading & computation ──────────────────────────────────────────────

def _build_master_df() -> pd.DataFrame:
    """Merge valuations, scores, and ratings into a single master DataFrame."""
    all_data = st.session_state.all_data
    valuations = st.session_state.valuations
    scores_df = st.session_state.scores_df
    ratings_df = st.session_state.ratings_df

    rows = []
    for ticker, data in all_data.items():
        info = data.get("info", {})
        val = valuations.get(ticker, {})
        rows.append({
            "ticker": ticker,
            "company": data.get("company", info.get("shortName", "")),
            "sector": data.get("sector", info.get("sector", "Unknown")),
            "current_price": val.get("current_price"),
            "intrinsic_value": val.get("intrinsic_value"),
            "margin_of_safety": val.get("margin_of_safety"),
            "base_eps": val.get("base_eps"),
            "growth_rate_used": val.get("growth_rate_used"),
            "val_error": val.get("error"),
        })

    df = pd.DataFrame(rows)
    df = df.merge(scores_df, on="ticker", how="left", suffixes=("", "_score"))
    df = df.merge(
        ratings_df[["ticker", "quality_rating", "rating_total",
                     "earnings_consistency_score", "debt_discipline_score",
                     "dividend_quality_score", "buyback_score"]],
        on="ticker", how="left",
    )

    # Use sector from scores_df if the main one is missing
    if "sector_score" in df.columns:
        df["sector"] = df["sector"].fillna(df["sector_score"])
        df.drop(columns=["sector_score"], inplace=True, errors="ignore")

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

    st.session_state.master_df = _build_master_df()
    st.session_state.data_loaded = True


# ── Sidebar ─────────────────────────────────────────────────────────────────

PAGES = ["Screener", "Stock Detail", "Sector Overview",
         "Scenario Analysis", "SEC Intelligence"]

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
            st.session_state.page = "Stock Detail"
            st.rerun()

    st.divider()

    page = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.page),
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

    weights_changed = (
        w_q != st.session_state.w_quality
        or w_g != st.session_state.w_growth
        or w_v != st.session_state.w_value
        or w_s != st.session_state.w_sentiment
    )
    dr_changed = dr != st.session_state.discount_rate

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
            st.session_state.master_df = _build_master_df()

    st.divider()
    if st.session_state.data_loaded and st.session_state.all_data:
        st.caption(f"{len(st.session_state.all_data)} stocks loaded")

    st.divider()
    cost = st.session_state.session_api_cost
    st.caption(f"Session API cost: ~${cost:.2f}")


# ── Cost tracking helper ────────────────────────────────────────────────────

_API_COSTS = {
    "sec_analysis": 0.02,
    "deep_dive": 0.10,
    "scenario": 0.03,
}

def _track_cost(call_type: str, count: int = 1):
    """Add estimated API cost to the session total."""
    st.session_state.session_api_cost += _API_COSTS[call_type] * count


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

    st.header("S&P 500 Stock Screener")

    # ── Search bar ───────────────────────────────────────────────────
    search_term = st.text_input(
        "Search by ticker or company name",
        value=st.session_state.get("screener_search_term", ""),
        key="screener_search_input",
        placeholder="e.g. AMZN or Amazon...",
    )
    st.session_state.screener_search_term = search_term

    # Quick filter buttons
    qf_cols = st.columns(6)
    with qf_cols[0]:
        show_all = st.button("All Stocks", use_container_width=True)
    with qf_cols[1]:
        show_above = st.button("Above Bar Only", use_container_width=True)
    with qf_cols[2]:
        show_buy = st.button("Buy Candidates", use_container_width=True,
                             help="Above Bar + MoS > 20%")
    with qf_cols[3]:
        show_under = st.button("Undervalued", use_container_width=True,
                               help="MoS > 15%")
    with qf_cols[4]:
        show_ai = st.button("AI Opportunity", use_container_width=True,
                            help="Tech + AI-exposed stocks with strong scores")
    with qf_cols[5]:
        show_dd_buy = st.button("DD Buy/Strong Buy", use_container_width=True,
                                help="Filter to Deep Dive Buy or Strong Buy only")

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
    if "Scenario" in table_df.columns:
        table_df["Scenario"] = table_df["Scenario"].apply(
            lambda x: f"{x:.2f}x" if pd.notna(x) else ""
        )

    st.dataframe(
        table_df,
        use_container_width=True,
        height=600,
        hide_index=True,
    )
    st.caption(f"Showing {len(table_df)} of {len(master)} stocks")

    # Single match — quick navigate
    if len(df) == 1:
        only_ticker = df["ticker"].iloc[0]
        if st.button(f"View Detail → {only_ticker}", type="primary", key="search_go_detail"):
            st.session_state.selected_ticker = only_ticker
            st.session_state.page = "Stock Detail"
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

                # Single stock — go straight to detail view
                if len(run_list) == 1 and succeeded == 1:
                    st.session_state.selected_ticker = run_list[0]
                    st.session_state.page = "Stock Detail"
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
                        st.session_state.page = "Stock Detail"
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
        st.session_state.page = "Stock Detail"
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: STOCK DETAIL (Tabbed)
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "Stock Detail":

    if st.button("← Back to Screener"):
        st.session_state.page = "Screener"
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

    st.markdown(f"**Quality Rating:** {_rating_badge(rating)}")

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

    _DD_STEP_LABELS = [
        "Fetching SEC documents...",
        "Assessing business quality...",
        "Reconciling metrics...",
        "Synthesizing investment thesis...",
    ]

    if dd_run or dd_refresh:
        progress_bar = st.progress(0, text="Starting deep dive analysis...")
        status_text = st.empty()

        def _dd_progress(step_idx, step_label):
            label = _DD_STEP_LABELS[step_idx] if step_idx < len(_DD_STEP_LABELS) else step_label
            n = len(_DD_STEP_LABELS)
            frac = (step_idx + 1) / n
            progress_bar.progress(frac, text=f"Step {step_idx + 1}/{n}: {label}")
            status_text.text(f"Step {step_idx + 1}/{n}: {label}")

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

    # ── Tabbed Detail View ──────────────────────────────────────────
    tab_names = ["Valuation", "Scores", "Quality Rating", "SEC Intelligence", "Raw Data"]
    if dd_result and "error" not in dd_result:
        tab_names.append("Deep Dive")
    tabs = st.tabs(tab_names)
    tab_val = tabs[0]
    tab_scores = tabs[1]
    tab_quality = tabs[2]
    tab_sec = tabs[3]
    tab_raw = tabs[4]
    tab_dd = tabs[5] if len(tabs) > 5 else None

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
                    st.dataframe(yr_df, hide_index=True, use_container_width=True)

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
                # Highlight cells where IV > current price
                def _highlight_undervalued(val_cell):
                    if price and val_cell > price:
                        return "background-color: #d4edda"
                    return ""

                st.dataframe(
                    sens_df.style.map(_highlight_undervalued),
                    use_container_width=True,
                )
                st.caption("Green = intrinsic value > current price")

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

        # Historical charts
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
                        st.dataframe(
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
                st.dataframe(financials, use_container_width=True)
        if balance_sheet is not None and not balance_sheet.empty:
            with st.expander("Balance Sheet"):
                st.dataframe(balance_sheet, use_container_width=True)
        if cashflow is not None and not cashflow.empty:
            with st.expander("Cash Flow"):
                st.dataframe(cashflow, use_container_width=True)

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

            st.dataframe(
                change_df.style.map(_color_delta, subset=["Delta"]),
                hide_index=True,
                use_container_width=True,
            )

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
                    st.dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

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
                    st.dataframe(pd.DataFrame(hw_rows), hide_index=True, use_container_width=True)

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
        display = sector_df[["ticker", "company", "composite_score",
                              "margin_of_safety", "quality_rating"]].copy()
        display["quality_rating"] = display["quality_rating"].apply(_rating_badge)
        display["margin_of_safety"] = display["margin_of_safety"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        display.columns = ["Ticker", "Company", "Score", "MoS %", "Rating"]
        st.dataframe(display, hide_index=True, use_container_width=True)


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
            for w in result.get("winners", []):
                st.markdown(f"- 🟢 {w}")
        with wl_cols[1]:
            st.markdown("**Losers**")
            for l_ticker in result.get("losers", []):
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
            st.dataframe(pd.DataFrame(si_rows), hide_index=True,
                         use_container_width=True)

        # Specific ticker impacts
        ticker_impacts = result.get("specific_tickers", {})
        if ticker_impacts:
            st.subheader("Specific Stock Impacts")
            ti_rows = []
            for tk, info_dict in ticker_impacts.items():
                ti_rows.append({
                    "Ticker": tk,
                    "Multiplier": f"{info_dict.get('multiplier', 1.0):.2f}x",
                    "Reasoning": info_dict.get("reasoning", ""),
                })
            st.dataframe(pd.DataFrame(ti_rows), hide_index=True,
                         use_container_width=True)

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
            st.session_state.page = "Screener"
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
                    st.dataframe(risk_df, hide_index=True, use_container_width=True)

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
                st.dataframe(
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
            st.dataframe(preview_df, hide_index=True, use_container_width=True)

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
        _dd_rows = []
        # Sort by recommendation strength
        _rec_order = {"Strong Buy": 0, "Buy": 1, "Hold": 2, "Avoid": 3, "Strong Avoid": 4}
        for t, dd in st.session_state.deep_dive_cache.items():
            thesis = dd.get("investment_thesis", {})
            rec = thesis.get("recommendation", "N/A")
            _dd_rows.append({
                "Ticker": t,
                "Recommendation": rec,
                "Conviction": thesis.get("conviction_level", "N/A"),
                "Thesis": thesis.get("one_line_thesis", ""),
                "_sort": _rec_order.get(rec, 5),
            })
        _dd_summary = pd.DataFrame(_dd_rows).sort_values("_sort").drop(columns=["_sort"])
        st.dataframe(_dd_summary, hide_index=True, use_container_width=True)

    # Show summary of all cached SEC analyses
    if st.session_state.sec_cache:
        st.subheader("Cached SEC Analysis Summary")
        summary_rows = []
        for t, sd in st.session_state.sec_cache.items():
            a10k = sd.get("analysis_10k", {}) or {}
            aproxy = sd.get("analysis_proxy", {}) or {}
            acred = sd.get("management_credibility", {}) or {}
            summary_rows.append({
                "Ticker": t,
                "SEC Flag": _sec_flag_badge(sd.get("overall_sec_flag", "green")),
                "10-K Sentiment": a10k.get("overall_sentiment", "N/A"),
                "Governance": (aproxy.get("governance_quality", {}) or {}).get("overall_grade", "N/A"),
                "Credibility": acred.get("credibility_score", "N/A"),
            })
        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows),
                hide_index=True,
                use_container_width=True,
            )
