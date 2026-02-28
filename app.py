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

    # Quick filter buttons
    qf_cols = st.columns(5)
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

    # Apply filters
    df = master.copy()

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

    # ── Tabbed Detail View ──────────────────────────────────────────
    tab_val, tab_scores, tab_quality, tab_sec, tab_raw = st.tabs([
        "Valuation", "Scores", "Quality Rating", "SEC Intelligence", "Raw Data"
    ])

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
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.warning("Set ANTHROPIC_API_KEY in .env to enable SEC analysis.")
        else:
            cached_sec = st.session_state.sec_cache.get(ticker)

            if cached_sec:
                _render_sec_analysis(ticker, cached_sec) if False else None

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
            else:
                progress.progress(
                    (i + 1) / len(batch_options),
                    text=f"Skipping {t} (cached)...",
                )
        progress.empty()
        st.success(f"Completed batch analysis for {len(batch_options)} stocks.")
        st.rerun()

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
