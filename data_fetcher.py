"""
Fetch and cache all S&P 500 fundamental and historical data using yfinance.

- Gets S&P 500 tickers from Wikipedia
- Fetches info, financials, balance_sheet, cashflow, dividends per ticker
- Caches as pickle with 24-hour TTL
- Shows st.progress during fetch, batches of 50 with 2s pause
"""

import os
import time
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

import io

import pandas as pd
import requests
import yfinance as yf
import streamlit as st

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_FILE = CACHE_DIR / "sp500_data.pkl"
FAILED_FILE = CACHE_DIR / "failed_tickers.txt"
CACHE_MAX_AGE = timedelta(hours=24)
BATCH_SIZE = 50
BATCH_PAUSE = 2  # seconds


def get_sp500_tickers() -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    # Normalize ticker: BRK.B -> BRK-B for yfinance
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={
            "Symbol": "ticker",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )


def _fetch_single_ticker(ticker: str) -> dict | None:
    """Fetch all data for a single ticker. Returns dict or None on failure."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if not info.get("regularMarketPrice") and not info.get("currentPrice"):
            return None

        financials = t.financials
        balance_sheet = t.balance_sheet
        cashflow = t.cashflow
        dividends = t.dividends

        return {
            "info": info,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cashflow": cashflow,
            "dividends": dividends,
        }
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {e}")
        return None


def _fetch_all_tickers(tickers_df: pd.DataFrame) -> dict:
    """Fetch data for all tickers with progress bar and batching."""
    tickers = tickers_df["ticker"].tolist()
    total = len(tickers)
    all_data = {}
    failed = []

    progress_bar = st.progress(0, text="Fetching S&P 500 data...")
    status_text = st.empty()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = tickers[batch_start:batch_end]

        for i, ticker in enumerate(batch):
            global_idx = batch_start + i
            status_text.text(
                f"Fetching {ticker} ({global_idx + 1}/{total})"
            )
            progress_bar.progress(
                (global_idx + 1) / total,
                text=f"Fetching {ticker} ({global_idx + 1}/{total})",
            )

            result = _fetch_single_ticker(ticker)
            if result is not None:
                # Attach metadata from the tickers_df
                row = tickers_df[tickers_df["ticker"] == ticker].iloc[0]
                result["company"] = row["company"]
                result["sector"] = row["sector"]
                result["sub_industry"] = row["sub_industry"]
                all_data[ticker] = result
            else:
                failed.append(ticker)

        # Pause between batches to avoid rate limiting
        if batch_end < total:
            time.sleep(BATCH_PAUSE)

    progress_bar.empty()
    status_text.empty()

    # Log failed tickers
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if failed:
        with open(FAILED_FILE, "w") as f:
            f.write(f"# Failed tickers — {datetime.now().isoformat()}\n")
            for t in failed:
                f.write(f"{t}\n")
        logger.warning(f"{len(failed)} tickers failed to fetch")

    return all_data


def _save_cache(data: dict):
    """Save data dict to pickle cache with timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(),
        "data": data,
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)


def _load_cache() -> dict | None:
    """Load cache if it exists and is less than 24 hours old."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        age = datetime.now() - payload["timestamp"]
        if age > CACHE_MAX_AGE:
            return None
        return payload
    except Exception:
        return None


def load_or_fetch_data(force_refresh: bool = False) -> dict:
    """Main entry point. Returns dict keyed by ticker with all data.

    Uses pickle cache with 24h TTL. Pass force_refresh=True to bypass cache.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            st.toast(
                f"Loaded cached data from {cached['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                icon="📦",
            )
            return cached["data"]

    # Fresh fetch
    st.info("Fetching S&P 500 data — this takes 15-25 minutes on first run...")
    tickers_df = get_sp500_tickers()
    data = _fetch_all_tickers(tickers_df)
    _save_cache(data)
    st.success(f"Fetched {len(data)} stocks successfully.")
    return data


def get_cache_timestamp() -> datetime | None:
    """Return the timestamp of the current cache, or None."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        return payload["timestamp"]
    except Exception:
        return None
