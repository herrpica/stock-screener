# S&P 500 Stock Screener

A Streamlit-based stock screening application that scores, rates, and values all S&P 500 stocks using fundamental analysis, AI-powered scenario modeling, and SEC filing intelligence.

## Features

### 1. Screener
- Score every S&P 500 stock 0–100 on four pillars: Quality, Growth, Value, Sentiment
- Sector-relative percentile ranking with adjustable pillar weights
- Quick filters: All, Above Bar, Buy Candidates, Undervalued, AI Opportunity
- Scenario overlay with multiplier-adjusted intrinsic values

### 2. Stock Detail (Tabbed)
- **Valuation**: Two-stage DCF model with sensitivity table
- **Scores**: Radar chart + pillar breakdowns
- **Quality Rating**: Above/At/Below Bar with component scores
- **SEC Intelligence**: AI-powered 10-K, proxy, and 8-K analysis
- **Raw Data**: Full yfinance data inspection

### 3. Sector Overview
- Average composite score by sector
- Buy candidate counts
- Average margin of safety
- Top 5 stocks per sector

### 4. Scenario Analysis
- Natural language geopolitical/macro scenario input
- Claude-powered sector and stock-level impact multipliers
- Second-order effects and AI exposure analysis
- Preset scenarios for common macro events
- Scenario history for quick recall
- Apply to screener for instant portfolio re-evaluation

### 5. SEC Intelligence Dashboard
- SEC EDGAR filing retrieval (10-K, 10-Q, 8-K, DEF 14A)
- Claude-powered narrative analysis of annual reports
- Management credibility scoring (stated vs actual goals)
- Governance quality assessment from proxy statements
- Material event timeline from 8-K filings
- Batch analysis for up to 10 stocks at once

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add:
   - `ANTHROPIC_API_KEY` — your Anthropic API key (required for scenario analysis and SEC intelligence)
   - `SEC_USER_AGENT` — your name and email for SEC EDGAR compliance (recommended)

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **First run** takes 15–25 minutes to fetch all S&P 500 data from yfinance. Subsequent runs use a 24-hour pickle cache and load in seconds.

## Architecture

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application (5 pages) |
| `data_fetcher.py` | S&P 500 data fetching + pickle cache |
| `valuation_engine.py` | Two-stage DCF intrinsic value model |
| `scoring_engine.py` | 4-pillar sector-relative scoring |
| `quality_rater.py` | Historical consistency rating |
| `scenario_engine.py` | Claude-powered macro scenario analysis |
| `sec_fetcher.py` | SEC EDGAR filing retrieval + Claude analysis |

## Valuation Model

- **Stage 1 (Years 1–5):** Grow EPS at observed CAGR (capped 25%, floored -5%)
- **Stage 2 (Years 6–10):** Grow at min(observed, 8%)
- **Terminal Value:** Year 10 EPS × 15 (conservative P/E exit)
- **Discount Rate:** Adjustable via sidebar slider (default 10%)

## Scoring Model

All metrics ranked within sector as percentiles (0–100), with outliers clipped at 1st/99th percentile.

| Pillar | Default Weight | Key Metrics |
|--------|---------------|-------------|
| Quality | 35% | ROE, profit margin, D/E, ROA, interest coverage |
| Growth | 30% | Revenue growth, earnings growth, FCF growth, forward EPS |
| Value | 25% | Trailing P/E, EV/EBITDA, P/B (inverted) |
| Sentiment | 10% | % of 52-week high, MA50/MA200, price momentum |

## Quality Rating

| Component | Weight |
|-----------|--------|
| Earnings Consistency | 40% |
| Debt Discipline | 25% |
| Dividend Quality | 20% |
| Buyback Discipline | 15% |

- **Above Bar** ≥ 75 | **At Bar** 50–74 | **Below Bar** < 50

## Data Sources

- **Stock data:** Yahoo Finance via `yfinance`
- **S&P 500 list:** Wikipedia
- **SEC filings:** SEC EDGAR API
- **AI analysis:** Anthropic Claude API
