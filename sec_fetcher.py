"""
SEC EDGAR filing fetcher and Claude-powered narrative analysis.

- Maps tickers to CIK numbers via SEC EDGAR company tickers JSON
- Fetches recent 10-K, 10-Q, 8-K, DEF 14A filings
- Extracts key sections (MD&A, Risk Factors, etc.) from 10-K filings
- Analyzes narratives using Claude for red flags, sentiment, and credibility
"""

import os
import re
import json
import time
import logging
import pathlib
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# SEC EDGAR requires: "AppName/Version email@domain.com" (no parentheses)
SEC_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "stockscreener@example.com")

_last_request_time = 0.0


def _create_edgar_session() -> requests.Session:
    """Create a requests session configured for SEC EDGAR compliance."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": f"StockScreener/1.0 {SEC_EMAIL}",
        "Accept-Encoding": "gzip, deflate",
    })
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


EDGAR_SESSION = _create_edgar_session()


def _rate_limited_get(url: str, timeout: int = 30) -> requests.Response:
    """GET with rate limiting (~6 req/sec) and per-host Host header."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 0.15:
        time.sleep(0.15 - elapsed)
    resp = EDGAR_SESSION.get(url, timeout=timeout)
    _last_request_time = time.time()
    return resp


# ── CIK Lookup ────────────────────────────────────────────────────────────

_CACHE_DIR = pathlib.Path(__file__).parent.parent / ".cache"
_TICKERS_CACHE_FILE = _CACHE_DIR / "company_tickers.json"
_TICKERS_MAX_AGE_SECS = 86400  # 24 hours


def _load_company_tickers() -> dict:
    """Load company tickers JSON, using a local file cache (refreshed every 24h)."""
    # Try local cache first
    if _TICKERS_CACHE_FILE.exists():
        age = time.time() - _TICKERS_CACHE_FILE.stat().st_mtime
        if age < _TICKERS_MAX_AGE_SECS:
            with open(_TICKERS_CACHE_FILE, "r") as f:
                return json.load(f)

    # Fetch fresh from SEC
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = _rate_limited_get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Save to cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_TICKERS_CACHE_FILE, "w") as f:
        json.dump(data, f)

    return data


_CIK_CACHE: dict = {}


def get_cik(ticker: str) -> str | None:
    """Map ticker to SEC CIK number.

    Uses SEC EDGAR company_tickers.json endpoint. Caches results in memory.
    """
    ticker = ticker.upper()
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    try:
        data = _load_company_tickers()

        for entry in data.values():
            t = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", ""))
            _CIK_CACHE[t] = cik.zfill(10)

        return _CIK_CACHE.get(ticker)
    except Exception as e:
        logger.warning(f"CIK lookup failed for {ticker}: {e}")
        return None


# ── Filing Retrieval ──────────────────────────────────────────────────────

FILING_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A"]


def get_recent_filings(
    ticker: str,
    filing_types: list[str] | None = None,
    count: int = 10,
) -> list[dict] | None:
    """Get recent filings for a ticker from SEC EDGAR.

    Returns list of dicts with: accession_number, filing_type, filing_date,
    primary_document, description.
    """
    cik = get_cik(ticker)
    if not cik:
        return None

    filing_types = filing_types or FILING_TYPES

    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = _rate_limited_get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings = []
        for i in range(len(forms)):
            if forms[i] in filing_types:
                accession = accessions[i].replace("-", "")
                filings.append({
                    "accession_number": accessions[i],
                    "filing_type": forms[i],
                    "filing_date": dates[i],
                    "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "url": (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik.lstrip('0')}/{accession}/{primary_docs[i]}"
                    ) if i < len(primary_docs) and primary_docs[i] else None,
                })
                if len(filings) >= count:
                    break

        return filings

    except Exception as e:
        logger.warning(f"Failed to get filings for {ticker}: {e}")
        return None


def get_filing_document(filing: dict, max_chars: int = 100_000) -> str | None:
    """Fetch the text content of a filing document.

    Fetches the primary document and strips HTML tags for text extraction.
    Truncates to max_chars to avoid overwhelming Claude.
    """
    url = filing.get("url")
    if not url:
        return None

    try:
        resp = _rate_limited_get(url, timeout=30)
        resp.raise_for_status()
        text = resp.text

        # Strip HTML tags for cleaner text
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars]
    except Exception as e:
        logger.warning(f"Failed to fetch filing document: {e}")
        return None


# ── Section Extraction ────────────────────────────────────────────────────

# Common 10-K section markers
_SECTION_PATTERNS = {
    "risk_factors": [
        r"item\s*1a[\.\s]*risk\s*factors",
        r"risk\s*factors",
    ],
    "mda": [
        r"item\s*7[\.\s]*management.s\s*discussion",
        r"management.s\s*discussion\s*and\s*analysis",
    ],
    "business": [
        r"item\s*1[\.\s]*business(?!\s*risk)",
        r"description\s*of\s*business",
    ],
    "legal": [
        r"item\s*3[\.\s]*legal\s*proceedings",
        r"legal\s*proceedings",
    ],
}


def extract_section(
    filing_text: str,
    section: str,
    max_chars: int = 30_000,
) -> str | None:
    """Extract a named section from a 10-K filing text.

    section: one of 'risk_factors', 'mda', 'business', 'legal'
    """
    patterns = _SECTION_PATTERNS.get(section, [])
    if not patterns:
        return None

    text_lower = filing_text.lower()

    # Find section start
    start_pos = None
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            start_pos = match.start()
            break

    if start_pos is None:
        return None

    # Find next section (Item N+1) as end boundary
    remaining = text_lower[start_pos + 100:]
    next_item = re.search(r"item\s*\d+[a-z]?[\.\s]", remaining)
    if next_item:
        end_pos = start_pos + 100 + next_item.start()
    else:
        end_pos = start_pos + max_chars

    extracted = filing_text[start_pos:end_pos].strip()
    return extracted[:max_chars]


# ── Claude Analysis Functions ─────────────────────────────────────────────

def _extract_json(text: str, call_name: str = "SEC Claude call") -> dict:
    """Robust JSON extraction with multiple fallback strategies."""
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
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


def _call_claude(system: str, user_prompt: str, api_key: str, call_name: str = "SEC analysis") -> dict | None:
    """Helper to call Claude and parse JSON response with robust extraction."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _extract_json(response.content[0].text, call_name)
    except anthropic.APIError as e:
        logger.warning(f"Anthropic API error: {e}")
        return {"error": f"API error: {e}"}
    except Exception as e:
        logger.warning(f"Claude analysis failed: {e}")
        return {"error": f"Analysis failed: {e}"}


def analyze_10k_narrative(
    filing_text: str,
    ticker: str,
    api_key: str,
) -> dict | None:
    """Analyze 10-K narrative sections for red flags and sentiment.

    Returns dict with: overall_sentiment, red_flags, key_risks,
    management_tone, forward_guidance, notable_changes
    """
    system = """You are an expert SEC filing analyst. Analyze the 10-K filing text
for investment-relevant signals.

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "overall_sentiment": "positive|neutral|cautious|negative",
  "red_flags": [
    {"flag": "description", "severity": "high|medium|low", "section": "where found"}
  ],
  "key_risks": [
    {"risk": "description", "category": "operational|financial|regulatory|market|legal", "severity": "high|medium|low"}
  ],
  "management_tone": {
    "confidence_level": "high|medium|low",
    "hedging_language_pct": 0.15,
    "notable_phrases": ["phrase1", "phrase2"]
  },
  "forward_guidance": {
    "outlook": "positive|neutral|cautious|negative",
    "key_initiatives": ["initiative1"],
    "capex_signals": "increasing|stable|decreasing"
  },
  "notable_changes": [
    {"change": "description", "significance": "high|medium|low"}
  ],
  "sec_flag": "green|yellow|red"
}

Rules:
- sec_flag: green = no material concerns, yellow = some caution warranted, red = significant red flags
- Be specific with red flags — cite what triggered the concern
- Keep descriptions concise (1-2 sentences each)
- hedging_language_pct is approximate ratio of hedging/uncertain language"""

    # Extract key sections for analysis
    sections_text = []
    for section_name in ["mda", "risk_factors", "business"]:
        extracted = extract_section(filing_text, section_name)
        if extracted:
            sections_text.append(f"=== {section_name.upper()} ===\n{extracted[:10000]}")

    if not sections_text:
        # Fall back to raw text
        sections_text = [filing_text[:30000]]

    user_prompt = (
        f"Ticker: {ticker}\n\n"
        f"10-K Filing Excerpts:\n\n{''.join(sections_text[:30000])}"
    )

    return _call_claude(system, user_prompt, api_key, call_name="10-K narrative")


def analyze_proxy_statement(
    filing_text: str,
    ticker: str,
    api_key: str,
) -> dict | None:
    """Analyze DEF 14A (proxy statement) for governance and compensation signals.

    Returns dict with: executive_compensation, governance_quality,
    shareholder_concerns, related_party_flags
    """
    system = """You are an expert proxy statement analyst. Analyze the DEF 14A filing
for corporate governance and compensation signals.

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "executive_compensation": {
    "ceo_total_comp": "approximate total",
    "pay_vs_performance_alignment": "strong|moderate|weak|poor",
    "stock_based_pct": 0.60,
    "concerns": ["concern1"]
  },
  "governance_quality": {
    "board_independence_pct": 0.80,
    "dual_class_shares": false,
    "poison_pill": false,
    "staggered_board": false,
    "overall_grade": "A|B|C|D|F"
  },
  "shareholder_concerns": [
    {"concern": "description", "severity": "high|medium|low"}
  ],
  "related_party_flags": [
    {"flag": "description", "severity": "high|medium|low"}
  ],
  "governance_flag": "green|yellow|red"
}"""

    user_prompt = (
        f"Ticker: {ticker}\n\n"
        f"Proxy Statement (DEF 14A) Excerpts:\n\n{filing_text[:30000]}"
    )

    return _call_claude(system, user_prompt, api_key, call_name="proxy statement")


def analyze_recent_8k(
    filings_text: list[dict],
    ticker: str,
    api_key: str,
) -> dict | None:
    """Analyze recent 8-K filings for material events.

    filings_text: list of {"date": "...", "text": "..."}

    Returns dict with: material_events, overall_signal, event_timeline
    """
    system = """You are an expert at analyzing 8-K material event filings.
Analyze the recent 8-K filings and identify material events.

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "material_events": [
    {
      "date": "2024-01-15",
      "event_type": "leadership_change|acquisition|divestiture|earnings|legal|restructuring|other",
      "description": "brief description",
      "impact": "positive|neutral|negative",
      "significance": "high|medium|low"
    }
  ],
  "overall_signal": "positive|neutral|cautious|negative",
  "event_summary": "1-2 sentence summary of recent 8-K activity",
  "event_flag": "green|yellow|red"
}"""

    combined = ""
    for f in filings_text[:5]:  # Limit to 5 most recent
        combined += f"\n=== 8-K Filed {f['date']} ===\n{f['text'][:5000]}\n"

    user_prompt = (
        f"Ticker: {ticker}\n\n"
        f"Recent 8-K Filings:\n{combined[:30000]}"
    )

    return _call_claude(system, user_prompt, api_key, call_name="8-K analysis")


def check_management_credibility(
    current_10k_text: str,
    prior_10k_text: str | None,
    ticker: str,
    api_key: str,
) -> dict | None:
    """Compare stated goals vs actual outcomes across two 10-K filings.

    Returns dict with: credibility_score (0-100), promises_kept,
    promises_broken, goal_tracking
    """
    system = """You are an expert at evaluating management credibility by comparing
what management said they would do (in prior filings) vs what actually happened
(in current filings).

CRITICAL: Your response must be ONLY valid JSON. No explanatory text before or after.
No markdown code fences. Start with { and end with }. Complete the entire structure.

{
  "credibility_score": 75,
  "promises_kept": [
    {"promise": "what was stated", "outcome": "what happened", "grade": "A|B|C|D|F"}
  ],
  "promises_broken": [
    {"promise": "what was stated", "outcome": "what happened", "severity": "high|medium|low"}
  ],
  "goal_tracking": {
    "revenue_targets": "met|partially_met|missed|not_stated",
    "margin_targets": "met|partially_met|missed|not_stated",
    "capex_plans": "met|partially_met|missed|not_stated",
    "strategic_initiatives": "met|partially_met|missed|not_stated"
  },
  "credibility_flag": "green|yellow|red",
  "summary": "1-2 sentence credibility assessment"
}

Rules:
- credibility_score: 0-100 where 100 = management delivers on all promises
- Only score promises that were clearly stated
- If no prior filing provided, score based on internal consistency of current filing
- Be fair — recognize industry headwinds when relevant"""

    user_prompt = f"Ticker: {ticker}\n\n"
    user_prompt += f"=== CURRENT 10-K ===\n{current_10k_text[:15000]}\n\n"
    if prior_10k_text:
        user_prompt += f"=== PRIOR 10-K ===\n{prior_10k_text[:15000]}"
    else:
        user_prompt += "(No prior 10-K available — assess internal consistency only)"

    return _call_claude(system, user_prompt, api_key, call_name="management credibility")


def get_full_sec_analysis(
    ticker: str,
    api_key: str,
) -> dict:
    """Run full SEC analysis pipeline for a ticker.

    Fetches filings and runs all relevant Claude analyses.

    Returns dict with:
        filings_found, analysis_10k, analysis_proxy, analysis_8k,
        management_credibility, overall_sec_flag, errors
    """
    result = {
        "ticker": ticker,
        "filings_found": [],
        "analysis_10k": None,
        "analysis_proxy": None,
        "analysis_8k": None,
        "management_credibility": None,
        "overall_sec_flag": "green",
        "errors": [],
        "timestamp": datetime.now().isoformat(),
    }

    # Get recent filings
    filings = get_recent_filings(ticker, count=20)
    if not filings:
        result["errors"].append("No filings found in SEC EDGAR")
        return result

    result["filings_found"] = [
        {"type": f["filing_type"], "date": f["filing_date"]}
        for f in filings[:10]
    ]

    # Categorize filings
    ten_ks = [f for f in filings if f["filing_type"] == "10-K"]
    proxies = [f for f in filings if f["filing_type"] == "DEF 14A"]
    eight_ks = [f for f in filings if f["filing_type"] == "8-K"]

    flags = []

    # Analyze most recent 10-K
    if ten_ks:
        current_10k_text = get_filing_document(ten_ks[0])
        if current_10k_text:
            analysis = analyze_10k_narrative(current_10k_text, ticker, api_key)
            if analysis and "error" not in analysis:
                result["analysis_10k"] = analysis
                flags.append(analysis.get("sec_flag", "green"))

            # Management credibility (compare with prior 10-K if available)
            prior_text = None
            if len(ten_ks) > 1:
                prior_text = get_filing_document(ten_ks[1])
            credibility = check_management_credibility(
                current_10k_text, prior_text, ticker, api_key
            )
            if credibility and "error" not in credibility:
                result["management_credibility"] = credibility
                flags.append(credibility.get("credibility_flag", "green"))
        else:
            result["errors"].append("Could not fetch 10-K document text")

    # Analyze most recent proxy statement
    if proxies:
        proxy_text = get_filing_document(proxies[0])
        if proxy_text:
            analysis = analyze_proxy_statement(proxy_text, ticker, api_key)
            if analysis and "error" not in analysis:
                result["analysis_proxy"] = analysis
                flags.append(analysis.get("governance_flag", "green"))
        else:
            result["errors"].append("Could not fetch proxy document text")

    # Analyze recent 8-Ks
    if eight_ks:
        eight_k_texts = []
        for f in eight_ks[:5]:
            text = get_filing_document(f, max_chars=10_000)
            if text:
                eight_k_texts.append({"date": f["filing_date"], "text": text})
        if eight_k_texts:
            analysis = analyze_recent_8k(eight_k_texts, ticker, api_key)
            if analysis and "error" not in analysis:
                result["analysis_8k"] = analysis
                flags.append(analysis.get("event_flag", "green"))

    # Compute overall SEC flag (worst of all flags)
    flag_severity = {"green": 0, "yellow": 1, "red": 2}
    worst = max(flags, key=lambda f: flag_severity.get(f, 0)) if flags else "green"
    result["overall_sec_flag"] = worst

    return result


# ── Diagnostic Test ──────────────────────────────────────────────────────

def test_edgar_connection():
    """Test connectivity to SEC EDGAR endpoints."""
    print(f"User-Agent: {EDGAR_SESSION.headers['User-Agent']}")
    print(f"SEC_EMAIL:  {SEC_EMAIL}")
    print()
    endpoints = [
        ("EDGAR Search", "https://efts.sec.gov/LATEST/search-index?q=AMZN&dateRange=custom&startdt=2024-01-01&enddt=2024-12-31&forms=10-K"),
        ("Submissions", "https://data.sec.gov/submissions/CIK0001018724.json"),
        ("Company Facts", "https://data.sec.gov/api/xbrl/companyfacts/CIK0001018724.json"),
    ]
    for name, url in endpoints:
        try:
            resp = _rate_limited_get(url, timeout=15)
            print(f"{name}: {resp.status_code} ({len(resp.content)} bytes)")
            if resp.status_code != 200:
                print(f"  Headers sent: {resp.request.headers}")
                print(f"  Response: {resp.text[:200]}")
        except Exception as e:
            print(f"{name}: ERROR - {e}")
