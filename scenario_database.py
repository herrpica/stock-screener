"""
Scenario Database — SQLite persistence for superforecasting scenarios.

Database file: superforecast.db (auto-initialized on import).

Tables:
  - scenarios: Full scenario lifecycle (brief → tree → opportunities → resolution)
  - recommendations: Individual instrument recommendations per scenario
  - calibration_records: Per-branch prediction vs outcome tracking
"""

import json
import logging
import pathlib
import sqlite3
import uuid
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH = pathlib.Path(__file__).parent / "superforecast.db"


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scenarios (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                raw_input TEXT,
                brief_json TEXT,
                tree_json TEXT,
                opportunities_json TEXT,
                report_json TEXT,
                status TEXT DEFAULT 'active',
                resolved_at TIMESTAMP,
                actual_branch TEXT,
                resolution_notes TEXT,
                brier_score REAL,
                tags TEXT
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                id TEXT PRIMARY KEY,
                scenario_id TEXT,
                ticker TEXT,
                direction TEXT,
                expected_impact_pct REAL,
                variance_score REAL,
                conviction_score REAL,
                causal_mechanism TEXT,
                actual_return_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
            );

            CREATE TABLE IF NOT EXISTS calibration_records (
                id TEXT PRIMARY KEY,
                scenario_id TEXT,
                branch_id TEXT,
                predicted_probability REAL,
                outcome_occurred INTEGER,
                brier_contribution REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
            );
        """)
        conn.commit()
    finally:
        conn.close()


# Initialize on import
_init_db()


# ── Scenario CRUD ────────────────────────────────────────────────────────────

def save_scenario(scenario_dict: dict) -> str:
    """Save a new scenario. Returns scenario_id (UUID)."""
    scenario_id = str(uuid.uuid4())
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO scenarios (id, title, raw_input, brief_json,
               tree_json, opportunities_json, report_json, status, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
            (
                scenario_id,
                scenario_dict.get("title", "Untitled Scenario"),
                scenario_dict.get("raw_input", ""),
                scenario_dict.get("brief_json", "{}"),
                scenario_dict.get("tree_json", "{}"),
                scenario_dict.get("opportunities_json", "{}"),
                scenario_dict.get("report_json", "{}"),
                scenario_dict.get("tags", ""),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return scenario_id


def get_scenario(scenario_id: str) -> dict | None:
    """Fetch a single scenario by ID."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM scenarios WHERE id = ?", (scenario_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_scenarios(status: str | None = None) -> list:
    """Fetch all scenarios, optionally filtered by status."""
    conn = _get_conn()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM scenarios WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM scenarios ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_scenario_results(
    scenario_id: str,
    tree: dict,
    opportunities: dict,
    report: dict,
):
    """Update a scenario with analysis results (tree, opportunities, report)."""
    conn = _get_conn()
    try:
        conn.execute(
            """UPDATE scenarios
               SET tree_json = ?, opportunities_json = ?, report_json = ?
               WHERE id = ?""",
            (
                json.dumps(tree, default=str),
                json.dumps(opportunities, default=str),
                json.dumps(report, default=str),
                scenario_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def resolve_scenario(
    scenario_id: str,
    actual_branch: str,
    notes: str,
) -> dict:
    """Resolve a scenario: set outcome, calculate Brier scores.

    Parameters
    ----------
    scenario_id : str
    actual_branch : str
        The branch ID that actually materialized.
    notes : str
        Resolution notes.

    Returns
    -------
    Updated scenario dict with brier_score.
    """
    scenario = get_scenario(scenario_id)
    if not scenario:
        return {"error": f"Scenario {scenario_id} not found"}

    tree = json.loads(scenario.get("tree_json", "{}"))
    branches = tree.get("branches", [])

    # Calculate Brier score for each branch
    brier_contributions = []
    conn = _get_conn()
    try:
        for branch in branches:
            branch_id = branch.get("id", "")
            predicted_prob = branch.get("probability", 0)
            outcome = 1 if branch_id == actual_branch else 0
            brier_contrib = (predicted_prob - outcome) ** 2
            brier_contributions.append(brier_contrib)

            # Insert calibration record
            cal_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO calibration_records
                   (id, scenario_id, branch_id, predicted_probability,
                    outcome_occurred, brier_contribution)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cal_id, scenario_id, branch_id, predicted_prob,
                 outcome, brier_contrib),
            )

        # Overall Brier score = mean of squared errors
        brier_score = (
            sum(brier_contributions) / len(brier_contributions)
            if brier_contributions else None
        )

        # Update scenario
        conn.execute(
            """UPDATE scenarios
               SET status = 'resolved', resolved_at = ?, actual_branch = ?,
                   resolution_notes = ?, brier_score = ?
               WHERE id = ?""",
            (
                datetime.now().isoformat(),
                actual_branch,
                notes,
                brier_score,
                scenario_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return get_scenario(scenario_id)


# ── Recommendations ──────────────────────────────────────────────────────────

def save_recommendations(scenario_id: str, opportunities: list):
    """Save instrument recommendations for a scenario."""
    conn = _get_conn()
    try:
        for opp in opportunities:
            rec_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO recommendations
                   (id, scenario_id, ticker, direction, expected_impact_pct,
                    variance_score, conviction_score, causal_mechanism)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec_id,
                    scenario_id,
                    opp.get("ticker", ""),
                    opp.get("direction", "neutral"),
                    opp.get("expected_impact_pct"),
                    opp.get("variance_score"),
                    opp.get("conviction_score"),
                    opp.get("causal_mechanism", ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def update_actual_returns(scenario_id: str, resolution_date: str):
    """Fetch actual prices via yfinance and update actual_return_pct.

    Parameters
    ----------
    scenario_id : str
    resolution_date : str
        Date string (YYYY-MM-DD) to measure returns against creation date.
    """
    scenario = get_scenario(scenario_id)
    if not scenario:
        return

    conn = _get_conn()
    try:
        recs = conn.execute(
            "SELECT * FROM recommendations WHERE scenario_id = ?",
            (scenario_id,),
        ).fetchall()
    finally:
        conn.close()

    if not recs:
        return

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available for return calculation")
        return

    creation_date = scenario["created_at"][:10]

    conn = _get_conn()
    try:
        for rec in recs:
            ticker = rec["ticker"]
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=creation_date, end=resolution_date)
                if len(hist) >= 2:
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    if start_price > 0:
                        actual_return = ((end_price - start_price) / start_price) * 100
                        conn.execute(
                            "UPDATE recommendations SET actual_return_pct = ? WHERE id = ?",
                            (round(actual_return, 2), rec["id"]),
                        )
            except Exception as e:
                logger.warning(f"Return calc failed for {ticker}: {e}")

        conn.commit()
    finally:
        conn.close()


# ── Calibration data ─────────────────────────────────────────────────────────

def get_calibration_data() -> pd.DataFrame:
    """Join scenarios and calibration_records for calibration analysis.

    Returns DataFrame with columns:
        scenario_id, branch_id, predicted_probability, outcome_occurred,
        brier_contribution, scenario_title, created_at
    """
    conn = _get_conn()
    try:
        query = """
            SELECT c.scenario_id, c.branch_id, c.predicted_probability,
                   c.outcome_occurred, c.brier_contribution,
                   s.title as scenario_title, c.created_at
            FROM calibration_records c
            JOIN scenarios s ON c.scenario_id = s.id
            ORDER BY c.created_at DESC
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


def calculate_brier_score(predictions: list, outcomes: list) -> float:
    """Calculate Brier score: mean((probability - outcome)^2).

    Parameters
    ----------
    predictions : list of float
        Predicted probabilities.
    outcomes : list of int
        Binary outcomes (0 or 1).

    Returns
    -------
    float : Brier score (lower is better, 0 = perfect, 0.25 = random).
    """
    if not predictions or len(predictions) != len(outcomes):
        return 0.25  # random baseline

    return sum(
        (p - o) ** 2 for p, o in zip(predictions, outcomes)
    ) / len(predictions)
