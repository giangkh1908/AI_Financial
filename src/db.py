"""
Database Execution Sandbox & Safety Guard.
Manages read-only connections, safety verification, query execution, and FTS5 fallback.
Zero external dependencies (uses Python standard library sqlite3).
"""

import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.config import DB_PATH, DEFAULT_ROW_LIMIT, MAX_ROW_LIMIT

# Blocked mutating SQL keywords for safety
DISALLOWED_KEYWORDS_REGEX = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)

# Detect aggregate queries where LIMIT is not needed
AGGREGATE_REGEX = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)


def get_connection() -> sqlite3.Connection:
    """Returns a thread-safe, read-only connection to the SQLite database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run scripts/build_db.py first.")

    uri = f"file:{DB_PATH.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def sanitize_and_prepare_query(query: str) -> str:
    """Validates query against SQL injection / mutation and enforces reasonable LIMIT."""
    cleaned = query.strip().rstrip(";")

    # Check for mutating statements
    match = DISALLOWED_KEYWORDS_REGEX.search(cleaned)
    if match:
        raise PermissionError(f"Security Alert: Mutating or restricted SQL keyword detected: '{match.group(0)}'")

    # Enforce SELECT
    if not cleaned.upper().startswith("SELECT") and not cleaned.upper().startswith("WITH"):
        raise PermissionError("Security Alert: Only SELECT / CTE queries are permitted.")

    # Apply LIMIT rule: if not aggregate and no LIMIT present, append DEFAULT_ROW_LIMIT
    has_limit = re.search(r"\bLIMIT\s+\d+", cleaned, re.IGNORECASE)
    is_aggregate = bool(AGGREGATE_REGEX.search(cleaned)) and not re.search(r"\bGROUP\s+BY\b", cleaned, re.IGNORECASE)

    if not has_limit and not is_aggregate:
        cleaned = f"{cleaned} LIMIT {DEFAULT_ROW_LIMIT}"

    return cleaned + ";"


def execute_query(query: str, params: tuple = ()) -> Dict[str, Any]:
    """
    Executes a sanitized read-only SQL query against financial.db.
    Returns rows as list of dicts along with latency and metadata.
    """
    start_time = time.perf_counter()

    try:
        safe_sql = sanitize_and_prepare_query(query)
    except Exception as err:
        return {
            "status": "error",
            "sql_query": query,
            "rows": [],
            "row_count": 0,
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "error": str(err),
        }

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(safe_sql, params)
        raw_rows = cur.fetchall()
        rows = [dict(r) for r in raw_rows]
        conn.close()

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "success",
            "sql_query": safe_sql,
            "rows": rows,
            "row_count": len(rows),
            "execution_time_ms": elapsed_ms,
            "error": None,
        }
    except Exception as err:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "sql_query": safe_sql if "safe_sql" in locals() else query,
            "rows": [],
            "row_count": 0,
            "execution_time_ms": elapsed_ms,
            "error": f"Execution failed: {err}",
        }


def fts_fallback_query(
    ticker: str,
    keyword: str,
    year: Optional[int] = None,
    report_type: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Full-Text Search (FTS5) fallback when direct B-Tree search returns 0 results.
    Matches variations of Vietnamese accounting item names.
    """
    start_time = time.perf_counter()

    # Clean FTS keyword (escape special characters)
    clean_keyword = re.sub(r'[^\w\s]', '', keyword).strip()
    if not clean_keyword:
        return {
            "status": "error",
            "sql_query": "FTS5 fallback empty keyword",
            "rows": [],
            "row_count": 0,
            "execution_time_ms": 0.0,
            "error": "Empty keyword for FTS search",
        }

    # Focus match on item_name_ascii and item_name to prevent matching bank names in section_title
    fts_match_expr = f'{{item_name_ascii item_name}} : "{clean_keyword}"'

    conditions = ["f.ticker = ?"]
    params = [ticker]

    if year:
        conditions.append("f.year = ?")
        params.append(year)

    if report_type:
        conditions.append("f.report_type = ?")
        params.append(report_type)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT f.id, f.ticker, f.company_name, f.year, f.report_type, f.statement,
               f.section_title, f.item_code, f.item_name, f.item_name_ascii,
               f.period_label, f.value_vnd, f.raw_value, f.unit, f.page_no, f.source_doc
        FROM financial_facts f
        JOIN facts_fts fts ON f.id = fts.rowid
        WHERE facts_fts MATCH ? AND {where_clause}
        ORDER BY f.page_no ASC
        LIMIT ?
    """

    all_params = (fts_match_expr, *params, limit)

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(sql, all_params)
        raw_rows = cur.fetchall()
        rows = [dict(r) for r in raw_rows]
        conn.close()

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "success",
            "sql_query": sql.strip(),
            "rows": rows,
            "row_count": len(rows),
            "execution_time_ms": elapsed_ms,
            "error": None,
        }
    except Exception as err:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "sql_query": sql.strip(),
            "rows": [],
            "row_count": 0,
            "execution_time_ms": elapsed_ms,
            "error": f"FTS5 Query failed: {err}",
        }


def extract_provenance(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates and standardizes provenance citations for audit trail."""
    provenance_list = []
    seen = set()

    for r in rows:
        key = (r.get("source_doc"), r.get("page_no"), r.get("raw_value"))
        if key not in seen and r.get("source_doc") and r.get("page_no"):
            seen.add(key)
            provenance_list.append({
                "ticker": r.get("ticker"),
                "company_name": r.get("company_name"),
                "source_doc": r.get("source_doc"),
                "page_no": r.get("page_no"),
                "statement": r.get("statement"),
                "section_title": r.get("section_title"),
                "period_label": r.get("period_label"),
                "raw_value": r.get("raw_value"),
            })

    return provenance_list
