"""
Independent Verification Tests for src/db.py.
Verifies read-only safety, latency, FTS5 fallback, and provenance extraction.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import execute_query, fts_fallback_query, extract_provenance, sanitize_and_prepare_query


def test_valid_query_execution():
    """Test standard SELECT query with parameterization and latency check."""
    sql = """
        SELECT item_name, period_label, raw_value, value_vnd, page_no, source_doc
        FROM financial_facts
        WHERE ticker = 'VJC' AND year = 2018 AND report_type = 'separate'
          AND item_name_ascii LIKE '%lai tien gui%'
        ORDER BY page_no ASC
        LIMIT 5
    """
    res = execute_query(sql)
    assert res["status"] == "success", f"Query failed: {res.get('error')}"
    assert res["row_count"] > 0, "No rows returned for VJC 2018"
    assert res["execution_time_ms"] < 100, f"Query too slow: {res['execution_time_ms']}ms"

    # Verify 208 tỷ row exists
    has_208 = any("208" in str(r["raw_value"]) for r in res["rows"])
    assert has_208, "Did not find expected 208 tỷ row for VJC 2018"
    print(f"  [PASS] test_valid_query_execution (Latency: {res['execution_time_ms']}ms, Rows: {res['row_count']})")


def test_security_blocked_mutations():
    """Test that destructive SQL operations are strictly blocked."""
    evil_queries = [
        "DROP TABLE financial_facts",
        "DELETE FROM financial_facts WHERE id = 1",
        "INSERT INTO financial_facts (ticker) VALUES ('FAKE')",
        "UPDATE financial_facts SET ticker = 'HACKED'",
        "ALTER TABLE financial_facts ADD COLUMN hacked TEXT",
        "SELECT 1; SELECT 2",
        "SELECT * FROM financial_facts; DROP TABLE financial_facts",
    ]

    for q in evil_queries:
        res = execute_query(q)
        assert res["status"] == "error", f"Evil query was not blocked: {q}"
        assert "Security Alert" in res["error"], f"Unexpected error message: {res['error']}"

    print(f"  [PASS] test_security_blocked_mutations ({len(evil_queries)} dangerous queries blocked)")


def test_auto_limit_enforcement():
    """Test that SELECT queries without LIMIT have DEFAULT_ROW_LIMIT appended."""
    query = "SELECT * FROM financial_facts WHERE ticker = 'HPG'"
    sanitized = sanitize_and_prepare_query(query)
    assert "LIMIT 10" in sanitized, f"LIMIT 10 was not enforced: {sanitized}"

    # Aggregate queries should NOT have LIMIT appended
    agg_query = "SELECT COUNT(*) FROM financial_facts"
    sanitized_agg = sanitize_and_prepare_query(agg_query)
    assert "LIMIT" not in sanitized_agg, f"LIMIT was incorrectly added to aggregate: {sanitized_agg}"

    print("  [PASS] test_auto_limit_enforcement")


def test_fts5_fallback():
    """Test full-text search fallback when exact matching is insufficient."""
    res = fts_fallback_query(ticker="ACB", keyword="thuong mai", year=2022, report_type="separate")
    assert res["status"] == "success", f"FTS query failed: {res.get('error')}"
    assert res["row_count"] > 0, "FTS returned 0 rows for ACB 2022 thuong mai"

    first_row = res["rows"][0]
    assert "thương mại" in first_row["item_name"].lower(), f"Unexpected item: {first_row['item_name']}"
    print(f"  [PASS] test_fts5_fallback (Rows found: {res['row_count']}, Latency: {res['execution_time_ms']}ms)")


def test_provenance_extraction():
    """Test that provenance records are extracted cleanly and deduplicated."""
    res = execute_query("""
        SELECT ticker, company_name, source_doc, page_no, statement, section_title, period_label, raw_value
        FROM financial_facts
        WHERE ticker = 'ACB' AND year = 2022 AND report_type = 'separate'
          AND item_name_ascii = 'thuong mai'
        ORDER BY page_no ASC
        LIMIT 2
    """)
    provenance = extract_provenance(res["rows"])
    assert len(provenance) > 0, "No provenance extracted"
    p = provenance[0]
    assert p["ticker"] == "ACB"
    assert p["page_no"] == 42
    assert "72.917.566" in p["raw_value"]
    print("  [PASS] test_provenance_extraction")


def run_all_tests():
    print("\n==================================================")
    print("RUNNING INDEPENDENT TESTS FOR src/db.py")
    print("==================================================")
    test_valid_query_execution()
    test_security_blocked_mutations()
    test_auto_limit_enforcement()
    test_fts5_fallback()
    test_provenance_extraction()
    print("==================================================")
    print("ALL TESTS PASSED PERFECTLY!\n")


if __name__ == "__main__":
    run_all_tests()
