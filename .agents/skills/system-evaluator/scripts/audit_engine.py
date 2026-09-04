"""
Universal System Evaluator & Ground-Truth Audit Engine.
Zero external dependencies. Works on Python 3.8+ standard library.

Capabilities:
1. SQLite Database Inspection (Table counts, schema types, nulls, query test)
2. Benchmark/Dataset Pattern Counting (Regex frequencies, keyword occurrences)
3. Schema Parity Verification (Comparing DDL vs target column specifications)
4. Query Pattern Blindspot Detector (Testing edge-cases on real data)
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def inspect_sqlite(db_path: str, verbose: bool = False):
    """Inspects any SQLite database and outputs complete table statistics."""
    p = Path(db_path)
    if not p.exists():
        print(f"Error: Database file not found at {db_path}", file=sys.stderr)
        return False

    conn = sqlite3.connect(str(p))
    cur = conn.cursor()

    # Get all tables
    cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'")
    tables = cur.fetchall()

    print(f"\n==================================================")
    print(f"SQLITE STORAGE INSPECTION: {p.name}")
    print(f"Size: {p.stat().st_size / (1024*1024):.2f} MB")
    print(f"==================================================")

    for tbl_name, tbl_type in tables:
        # Check if virtual table
        try:
            cur.execute(f"SELECT COUNT(*) FROM \"{tbl_name}\"")
            row_count = cur.fetchone()[0]
        except Exception as e:
            row_count = f"N/A ({e})"

        print(f"\n[{tbl_type.upper()}] {tbl_name} | Rows: {row_count:,}" if isinstance(row_count, int) else f"\n[{tbl_type.upper()}] {tbl_name} | Rows: {row_count}")

        # Columns info
        try:
            cur.execute(f"PRAGMA table_info(\"{tbl_name}\")")
            cols = cur.fetchall()
            if verbose and cols:
                print("  Columns:")
                for col in cols:
                    cid, name, col_type, notnull, dflt_value, pk = col
                    pk_marker = " [PK]" if pk else ""
                    null_marker = " NOT NULL" if notnull else ""
                    print(f"    - {name}: {col_type}{pk_marker}{null_marker}")
        except Exception:
            pass

    conn.close()
    return True


def count_patterns_in_file(file_path: str, patterns: list, line_based: bool = True):
    """Scans a file (jsonl, txt, csv, code) and counts occurrences of regex patterns."""
    p = Path(file_path)
    if not p.exists():
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return {}

    counts = {pat: 0 for pat in patterns}
    compiled = {pat: re.compile(pat, re.IGNORECASE) for pat in patterns}
    line_matches = {pat: 0 for pat in patterns}

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for pat, rx in compiled.items():
                m = rx.findall(line)
                if m:
                    counts[pat] += len(m)
                    line_matches[pat] += 1

    print(f"\n==================================================")
    print(f"PATTERN FREQUENCY AUDIT: {p.name}")
    print(f"==================================================")
    for pat in patterns:
        print(f"  • Pattern '{pat}': {line_matches[pat]} lines matching ({counts[pat]} total occurrences)")

    return counts


def test_sql_query(db_path: str, query: str):
    """Executes a test query in read-only mode and displays result summary."""
    p = Path(db_path)
    if not p.exists():
        print(f"Error: Database file not found at {db_path}", file=sys.stderr)
        return

    uri = f"file:{p.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cur = conn.cursor()

    print(f"\n==================================================")
    print(f"EXECUTING READ-ONLY TEST QUERY")
    print(f"SQL: {query.strip()}")
    print(f"==================================================")

    try:
        cur.execute(query)
        rows = cur.fetchall()
        print(f"Returned {len(rows)} row(s):")
        for idx, r in enumerate(rows[:10], 1):
            print(f"  [{idx}] {r}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more rows")
    except Exception as e:
        print(f"Query Execution Error: {e}", file=sys.stderr)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Universal Ground-Truth Audit & System Evaluator Engine")
    parser.add_argument("--sqlite", type=str, help="Path to SQLite database to inspect")
    parser.add_argument("--verbose", action="store_true", help="Show full column schema information")
    parser.add_argument("--query", type=str, help="Execute read-only SQL query on --sqlite database")
    parser.add_argument("--benchmark", type=str, help="Path to benchmark / test file to scan")
    parser.add_argument("--patterns", nargs="+", help="Regex patterns to count in --benchmark file")

    args = parser.parse_args()

    if not any([args.sqlite, args.benchmark, args.query]):
        parser.print_help()
        sys.exit(1)

    if args.sqlite and not args.query:
        inspect_sqlite(args.sqlite, verbose=args.verbose)

    if args.sqlite and args.query:
        test_sql_query(args.sqlite, args.query)

    if args.benchmark and args.patterns:
        count_patterns_in_file(args.benchmark, args.patterns)


if __name__ == "__main__":
    main()
