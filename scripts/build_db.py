"""
ETL Pipeline: Financial Statements (TXT OCR) -> SQLite Grounded Database
Ponytail Standard: Zero external dependencies (stdlib only: sqlite3, re, unicodedata, csv).
Extracts:
- 4 Primary Statements: Balance Sheet, Income Statement, Cash Flow (Direct/Indirect)
- Financial Notes (Thuyết minh BCTC) with contextual section headers
- Full provenance: page_no, source_doc, raw_value, unit, period_label
"""

import os
import re
import csv
import sys
import time
import sqlite3
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(r"D:\GURU\data")
STATEMENTS_DIR = DATA_DIR / "financial_statements"
CODE_STOCK_FILE = DATA_DIR / "code_stock.csv"
DB_PATH = DATA_DIR / "financial.db"

# ----------------------------------------------------------------------
# 1. Normalization & Utility Functions
# ----------------------------------------------------------------------

def remove_accents(s: str) -> str:
    """Converts Vietnamese text to unaccented lowercase ASCII."""
    if not s:
        return ""
    s = s.replace("đ", "d").replace("Đ", "d")
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_bytes = nfkd.encode("ASCII", "ignore")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_bytes.decode("utf-8").lower())
    return " ".join(cleaned.split())


def parse_number(s: str):
    """
    Parses Vietnamese accounting numbers into float.
    Handles:
      - Parentheses for negative: '(208.253.201.298)' -> -208253201298.0
      - Standard VN dot separation: '3.989.369.447.153' -> 3989369447153.0
      - Comma decimals: '1.234,56' -> 1234.56
      - Dashes / empty: '-' -> 0.0
    """
    if not s:
        return None
    s = s.strip()
    if s in ("-", "–", "—", "", "None", "null", "N/A"):
        return 0.0

    is_neg = False
    if s.startswith("(") and s.endswith(")"):
        is_neg = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        is_neg = True
        s = s[1:].strip()

    # Determine thousand vs decimal separators
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s and "," not in s:
        parts = s.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "")
        elif len(parts) == 2 and len(parts[1]) in (1, 2):
            pass
        else:
            s = s.replace(".", "")
    elif "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = s.replace(",", ".")
        elif len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(",", "")

    try:
        val = float(s)
        return -val if is_neg else val
    except ValueError:
        return None


def detect_unit_and_factor(text: str):
    """Detects unit string and numerical multiplier from text."""
    text_lower = text.lower()
    if "triệu đồng" in text_lower or "trieu dong" in text_lower or "triệu vnd" in text_lower or "trieu vnd" in text_lower:
        return "Triệu VND", 1_000_000.0
    elif "tỷ đồng" in text_lower or "ty dong" in text_lower or "tỷ vnd" in text_lower:
        return "Tỷ VND", 1_000_000_000.0
    elif "nghìn đồng" in text_lower or "nghin dong" in text_lower or "nghìn vnd" in text_lower:
        return "Nghìn VND", 1_000.0
    elif "usd" in text_lower:
        return "USD", 1.0
    elif "vnd" in text_lower or "đồng" in text_lower:
        return "VND", 1.0
    return "VND", 1.0


def classify_statement(title_text: str) -> str:
    """Classifies statement type from section or page title."""
    t = title_text.upper()
    if "CÂN ĐỐI KẾ TOÁN" in t or "TÌNH HÌNH TÀI CHÍNH" in t:
        return "balance_sheet"
    elif "KẾT QUẢ HOẠT ĐỘNG" in t or "KẾT QUẢ KINH DOANH" in t or "THU NHẬP TOÀN DIỆN" in t:
        return "income_statement"
    elif "LƯU CHUYỂN TIỀN" in t:
        return "cash_flow"
    elif "THUYẾT MINH" in t:
        return "notes"
    return "notes"


# ----------------------------------------------------------------------
# 2. Table Extraction & Parsing
# ----------------------------------------------------------------------

RE_PAGE = re.compile(r"=====\s*PAGE\s*(\d+)\s*=====", re.IGNORECASE)
RE_TABLE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
RE_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
RE_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
RE_TAG = re.compile(r"<[^>]+>")


def clean_cell(html: str) -> str:
    return RE_TAG.sub("", html).strip()


def parse_file(file_path: str, company_map: dict):
    """
    Parses a single financial statement text file into clean fact records.
    Returns: list of tuples ready for SQLite insert.
    """
    path = Path(file_path)
    file_name = path.name
    doc_id = path.stem.replace("_extracted", "")

    # Extract metadata from path: .../<TICKER>/<YEAR>/<DOC_ID>/<DOC_ID>_extracted.txt
    parts = doc_id.split("_financial_statements_")
    if len(parts) == 2:
        ticker = parts[0]
        rest = parts[1].split("_")
        year = int(rest[0]) if rest[0].isdigit() else 2020
        report_type = rest[1] if len(rest) > 1 else "consolidated"
    else:
        ticker = path.parent.parent.parent.name
        year = int(path.parent.parent.name) if path.parent.parent.name.isdigit() else 2020
        report_type = "consolidated" if "consolidated" in doc_id else "separate"

    company_name = company_map.get(ticker, ticker)

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    pages_raw = RE_PAGE.split(text)
    if len(pages_raw) < 2:
        return []

    facts = []

    # Iterate over pages: pages_raw[1::2] = page_nos, pages_raw[2::2] = page_contents
    for i in range(1, len(pages_raw), 2):
        page_no = int(pages_raw[i])
        page_content = pages_raw[i + 1]

        # Extract page-level unit if present
        page_unit, page_factor = detect_unit_and_factor(page_content[:1500])

        # Find tables on this page
        table_matches = RE_TABLE.findall(page_content)
        if not table_matches:
            continue

        # Detect section title preceding tables
        text_before_tables = RE_TABLE.split(page_content)[0]
        header_lines = [line.strip() for line in text_before_tables.split("\n") if line.strip()]
        page_title = " // ".join(header_lines[-3:]) if header_lines else ""
        statement_type = classify_statement(page_title)

        for tbl_idx, tbl_html in enumerate(table_matches):
            tr_matches = RE_TR.findall(tbl_html)
            if not tr_matches:
                continue

            # Parse all rows in table
            parsed_rows = []
            for tr in tr_matches:
                tds = [clean_cell(c) for c in RE_TD.findall(tr)]
                if any(tds):  # non-empty row
                    parsed_rows.append(tds)

            if len(parsed_rows) < 2:
                continue

            # Heuristic Header Analysis
            # Look at first 1-3 rows to find headers and period columns
            header_row = parsed_rows[0]
            col_periods = {}  # col_idx -> (period_label, unit_factor)

            # Check if headers span row 0 or 1
            header_candidates = [header_row]
            if len(parsed_rows) > 1 and not any(parse_number(c) is not None for c in parsed_rows[1]):
                header_candidates.append(parsed_rows[1])

            data_start_idx = len(header_candidates)

            # Map column indices to period & unit
            for r in header_candidates:
                for col_idx, cell in enumerate(r):
                    if not cell:
                        continue
                    cell_clean = cell.replace("\n", " ").strip()
                    # Check if cell describes a period (e.g. '31.12.2022', 'Số cuối năm', 'Năm 2022', '2018VND')
                    cell_u, cell_factor = detect_unit_and_factor(cell_clean)
                    factor = cell_factor if cell_factor != 1.0 else page_factor

                    if any(k in cell_clean.lower() for k in ["cuối năm", "dau nam", "đầu năm", "năm nay", "năm trước", "201", "202", "31/12", "31.12", "01/01"]):
                        col_periods[col_idx] = (cell_clean, factor)

            # If no period columns detected in header, fallback to rightmost numeric columns
            if not col_periods:
                # Find columns that have numbers in data rows
                num_cols = []
                for col_idx in range(len(header_row)):
                    count_nums = sum(1 for dr in parsed_rows[data_start_idx:data_start_idx + 8] if col_idx < len(dr) and parse_number(dr[col_idx]) is not None)
                    if count_nums >= 2:
                        num_cols.append(col_idx)
                for idx, col_idx in enumerate(num_cols):
                    p_label = f"period_col_{idx + 1}"
                    col_periods[col_idx] = (p_label, page_factor)

            if not col_periods:
                continue

            # Process Data Rows
            for row in parsed_rows[data_start_idx:]:
                if not row or len(row) < 2:
                    continue

                # Identify text column vs code column
                item_name = ""
                item_code = ""

                # Text is usually the first non-numeric string column
                for c_idx, cell in enumerate(row):
                    if c_idx in col_periods:
                        continue
                    cell_s = cell.strip()
                    if not cell_s:
                        continue
                    # Check if code (short digits, e.g. '100', '05', 'I', 'II', '1.1')
                    if re.match(r"^(\d{1,4}[a-z]?|[IVXLCDM]+|\d+\.\d+)$", cell_s) and not item_code and len(cell_s) <= 5:
                        item_code = cell_s
                    elif len(cell_s) > len(item_name) and parse_number(cell_s) is None:
                        item_name = cell_s

                if not item_name or len(item_name) < 2:
                    continue

                # Filter out pure headers masquerading as rows
                if item_name.upper() in ("TÀI SẢN", "NGUỒN VỐN", "CHỈ TIÊU", "MÃ SỐ", "THUYẾT MINH"):
                    continue

                item_ascii = remove_accents(item_name)

                # Extract values for each period column
                for col_idx, (p_label, factor) in col_periods.items():
                    if col_idx >= len(row):
                        continue
                    raw_val = row[col_idx].strip()
                    num_val = parse_number(raw_val)
                    if num_val is None:
                        continue

                    # Grounded value in VND
                    val_vnd = num_val * factor

                    facts.append((
                        ticker,
                        company_name,
                        year,
                        report_type,
                        statement_type,
                        page_title[:120],
                        item_code,
                        item_name,
                        item_ascii,
                        p_label[:40],
                        val_vnd,
                        raw_val,
                        page_unit,
                        page_no,
                        doc_id
                    ))

    return facts


# ----------------------------------------------------------------------
# 3. Database Initialization & Pipeline Runner
# ----------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

DROP TABLE IF EXISTS financial_facts;
CREATE TABLE financial_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    company_name TEXT,
    year INTEGER NOT NULL,
    report_type TEXT NOT NULL,
    statement TEXT NOT NULL,
    section_title TEXT,
    item_code TEXT,
    item_name TEXT NOT NULL,
    item_name_ascii TEXT NOT NULL,
    period_label TEXT,
    value_vnd REAL,
    raw_value TEXT,
    unit TEXT,
    page_no INTEGER,
    source_doc TEXT NOT NULL
);

CREATE INDEX idx_facts_core ON financial_facts (ticker, year, report_type);
CREATE INDEX idx_facts_ascii ON financial_facts (item_name_ascii);
CREATE INDEX idx_facts_stmt ON financial_facts (statement);

-- Native Full-Text Search (FTS5) for flexible fuzzy search on Vietnamese financial indicators
DROP TABLE IF EXISTS facts_fts;
CREATE VIRTUAL TABLE facts_fts USING fts5(
    item_name,
    item_name_ascii,
    section_title,
    content='financial_facts',
    content_rowid='id'
);
"""

TRIGGER_SQL = """
CREATE TRIGGER facts_ai AFTER INSERT ON financial_facts BEGIN
  INSERT INTO facts_fts(rowid, item_name, item_name_ascii, section_title)
  VALUES (new.id, new.item_name, new.item_name_ascii, new.section_title);
END;
"""

INSERT_FACT_SQL = """
INSERT INTO financial_facts (
    ticker, company_name, year, report_type, statement,
    section_title, item_code, item_name, item_name_ascii,
    period_label, value_vnd, raw_value, unit, page_no, source_doc
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def load_company_map() -> dict:
    company_map = {}
    if CODE_STOCK_FILE.exists():
        with open(CODE_STOCK_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    company_map[row[0].strip()] = row[1].strip()
    return company_map


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    cur.executescript(TRIGGER_SQL)
    conn.commit()
    conn.close()


def verify_database():
    """Independent verification check on real test questions."""
    print("\n" + "=" * 50)
    print("RUNNING INDEPENDENT VERIFICATION ON SQLITE FACTS")
    print("=" * 50)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Count total facts
    cur.execute("SELECT COUNT(*) FROM financial_facts")
    total_facts = cur.fetchone()[0]
    print(f"[Check 0] Total rows in financial_facts: {total_facts:,}")
    assert total_facts > 0, "Database is empty!"

    # Test Question 1: VJC 2018 separate lãi tiền gửi
    print("\n[Check 1] Verify Question 1 (VJC 2018 separate lãi tiền gửi):")
    cur.execute("""
        SELECT item_name, period_label, raw_value, value_vnd, page_no, source_doc
        FROM financial_facts
        WHERE ticker = 'VJC' AND year = 2018 AND report_type = 'separate'
          AND item_name_ascii LIKE '%lai tien gui%'
        ORDER BY page_no ASC
        LIMIT 5
    """)
    rows = cur.fetchall()
    for r in rows:
        print(f"  -> {r[0]} | Kỳ: {r[1]} | Raw: {r[2]} | VND: {r[3]:,.0f} | Trang {r[4]} ({r[5]})")
    assert any("208" in str(r[2]) for r in rows), "Did not find expected 208 tỷ row for VJC 2018!"
    print("  => CHECK 1 PASSED!")

    # Test Question 2: ACB 2022 cho vay khách hàng ngành Thương mại
    print("\n[Check 2] Verify Question 2 (ACB 2022 cho vay ngành Thương mại):")
    cur.execute("""
        SELECT item_name, period_label, raw_value, value_vnd, page_no, source_doc
        FROM financial_facts
        WHERE ticker = 'ACB' AND year = 2022
          AND item_name_ascii = 'thuong mai'
        ORDER BY page_no ASC
        LIMIT 5
    """)
    acb_rows = cur.fetchall()
    for r in acb_rows:
        print(f"  -> {r[0]} | Kỳ: {r[1]} | Raw: {r[2]} | VND: {r[3]:,.0f} | Trang {r[4]} ({r[5]})")
    assert len(acb_rows) > 0, "Did not find ACB Thương mại facts!"
    print("  => CHECK 2 PASSED!")

    conn.close()
    print("\nALL SYSTEM VERIFICATION CHECKS PASSED PERFECTLY!\n")


def build_database(max_files=None):
    start_time = time.time()
    company_map = load_company_map()
    init_db()

    all_files = sorted(list(STATEMENTS_DIR.rglob("*_extracted.txt")))
    if max_files:
        all_files = all_files[:max_files]

    total_files = len(all_files)
    print(f"Starting ETL on {total_files} files -> SQLite: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    batch = []
    total_inserted = 0

    for idx, f in enumerate(all_files, 1):
        facts = parse_file(str(f), company_map)
        if facts:
            batch.extend(facts)

        if len(batch) >= 10000:
            cur.executemany(INSERT_FACT_SQL, batch)
            conn.commit()
            total_inserted += len(batch)
            batch = []

        if idx % 200 == 0 or idx == total_files:
            elapsed = time.time() - start_time
            print(f"  [{idx}/{total_files}] files processed | {total_inserted + len(batch):,} facts parsed | {elapsed:.1f}s")

    if batch:
        cur.executemany(INSERT_FACT_SQL, batch)
        conn.commit()
        total_inserted += len(batch)

    conn.close()
    elapsed = time.time() - start_time
    print(f"\nETL COMPLETED in {elapsed:.1f}s! Total facts inserted: {total_inserted:,}")
    verify_database()


if __name__ == "__main__":
    build_database()
