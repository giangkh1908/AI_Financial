"""facts_builder.py — Tier A facts: 3 BCTC lõi → long-format CSV chuẩn VND.

Pipeline (M2):
  scan report (parse HTML + classify + detect unit) → list table fragments
  → group_statement_fragments (gom liên tiếp cùng stmt+signature)
  → build_asset + emit_facts (item_code, label, period, value_vnd, src_table_ids)
  → facts/{report_id}_facts.csv ; gộp facts_all.csv

Tự chứa (dùng cùng classify_statement/detect_unit/parse với M1 → facts & catalog
nhất quán). Unit chuẩn về VND: value_vnd = raw_number * unit_factor.
"""

from __future__ import annotations

import csv
from pathlib import Path

from vifinqa.etl.catalog_builder import anchor_text
from vifinqa.etl.format_classify import classify_table
from vifinqa.etl.numbers import detect_unit
from vifinqa.etl.parser import find_header_row, parse_table_grid, split_pages
from vifinqa.etl.statements import (
    build_asset,
    emit_facts,
    group_statement_fragments,
)
from vifinqa.loader import ReportMeta

FACTS_HEADER = [
    "ticker", "year", "report_type", "statement", "item_code",
    "item_label", "item_label_raw", "period_key", "period_label",
    "value_vnd", "src_table_ids",
]


def _scan_report_tables(report: ReportMeta) -> list[tuple]:
    """Quét report theo thứ tự → list (table_idx, table_id, grid, anchor, stmt, uf, ul, layout).

    table_idx 1-based toàn report (không reset theo trang) — khớp với wide CSV
    table_N.csv của M1. `layout` = TableLayout chuẩn hoá (format_classify) —
    FORMAT CHUNG cho mọi loại DN/bank/chứng khoán/header 2 tầng.
    """
    text = report.path.read_text(encoding="utf-8", errors="replace")
    pages = split_pages(text)
    out: list[tuple] = []
    table_idx = 0
    for page in pages:
        pos = 0
        for table_html in page.tables:
            table_idx += 1
            grid = parse_table_grid(table_html)
            table_id = f"table_{table_idx}"
            if grid.n_rows == 0:
                out.append((table_idx, table_id, grid, "", None, 1.0, "VND", None))
                pos = page.text.find(table_html, pos)
                if pos != -1:
                    pos += len(table_html)
                continue
            anchor = anchor_text(page.text, table_html, pos)
            stmt, layout = classify_table(grid, anchor, report.report_type)
            header_idx = layout.header_idx if layout else find_header_row(grid)
            header_cells = grid.rows[header_idx] if header_idx < grid.n_rows else []
            uf, ul = detect_unit(header_cells, anchor)
            out.append((table_idx, table_id, grid, anchor, stmt, uf, ul, layout))
            pos = page.text.find(table_html, pos)
            if pos != -1:
                pos += len(table_html)
    return out


def build_report_facts(report: ReportMeta) -> list[dict]:
    """Xử lý 1 report → list fact dict (chưa ghi file)."""
    tables = _scan_report_tables(report)
    groups = group_statement_fragments(tables)
    facts: list[dict] = []
    for stmt, frags, uf, ul in groups:
        asset = build_asset(stmt, frags, uf, ul)
        facts.extend(emit_facts(asset, report.ticker, report.year, report.report_type))
    return facts


def write_facts_csv(facts: list[dict], path: Path) -> None:
    """Ghi facts CSV (UTF-8). Ô rỗng → xâu rỗng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FACTS_HEADER)
        for fact in facts:
            w.writerow([fact.get(c, "") for c in FACTS_HEADER])


def write_facts_part(facts: list[dict], path: Path) -> int:
    """Ghi 1 phần facts (theo ticker) → trả số dòng (LOGICAL)."""
    write_facts_csv(facts, path)
    return len(facts)


def merge_facts_parts(parts_dir: Path, out_path: Path) -> int:
    """Gộp các phần facts theo ticker → facts_all.csv. Trả số dòng LOGICAL."""
    parts = sorted(parts_dir.glob("facts_*.csv"))
    n_rows = 0
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(FACTS_HEADER)
        for i, part in enumerate(parts):
            with open(part, encoding="utf-8", newline="") as pf:
                for j, row in enumerate(csv.reader(pf)):
                    if i > 0 and j == 0:
                        continue  # bỏ header phần sau
                    if i == 0 and j == 0:
                        continue  # bỏ header phần đầu (đã ghi)
                    w.writerow(row)
                    n_rows += 1
    return n_rows