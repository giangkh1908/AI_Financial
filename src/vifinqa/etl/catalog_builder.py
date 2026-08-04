"""catalog_builder.py — mọi bảng → wide CSV + catalog_tables.csv + documents.csv.

- Wide raw tier: `derived/tables/{report_id}/table_{N}.csv` (grid giữ chuỗi OCR, KHÔNG parse số).
- `catalog_tables.csv`: metadata mọi bảng (dùng cho retrieval).
- `documents.csv`: metadata mỗi báo cáo.
N = số thứ tự <table> trong report (1-based, toàn report không reset theo trang).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from vifinqa.etl.numbers import detect_unit
from vifinqa.etl.parser import TableGrid, find_header_row, parse_table_grid, split_pages
from vifinqa.etl.statements import classify_statement
from vifinqa.loader import ReportMeta

ANCHOR_LINES = 6  # số dòng text trước bảng dùng làm context phân loại

CATALOG_HEADER = [
    "report_id", "ticker", "year", "report_type", "table_id", "page_no",
    "unit", "is_statement", "statement", "header_text", "row_labels",
    "n_rows", "n_cols", "anchor_context",
]
DOC_HEADER = [
    "report_id", "ticker", "year", "report_type", "company_name",
    "has_consolidated", "has_separate",
]


@dataclass
class CatalogRow:
    report_id: str
    ticker: str
    year: int
    report_type: str
    table_id: str
    page_no: int | None
    unit: str
    is_statement: bool
    statement: str
    header_text: str
    row_labels: str
    n_rows: int
    n_cols: int
    anchor_context: str

    def as_row(self) -> list[str]:
        return [
            self.report_id, self.ticker, str(self.year), self.report_type,
            self.table_id, "" if self.page_no is None else str(self.page_no),
            self.unit, "1" if self.is_statement else "0", self.statement,
            self.header_text, self.row_labels,
            str(self.n_rows), str(self.n_cols), self.anchor_context,
        ]


def anchor_text(page_text: str, table_html: str, start_pos: int) -> str:
    """Text vài dòng trước bảng trên cùng trang (context phân loại)."""
    idx = page_text.find(table_html, start_pos)
    if idx == -1:
        idx = page_text.find(table_html)
    before = page_text[:idx]
    lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
    return "\n".join(lines[-ANCHOR_LINES:])


def write_table_csv(tables_dir: Path, table_id: str, grid: TableGrid) -> None:
    """Ghi wide CSV — giữ chuỗi OCR gốc, mọi cột đọc lại dtype=str."""
    path = tables_dir / f"{table_id}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in grid.rows:
            padded = row + [""] * (grid.n_cols - len(row))
            w.writerow(padded)


def header_and_labels(grid: TableGrid, header_idx: int) -> tuple[str, str]:
    """(header_text, row_labels): header nối dọc theo cột; labels = cột đầu ~10 dòng dữ liệu."""
    header = grid.rows[header_idx] if header_idx < grid.n_rows else []
    header_text = " | ".join(c for c in header)
    labels = []
    for row in grid.rows[header_idx + 1:]:
        if not row or not row[0]:
            continue
        first = row[0]
        # bỏ dòng section title (colspan — thường là dòng có số cell = n_cols và text in hoa)
        if len(row) >= grid.n_cols and len(labels) >= 1:
            pass
        labels.append(first)
        if len(labels) >= 10:
            break
    return header_text, " | ".join(labels)


def process_report(report: ReportMeta, derived_dir: Path) -> list[CatalogRow]:
    """Xử lý 1 report → ghi wide CSVs, trả catalog rows."""
    text = report.path.read_text(encoding="utf-8", errors="replace")
    pages = split_pages(text)
    rows: list[CatalogRow] = []
    tables_dir = derived_dir / "tables" / report.report_id
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_idx = 0  # 1-based, toàn report
    for page in pages:
        pos = 0
        for table_html in page.tables:
            table_idx += 1
            grid = parse_table_grid(table_html)
            if grid.n_rows == 0:
                pos = page.text.find(table_html, pos)
                if pos != -1:
                    pos += len(table_html)
                continue
            anchor = anchor_text(page.text, table_html, pos)
            stmt = classify_statement(grid, anchor)
            header_idx = find_header_row(grid)
            header_cells = grid.rows[header_idx] if header_idx < grid.n_rows else []
            unit_label = detect_unit(header_cells, page.text)[1]
            header_text, row_labels = header_and_labels(grid, header_idx)
            table_id = f"table_{table_idx}"
            write_table_csv(tables_dir, table_id, grid)
            rows.append(
                CatalogRow(
                    report_id=report.report_id,
                    ticker=report.ticker,
                    year=report.year,
                    report_type=report.report_type,
                    table_id=table_id,
                    page_no=page.page_no,
                    unit=unit_label,
                    is_statement=stmt is not None,
                    statement=stmt or "",
                    header_text=header_text,
                    row_labels=row_labels,
                    n_rows=grid.n_rows,
                    n_cols=grid.n_cols,
                    anchor_context=anchor,
                )
            )
            pos = page.text.find(table_html, pos)
            if pos != -1:
                pos += len(table_html)
    return rows


def write_catalog_csv(rows: list[CatalogRow], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CATALOG_HEADER)
        for r in rows:
            w.writerow(r.as_row())


def write_documents_csv(reports: list[ReportMeta], stocks: dict[str, str], path: Path) -> None:
    """documents.csv — 1 dòng/report kèm cờ có consolidated/separate theo ticker."""
    ticker_docs: dict[str, dict] = {}
    for r in reports:
        info = ticker_docs.setdefault(
            r.ticker,
            {"has_consolidated": False, "has_separate": False},
        )
        if r.report_type == "consolidated":
            info["has_consolidated"] = True
        elif r.report_type == "separate":
            info["has_separate"] = True
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(DOC_HEADER)
        for r in reports:
            info = ticker_docs[r.ticker]
            w.writerow([
                r.report_id, r.ticker, str(r.year), r.report_type,
                stocks.get(r.ticker, ""),
                "1" if info["has_consolidated"] else "0",
                "1" if info["has_separate"] else "0",
            ])


def build_catalog(reports: list[ReportMeta], stocks: dict[str, str], derived_dir: Path) -> list[CatalogRow]:
    """Chạy toàn bộ reports → catalog_tables.csv + documents.csv. Trả catalog rows."""
    all_rows: list[CatalogRow] = []
    for report in reports:
        all_rows.extend(process_report(report, derived_dir))
    write_catalog_csv(all_rows, derived_dir / "catalog_tables.csv")
    write_documents_csv(reports, stocks, derived_dir / "documents.csv")
    return all_rows
