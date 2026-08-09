"""parser.py — tách trang, trích bảng HTML, dựng grid ô từ OCR BCTC.

Khảo sát (HPG/VCB/VJC): `<table>` luôn nằm gọn 1 dòng, không lồng nhau,
HTML-escape có (&#x27;) → dùng BeautifulSoup để unescape chuẩn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from vifinqa.etl.numbers import is_period_cell

# Marker trang: "===== PAGE 1 ====="
PAGE_RE = re.compile(r"=====\s*PAGE\s+(\d+)\s*=====")
# Bảng HTML nằm gọn trong 1 dòng (re.S cho an toàn nếu xuống dòng)
TABLE_RE = re.compile(r"<table>.*?</table>", re.S)


@dataclass
class Page:
    page_no: int
    text: str
    tables: list[str] = field(default_factory=list)


@dataclass
class TableGrid:
    rows: list[list[str]]  # đã expand colspan
    n_cols: int

    @property
    def n_rows(self) -> int:
        return len(self.rows)


def split_pages(text: str) -> list[Page]:
    """Tách toàn văn OCR theo `===== PAGE N =====` → list[Page].

    Phần trước marker PAGE 1 (nếu có) là preamble, bỏ qua.
    """
    parts = PAGE_RE.split(text)
    pages: list[Page] = []
    # parts[0] = preamble; sau đó cặp (page_no, content)
    for i in range(1, len(parts), 2):
        try:
            page_no = int(parts[i].strip())
        except ValueError:
            continue
        content = parts[i + 1]
        pages.append(Page(page_no=page_no, text=content, tables=extract_tables(content)))
    return pages


def extract_tables(page_text: str) -> list[str]:
    """Trích mọi `<table>...</table>` trong một trang (giữ nguyên HTML)."""
    return [m.group(0) for m in TABLE_RE.finditer(page_text)]


def parse_table_grid(table_html: str) -> TableGrid:
    """Parse HTML bảng → grid 2 chiều, expand `colspan` VÀ `rowspan`.

    rowspan: ô `rowspan=N` giữ nguyên text ở N dòng liên tiếp (cùng cột). Fix
    session 8/8: header ngân hàng (`<td rowspan=2 colspan=2></td>`) trước đây
    lệch cột → mọi bảng statement giờ align đúng code/label/period columns.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    rows: list[list[str]] = []
    if table is None:
        return TableGrid(rows=rows, n_cols=0)
    active: dict[int, tuple[str, int]] = {}  # col → (text, remaining_rows)
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        row: list[str] = []
        col = 0
        # điền các ô rowspan còn hiệu lực từ dòng trước
        new_active: dict[int, tuple[str, int]] = {}
        for c, (text, remaining) in sorted(active.items()):
            while col < c:
                row.append("")
                col += 1
            row.append(text)
            col += 1
            if remaining > 1:
                new_active[c] = (text, remaining - 1)
        active = new_active
        # các ô mới của dòng này
        for cell in cells:
            colspan = int(cell.get("colspan") or 1)
            rowspan = int(cell.get("rowspan") or 1)
            text = cell.get_text(" ", strip=True)
            while col in active and col < len(row):
                col += 1
            for i in range(colspan):
                if rowspan > 1:
                    active[col + i] = (text, rowspan - 1)
                row.append(text)
            col += colspan
        rows.append(row)
    n_cols = max((len(r) for r in rows), default=0)
    for r in rows:
        r += [""] * (n_cols - len(r))
    return TableGrid(rows=rows, n_cols=n_cols)


def find_header_row(grid: TableGrid) -> int:
    """Chỉ số dòng header: dòng đầu có cell chứa năm/kỳ (ngày hoặc "Số cuối năm"). Fallback 0."""
    for i, row in enumerate(grid.rows):
        if any(is_period_cell(c) for c in row):
            return i
    return 0
