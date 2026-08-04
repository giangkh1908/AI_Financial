"""Test etl/statements.py — phân loại bảng (case vàng M1 trên dữ liệu thật)."""

from pathlib import Path

from vifinqa.etl.catalog_builder import anchor_text
from vifinqa.etl.parser import parse_table_grid, split_pages
from vifinqa.etl.statements import classify_statement
from vifinqa.loader import iter_reports

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SAMPLE_IDS = {
    "HPG": "HPG_financial_statements_2018_consolidated",
    "VCB": "VCB_financial_statements_2022_consolidated",
    "VJC": "VJC_financial_statements_2018_separate",
}


def _scan(report_id: str) -> dict[str, list[int]]:
    """Trả {statement: [table_idx...]} với table_idx = số thứ tự <table> trong report."""
    report = next(r for r in iter_reports(DATA) if r.report_id == report_id)
    text = report.path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, list[int]] = {}
    table_idx = 0
    for page in split_pages(text):
        pos = 0
        for t in page.tables:
            table_idx += 1
            grid = parse_table_grid(t)
            if grid.n_rows == 0:
                pos = page.text.find(t, pos)
                if pos != -1:
                    pos += len(t)
                continue
            anchor = anchor_text(page.text, t, pos)
            stmt = classify_statement(grid, anchor)
            if stmt:
                result.setdefault(stmt, []).append(table_idx)
            pos = page.text.find(t, pos)
            if pos != -1:
                pos += len(t)
    return result


# --- golden: đúng số bảng & đúng vị trí trên 3 mẫu ---


def test_hpg_statements():
    s = _scan(SAMPLE_IDS["HPG"])
    assert s["balance_sheet"] == [3, 4, 5]
    assert s["income"] == [6, 7]
    assert s["cash_flow"] == [8, 9]


def test_vcb_bank_statements():
    s = _scan(SAMPLE_IDS["VCB"])
    assert s["balance_sheet"] == [7, 8]   # "Báo cáo tình hình tài chính"; off-balance bị loại
    assert s["income"] == [11, 12]        # "Báo cáo kết quả hoạt động" (không "kinh doanh")
    assert s["cash_flow"] == [13, 14]


def test_vjc_statements():
    s = _scan(SAMPLE_IDS["VJC"])
    assert s["balance_sheet"] == [5, 6]
    assert s["income"] == [7]
    assert s["cash_flow"] == [8, 9]


# --- golden: notes KHÔNG bị phân loại nhầm ---


def test_hpg_off_balance_notes_not_statement():
    # "29. Các khoản mục ngoài bảng cân đối kế toán" → không phải BCTC
    s = _scan(SAMPLE_IDS["HPG"])
    assert len(s["balance_sheet"]) == 3  # chỉ 3 fragment CĐKT thật


def test_vcb_off_balance_items_not_statement():
    # "CÁC CHỈ TIÊU NGOÀI BÁO CÁO TÌNH HÌNH TÀI CHÍNH" (cam kết) → không tính là BS
    s = _scan(SAMPLE_IDS["VCB"])
    assert len(s["balance_sheet"]) == 2


def test_vjc_notes_referencing_statement_not_classified():
    # Notes "Ghi nhận trong báo cáo kết quả hoạt động kinh doanh" → income chỉ có bảng thật
    s = _scan(SAMPLE_IDS["VJC"])
    assert s["income"] == [7]


# --- đơn vị case nhỏ ---


def test_classify_negative_filter():
    from vifinqa.etl.parser import parse_table_grid

    grid = parse_table_grid(
        '<table><tr><td></td><td colspan="3">CÁC CHỈ TIÊU NGOÀI BÁO CÁO TÌNH HÌNH TÀI CHÍNH</td></tr></table>'
    )
    anchor = "Báo cáo tình hình tài chính hợp nhất tại ngày 31 tháng 12 năm 2022"
    assert classify_statement(grid, anchor) is None
