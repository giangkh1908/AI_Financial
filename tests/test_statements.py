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


def test_notes_with_sequence_column_no_year_not_statement():
    # Notes "Số năm khấu hao" (50/25/40) KHÔNG có cột năm — trước đây fallback cột mã
    # (^\d{1,2}$) nhận nhầm thành BCTC
    from vifinqa.etl.parser import parse_table_grid

    grid = parse_table_grid(
        "<table>"
        "<tr><td>Số năm</td></tr>"
        "<tr><td>50</td></tr><tr><td>25</td></tr><tr><td>40</td></tr>"
        "</table>"
    )
    anchor = "được ghi nhận vào Báo cáo kết quả hoạt động kinh doanh"
    assert classify_statement(grid, anchor) is None


def test_statement_with_period_label_header_classified():
    # BCTC dùng "Số cuối năm"/"Số đầu năm" thay ngày tháng (DCM 2025) — không được bỏ sót
    from vifinqa.etl.parser import parse_table_grid

    grid = parse_table_grid(
        "<table>"
        "<tr><td></td><td>TÀI SẢN</td><td>Mã số</td><td>Thuyết minh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>"
        "<tr><td>Tiền và các khoản tương đương tiền</td><td></td><td>110</td><td></td><td>1.000</td><td>900</td></tr>"
        "<tr><td>Hàng tồn kho</td><td></td><td>140</td><td></td><td>2.000</td><td>1.500</td></tr>"
        "</table>"
    )
    anchor = "Bảng cân đối kế toán riêng tại ngày 31 tháng 12 năm 2025"
    assert classify_statement(grid, anchor) == "balance_sheet"


def test_english_statement_classified():
    # BCTC tiếng Anh (FPT/DBC/VGC 2024-2025): "Balance Sheet" + "Closing/Opening balance"
    from vifinqa.etl.parser import parse_table_grid

    grid = parse_table_grid(
        "<table>"
        "<tr><td></td><td>ASSETS</td><td>Codes</td><td>Notes</td><td>Closing balance</td><td>Opening balance</td></tr>"
        "<tr><td>A.</td><td>CURRENT ASSETS</td><td>100</td><td></td><td>8,198,590,237,083</td><td>4,283,157,223,963</td></tr>"
        "<tr><td>I.</td><td>Cash and cash equivalents</td><td>110</td><td>4</td><td>2,062,744,834,148</td><td>720,832,090,017</td></tr>"
        "<tr><td>II.</td><td>Short-term financial investments</td><td>120</td><td></td><td>4,534,100,000,000</td><td>2,526,500,000,000</td></tr>"
        "</table>"
    )
    anchor = "BALANCE SHEET\nAs at 31 December 2020\nUnit: VND"
    assert classify_statement(grid, anchor) == "balance_sheet"


def test_statement_with_year_and_garbled_header_still_classified():
    # BCTC thật có cột năm + cột mã (dù header "Mãsố" dính chữ) → vẫn income
    from vifinqa.etl.parser import parse_table_grid

    grid = parse_table_grid(
        "<table>"
        '<tr><td></td><td>Mãsố</td><td>2018 VND</td><td>2017 VND</td></tr>'
        "<tr><td>Doanh thu thuần</td><td>10</td><td>55.836</td><td>46.161</td></tr>"
        "<tr><td>Lợi nhuận gộp</td><td>20</td><td>11.670</td><td>10.625</td></tr>"
        "</table>"
    )
    anchor = "Báo cáo kết quả hoạt động kinh doanh hợp nhất cho năm kết thúc ngày 31 tháng 12 năm 2018"
    assert classify_statement(grid, anchor) == "income"


def test_notes_with_year_column_but_no_code_not_statement():
    # Notes có cột năm nhưng không có mã số ("Lãi tiền gửi") → None
    from vifinqa.etl.parser import parse_table_grid

    grid = parse_table_grid(
        "<table>"
        "<tr><td></td><td>2018VND</td><td>2017VND</td></tr>"
        "<tr><td>Lãi tiền gửi</td><td>208.253</td><td>69.917</td></tr>"
        "<tr><td>Lãi chênh lệch tỷ giá</td><td>85.422</td><td>43.977</td></tr>"
        "</table>"
    )
    anchor = "Thu nhập tài chính"
    assert classify_statement(grid, anchor) is None
