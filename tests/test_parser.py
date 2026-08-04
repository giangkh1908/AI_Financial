"""Test etl/parser.py — tách trang, trích bảng, grid (case vàng M1).

Dùng dữ liệu thật HPG 2018 (đã khảo sát) + case tổng hợp.
"""

from pathlib import Path

from vifinqa.etl.parser import Page, extract_tables, parse_table_grid, split_pages

ROOT = Path(__file__).resolve().parents[1]
HPG_TXT = (
    ROOT
    / "data/financial_statements/HPG/2018/HPG_financial_statements_2018_consolidated"
    / "HPG_financial_statements_2018_consolidated_extracted.txt"
)


def _hpg_text() -> str:
    return HPG_TXT.read_text(encoding="utf-8", errors="replace")


# --- split_pages ---


def test_split_pages_counts():
    pages = split_pages(_hpg_text())
    assert len(pages) >= 50
    assert pages[0].page_no == 1
    # page_no tăng dần
    assert [p.page_no for p in pages] == sorted(p.page_no for p in pages)


def test_split_pages_has_tables_on_statement_pages():
    pages = split_pages(_hpg_text())
    page_with_bs = next(p for p in pages if "cân đối" in p.text.lower() and p.tables)
    assert page_with_bs.tables, "trang CĐKT phải có ≥1 bảng"


# --- extract_tables ---


def test_extract_tables_single_line():
    html = 'text <table><tr><td>a</td></tr></table> more <table><tr><td>b</td></tr></table>'
    tables = extract_tables(html)
    assert len(tables) == 2
    assert tables[0].startswith("<table>")


def test_extract_tables_real_balance_sheet():
    # Tìm bảng CĐKT đầu tiên trong HPG (header "31/12/2018 VND")
    pages = split_pages(_hpg_text())
    found = None
    for p in pages:
        for t in p.tables:
            if "31/12/2018 VND" in t and "1/1/2018 VND" in t and "Mã số" in t:
                found = t
                break
        if found:
            break
    assert found, "không tìm thấy bảng CĐKT HPG"
    grid = parse_table_grid(found)
    assert grid.n_cols == 5
    assert grid.n_rows > 20
    # Dòng TÀI SẢN là section title colspan=5 → đủ 5 cột
    assert any(r == ["TÀI SẢN"] * 5 for r in grid.rows)


# --- parse_table_grid ---


def test_grid_colspan_expansion():
    html = '<table><tr><td colspan="3">NGUỒN VỐN</td></tr><tr><td>a</td><td>b</td><td>c</td></tr></table>'
    grid = parse_table_grid(html)
    assert grid.n_cols == 3
    assert grid.rows[0] == ["NGUỒN VỐN"] * 3


def test_grid_unescape_html_entities():
    # &#x27; = dấu nháy đơn — BeautifulSoup tự unescape
    html = '<table><tr><td>LU&#x27;U CHUYÊN TIÊN</td></tr></table>'
    grid = parse_table_grid(html)
    assert grid.rows[0][0] == "LU'U CHUYÊN TIÊN"


def test_grid_multi_cell_row():
    html = '<table><tr><td>Lợi nhuận trước thuế</td><td>01</td><td></td><td>10.071.072.872.302</td><td>9.288.369.742.697</td></tr></table>'
    grid = parse_table_grid(html)
    assert grid.rows[0] == [
        "Lợi nhuận trước thuế",
        "01",
        "",
        "10.071.072.872.302",
        "9.288.369.742.697",
    ]


def test_grid_empty_table():
    grid = parse_table_grid("<table></table>")
    assert grid.n_rows == 0
    assert grid.n_cols == 0
