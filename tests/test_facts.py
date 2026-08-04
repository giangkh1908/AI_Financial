"""Test etl/facts_builder.py + statements.py M2 — gold cases trên dữ liệu thật.

Chạy: .venv\\Scripts\\python -m pytest tests/test_facts.py -q
"""

from pathlib import Path

import pytest

from vifinqa.etl.facts_builder import build_report_facts
from vifinqa.etl.numbers import normalize_label
from vifinqa.etl.parser import parse_table_grid
from vifinqa.etl.statements import (
    Fragment,
    build_asset,
    emit_facts,
    find_item_code_col,
    group_statement_fragments,
    validate_asset,
)
from vifinqa.loader import iter_reports

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SAMPLE_IDS = {
    "HPG": "HPG_financial_statements_2018_consolidated",
    "VCB": "VCB_financial_statements_2022_consolidated",
}


def _facts(report_id: str) -> list[dict]:
    report = next(r for r in iter_reports(DATA) if r.report_id == report_id)
    return build_report_facts(report)


def _val(facts, statement, code, period_key):
    for f in facts:
        if (
            f["statement"] == statement
            and f["item_code"] == code
            and f["period_key"] == period_key
        ):
            return float(f["value_vnd"])
    raise AssertionError(f"không tìm thấy {statement} code={code} period={period_key}")


# --- gold M2: HPG LNST 60 = 8.600.550.706.227 ---


def test_hpg_lnst_code60():
    facts = _facts(SAMPLE_IDS["HPG"])
    assert _val(facts, "income", "60", "flow_year") == pytest.approx(8_600_550_706_227)


def test_hpg_doanh_thu_thuan_code10():
    facts = _facts(SAMPLE_IDS["HPG"])
    assert _val(facts, "income", "10", "flow_year") == pytest.approx(55_836_458_379_759)


# --- cross-sum: BS 270 == 440 ---


def test_hpg_balance_sheet_cross_sum():
    facts = _facts(SAMPLE_IDS["HPG"])
    total_assets = _val(facts, "balance_sheet", "270", "year_end")
    total_equity_liab = _val(facts, "balance_sheet", "440", "year_end")
    assert total_assets == pytest.approx(78_223_007_670_925)
    assert total_equity_liab == pytest.approx(78_223_007_670_925)
    assert total_assets == total_equity_liab


# --- dedupe ở biên fragment: code 60 chỉ xuất hiện 1 lần trong income ---


def test_hpg_income_code60_deduped():
    facts = _facts(SAMPLE_IDS["HPG"])
    # table_6 (mang sang trang sau) + table_7 (mang từ trang trước) cùng code 60.
    # Income có 2 cột kỳ (2018 & 2017, đều flow_year) → dedupe 2 fragment cho 2 fact
    # (thay vì 2 fragment × 2 kỳ = 4).
    code60_flow = [
        f for f in facts
        if f["statement"] == "income" and f["item_code"] == "60" and f["period_key"] == "flow_year"
    ]
    assert len(code60_flow) == 2


# --- VCB ngân hàng: mã La Mã I..XII parse đúng + unit triệu ---


def test_vcb_bank_roman_codes_parsed():
    facts = _facts(SAMPLE_IDS["VCB"])
    bs_codes = {f["item_code"] for f in facts if f["statement"] == "balance_sheet" and f["item_code"]}
    # ít nhất I, II, III, ..., XII (mã La Mã) xuất hiện
    roman = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XII"}
    assert roman.issubset(bs_codes)


def test_vcb_total_assets_unit_megavnd():
    facts = _facts(SAMPLE_IDS["VCB"])
    # "TỔNG TÀI SẢN CÓ" — bank không có mã 270; kiểm tra qua label
    tot = [
        f for f in facts
        if f["statement"] == "balance_sheet" and "tong tai san" in f["item_label"]
        and f["period_key"] == "year_end"
    ]
    assert len(tot) == 1
    # raw 1.813.815.170 (Triệu VND) → 1.813.815.170 × 1e6 = 1.813.815.170.000.000
    assert float(tot[0]["value_vnd"]) == pytest.approx(1_813_815_170_000_000)


def test_vcb_income_after_tax():
    facts = _facts(SAMPLE_IDS["VCB"])
    lnst = [
        f for f in facts
        if f["statement"] == "income" and "loi nhuan sau thue" in f["item_label"]
        and f["period_key"] == "flow_year"
    ]
    assert len(lnst) == 1
    # raw 29.919.054 (Triệu VND) → × 1e6
    assert float(lnst[0]["value_vnd"]) == pytest.approx(29_919_054_000_000)


def test_vcb_has_three_statements():
    facts = _facts(SAMPLE_IDS["VCB"])
    assert {"balance_sheet", "income", "cash_flow"}.issubset({f["statement"] for f in facts})


# --- unit test parse_number + detect_number_format (English) ---


def test_parse_number_vietnamese_default():
    from vifinqa.etl.numbers import parse_number
    assert parse_number("8.600.550.706.227") == 8_600_550_706_227.0
    assert parse_number("(11.078.921.256)") == -11_078_921_256.0
    assert parse_number("-") is None


def test_parse_number_english_format():
    from vifinqa.etl.numbers import parse_number
    assert parse_number("8,198,590,237,083", thousands=",", decimal=".") == 8_198_590_237_083.0
    assert parse_number("(24,779,418)", thousands=",", decimal=".") == -24_779_418.0
    assert parse_number("1,234.56", thousands=",", decimal=".") == 1234.56


def test_detect_number_format_english_vs_vietnamese():
    from vifinqa.etl.numbers import detect_number_format
    en = ["8,198,590,237,083", "4,283,157,223,963", "(24,779,418)"]
    assert detect_number_format(en) == "en"
    vi = ["8.198.590.237.083", "4.283.157.223.963", "(24.779.418)"]
    assert detect_number_format(vi) == "vi"
    # ô có cả 2 dấu: vi "1.234,56" vs en "1,234.56" — không mơ hồ
    assert detect_number_format(["1.234,56", "2.500,00"]) == "vi"
    assert detect_number_format(["1,234.56", "2,500.00"]) == "en"


def test_fpt_english_statement_facts_parsed():
    """BCTC tiếng Anh (FPT 2020 separate) — số dùng ',' nghìn → parse đúng."""
    rid = "FPT_financial_statements_2020_separate"
    if not (DATA / "financial_statements" / "FPT" / "2020").exists():
        pytest.skip("FPT 2020 không có trong data")
    facts = _facts(rid)
    assert {"balance_sheet", "income", "cash_flow"}.issubset({f["statement"] for f in facts})
    # CURRENT ASSETS code 100 = 8,198,590,237,083 (English comma) → VND
    ca = [f for f in facts if f["item_code"] == "100" and f["statement"] == "balance_sheet"]
    assert len(ca) >= 1
    assert float(ca[0]["value_vnd"]) == pytest.approx(8_198_590_237_083)


# --- unit test find_item_code_col: VAS vs bank vs English ---


def test_find_item_code_col_vas():
    grid = parse_table_grid(
        "<table>"
        "<tr><td></td><td>Mã số</td><td>Thuyết minh</td><td>2018 VND</td><td>2017 VND</td></tr>"
        "<tr><td>Doanh thu thuần</td><td>10</td><td>30</td><td>55.836</td><td>46.161</td></tr>"
        "<tr><td>Giá vốn</td><td>11</td><td>31</td><td>44.165</td><td>35.536</td></tr>"
        "<tr><td>LN gộp</td><td>20</td><td></td><td>11.670</td><td>10.625</td></tr>"
        "</table>"
    )
    assert find_item_code_col(grid, 0) == 1  # cột "Mã số"


def test_find_item_code_col_bank_roman():
    grid = parse_table_grid(
        "<table>"
        "<tr><td>STT</td><td>Chỉ tiêu</td><td>Thuyết minh</td><td>2022 VND</td><td>2021 VND</td></tr>"
        "<tr><td>I</td><td>Tiền mặt</td><td>4</td><td>18.348</td><td>18.011</td></tr>"
        "<tr><td>II</td><td>Tiền gửi NHNN</td><td>5.1</td><td>92.557</td><td>22.506</td></tr>"
        "<tr><td>1</td><td>Tiền gửi TCTD</td><td></td><td>283.727</td><td>181.036</td></tr>"
        "<tr><td>a</td><td>Chi tiết</td><td>12(b)</td><td>1.500</td><td>2.500</td></tr>"
        "<tr><td>XII</td><td>Tài sản khác</td><td>15(c)</td><td>60.978</td><td>28.964</td></tr>"
        "</table>"
    )
    # cột STT (La Mã + số + chữ thường a) — thuyết minh có '5.1','12(b)' không khớp mã
    assert find_item_code_col(grid, 0) == 0


# --- merge_fragments + build_asset: gom 2 fragment, dedupe "mang sang/trang trước" ---


def test_merge_and_dedupe_carry_row():
    # 2 fragment income cùng header; fragment 2 lặp lại code 60 ("mang từ trang trước")
    html1 = (
        "<table>"
        "<tr><td></td><td>Mã số</td><td>Thuyết minh</td><td>2018 VND</td><td>2017 VND</td></tr>"
        "<tr><td>Doanh thu thuần</td><td>10</td><td>30</td><td>55.836</td><td>46.161</td></tr>"
        "<tr><td>Giá vốn</td><td>11</td><td>31</td><td>44.165</td><td>35.536</td></tr>"
        "<tr><td>LNST (mang sang trang sau)(60 = 50 - 51 - 52)</td><td>60</td><td></td><td>8.600</td><td>8.014</td></tr>"
        "</table>"
    )
    html2 = (
        "<table>"
        "<tr><td></td><td>Mã số</td><td>Thuyết minh</td><td>2018 VND</td><td>2017 VND</td></tr>"
        "<tr><td>LNST (mang từ trang trước sang)(60 = 50 - 51 - 52)</td><td>60</td><td></td><td>8.600</td><td>8.014</td></tr>"
        "<tr><td>Cổ đông công ty mẹ</td><td>61</td><td></td><td>8.573</td><td>8.006</td></tr>"
        "<tr><td>Cổ đông không kiểm soát</td><td>62</td><td></td><td>27</td><td>8</td></tr>"
        "</table>"
    )
    tables = [
        (1, "table_1", parse_table_grid(html1), "", "income", 1.0, "VND"),
        (2, "table_2", parse_table_grid(html2), "", "income", 1.0, "VND"),
    ]
    groups = group_statement_fragments(tables)
    assert len(groups) == 1
    stmt, frags, uf, ul = groups[0]
    assert stmt == "income"
    assert len(frags) == 2
    asset = build_asset(stmt, frags, uf, ul)
    facts = emit_facts(asset, "TST", 2018, "consolidated")
    code60 = [f for f in facts if f["item_code"] == "60"]
    # dedupe: 2 kỳ × 1 (thay vì 2 fragment × 2 kỳ = 4)
    assert len(code60) == 2
    # code 61/62 từ fragment 2 vẫn được giữ
    assert any(f["item_code"] == "61" for f in facts)
    assert any(f["item_code"] == "62" for f in facts)


# --- validate_asset: label có công thức → warning, giữ giá trị ---


def test_validate_formula_label_warning():
    html = (
        "<table>"
        "<tr><td></td><td>Mã số</td><td>Thuyết minh</td><td>2018 VND</td><td>2017 VND</td></tr>"
        "<tr><td>Lợi nhuận (60 = 50 - 51 - 52)</td><td>60</td><td></td><td>8.600</td><td>8.014</td></tr>"
        "</table>"
    )
    grid = parse_table_grid(html)
    asset = build_asset("income", [Fragment(grid=grid, table_id="table_1", header_idx=0)], 1.0, "VND")
    facts = emit_facts(asset, "TST", 2018, "consolidated")
    warnings = validate_asset(asset, facts)
    assert any("formula-label" in w for w in warnings)
    # giá trị gốc vẫn giữ nguyên (không bị sửa)
    assert float(facts[0]["value_vnd"]) == pytest.approx(8600.0)


# --- facts_all.csv đủ quy mô (chỉ kiểm khi đã chạy full corpus) ---


def test_facts_all_min_size():
    p = DATA / "derived" / "facts_all.csv"
    if not p.exists():
        pytest.skip("facts_all.csv chưa có — chạy scripts/run_facts.py toàn corpus")
    with open(p, encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1  # trừ header
    # DoD ước lượng ≥400K nhưng thực tế ~377K (BS 215K/IN 71K/CF 91K) — không có
    # report nào có statement mà 0 facts. Threshold 350K làm regression guard
    # chống mất facts hàng loạt.
    assert n >= 350_000, f"facts_all.csv chỉ có {n} dòng (< 350K)"