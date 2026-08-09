"""test_merged_evidence.py — merge facts → evidence bảng statement gộp."""

from __future__ import annotations

import pandas as pd

from vifinqa.etl.merged_evidence import merge_statement_facts


def _facts_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["statement", "item_code", "item_label", "period_key", "period_label", "value_vnd", "src_table_ids"]
    return pd.DataFrame(rows, columns=cols)


def test_cash_flow_fragments_merged():
    """CF tách 2 fragment (table_9 + table_10) → 1 bảng gộp chứa mã 20."""
    facts = _facts_df([
        {"statement": "cash_flow", "item_code": "01", "item_label": "loi nhuan truoc thue",
         "period_key": "flow_year", "period_label": "2017 VND",
         "value_vnd": 160075617870.0, "src_table_ids": "table_9"},
        {"statement": "cash_flow", "item_code": "20", "item_label": "luu chuyen tien thuan tu hoat dong kinh doanh",
         "period_key": "flow_year", "period_label": "2017 VND",
         "value_vnd": 145731366146.0, "src_table_ids": "table_9"},
        {"statement": "cash_flow", "item_code": "50", "item_label": "luu chuyen tien thuan trong nam",
         "period_key": "flow_year", "period_label": "2017 VND",
         "value_vnd": -88398966694.0, "src_table_ids": "table_10"},
    ])
    merged = merge_statement_facts(2017, facts)
    assert "cash_flow" in merged
    df, table_ids = merged["cash_flow"]
    assert table_ids == ["table_9", "table_10"]
    assert len(df) == 3
    row20 = df[df["Mãsố"] == "20"].iloc[0]
    assert row20["chi_tieu"] == "luu chuyen tien thuan tu hoat dong kinh doanh"
    assert float(row20["value"]) == 145731366146.0
    assert row20["ky"] == "2017"


def test_year_start_dropped():
    """Số dư đầu kỳ (year_start) bị bỏ — tránh trùng năm cuối kỳ."""
    facts = _facts_df([
        {"statement": "balance_sheet", "item_code": "100", "item_label": "tai san ngan han",
         "period_key": "year_end", "period_label": "31/12/2017",
         "value_vnd": 1000.0, "src_table_ids": "table_2"},
        {"statement": "balance_sheet", "item_code": "100", "item_label": "tai san ngan han",
         "period_key": "year_start", "period_label": "01/01/2017",
         "value_vnd": 500.0, "src_table_ids": "table_2"},
    ])
    merged = merge_statement_facts(2017, facts)
    df, _ = merged["balance_sheet"]
    assert len(df) == 1
    assert df.iloc[0]["ky"] == "2017"
    assert float(df.iloc[0]["value"]) == 1000.0


def test_ky_from_flow_label_vs_fallback():
    """flow_year: lấy năm trong label; label không chứa năm → năm báo cáo."""
    facts = _facts_df([
        {"statement": "income", "item_code": "10", "item_label": "doanh thu",
         "period_key": "flow_year", "period_label": "2016 VND",
         "value_vnd": 10.0, "src_table_ids": "table_4"},
        {"statement": "income", "item_code": "10", "item_label": "doanh thu",
         "period_key": "flow_year", "period_label": "Trong nam",
         "value_vnd": 20.0, "src_table_ids": "table_4"},
    ])
    merged = merge_statement_facts(2017, facts)
    df, _ = merged["income"]
    assert set(df["ky"]) == {"2016", "2017"}


def test_dedupe_same_code_and_ky():
    """Cùng (Mãsố, ky) xuất hiện 2 lần (restated_cur vs year_end) → giữ dòng đầu."""
    facts = _facts_df([
        {"statement": "balance_sheet", "item_code": "100", "item_label": "tai san",
         "period_key": "year_end", "period_label": "31/12/2017",
         "value_vnd": 1000.0, "src_table_ids": "table_2"},
        {"statement": "balance_sheet", "item_code": "100", "item_label": "tai san",
         "period_key": "restated_cur", "period_label": "31/12/2017",
         "value_vnd": 999.0, "src_table_ids": "table_2"},
    ])
    merged = merge_statement_facts(2017, facts)
    df, _ = merged["balance_sheet"]
    assert len(df) == 1
    assert float(df.iloc[0]["value"]) == 1000.0


def test_empty_label_and_noise_dropped():
    """Label rỗng (section title) và period_label noise "Quyết định" bị bỏ."""
    facts = _facts_df([
        {"statement": "balance_sheet", "item_code": "", "item_label": "",
         "period_key": "flow_year", "period_label": "2017 VND",
         "value_vnd": 5.0, "src_table_ids": "table_2"},
        {"statement": "balance_sheet", "item_code": "", "item_label": "phai thu khac",
         "period_key": "flow_year", "period_label": "Theo Quyết định 15/2006/QĐ-BTC",
         "value_vnd": 6.0, "src_table_ids": "table_2"},
        {"statement": "balance_sheet", "item_code": "130", "item_label": "phai thu khac",
         "period_key": "flow_year", "period_label": "2017 VND",
         "value_vnd": 7.0, "src_table_ids": "table_2"},
    ])
    merged = merge_statement_facts(2015, facts)
    df, _ = merged["balance_sheet"]
    assert len(df) == 1
    assert float(df.iloc[0]["value"]) == 7.0


def test_escaped_parens_cleaned():
    """Label escape `\(`/`\)` (formula-label) được làm sạch."""
    facts = _facts_df([
{"statement": "cash_flow", "item_code": "50",
         "item_label": "luu chuyen tien thuan trong nam\\( (50 = 20 + 30 + 40) \\)",
         "period_key": "flow_year", "period_label": "2017 VND",
         "value_vnd": -88.0, "src_table_ids": "table_10"},
    ])
    merged = merge_statement_facts(2017, facts)
    df, _ = merged["cash_flow"]
    assert "\\" not in df.iloc[0]["chi_tieu"]
    assert "(50 = 20 + 30 + 40)" in df.iloc[0]["chi_tieu"]


def test_no_statement_returns_empty():
    """Facts không có statement nào → dict rỗng."""
    facts = _facts_df([
        {"statement": "note", "item_code": "x", "item_label": "note",
         "period_key": "flow_year", "period_label": "2017",
         "value_vnd": 1.0, "src_table_ids": "table_50"},
    ])
    assert merge_statement_facts(2017, facts) == {}