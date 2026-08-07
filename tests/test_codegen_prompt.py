"""test_codegen_prompt.py — build_messages dựng đúng schema/facts."""

from __future__ import annotations

from vifinqa.codegen import build_messages
from vifinqa.retrieval.entity import Entities


def _entities() -> Entities:
    return Entities(
        tickers=frozenset({"VJC"}),
        years=frozenset({2018}),
        year_ranges=(),
        report_type="separate",
        statement="income",
        unit_factor=1e6,
        unit_label="trieu dong",
        matched_names={"VJC": "vjc"},
    )


def _card(var: int = 1) -> dict:
    return {
        "table_ref": "VJC_financial_statements_2018_separate|table_50",
        "report_id": "VJC_financial_statements_2018_separate",
        "position": 50,
        "statement": "",
        "unit": "VND",
        "unit_factor": 1.0,
        "columns": ["chi_tieu", "Mãsố", "ky", "value"],
        "fact_hints": [("50", "Lãi tiền gửi")],
        "sample_rows": "Lãi tiền gửi |  | 2018 | 208253201298.0\nLãi tiền gửi |  | 2017 | 190000000000.0",
    }


def test_build_messages_structure() -> None:
    msgs = build_messages(
        "Lãi tiền gửi năm 2018 của VJC là bao nhiêu triệu đồng?",
        _entities(),
        [_card()],
    )
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    sys_text = msgs[0]["content"]
    # schema tidy + quy tắc cốt lõi có mặt
    assert "chi_tieu" in sys_text
    assert "Mãsố" in sys_text
    assert "ky" in sys_text
    assert "value" in sys_text
    assert "round" in sys_text  # round 2 decimals
    assert "```python" in sys_text  # few-shot
    # contract grader (bare variables): nhắc df1 + CẤM dfs["..."]
    assert "df1" in sys_text
    assert 'dfs["' not in sys_text or "CẤM" in sys_text


def test_build_messages_user_single_table_uses_df() -> None:
    q = "Lãi tiền gửi năm 2018 của VJC là bao nhiêu triệu đồng?"
    msgs = build_messages(q, _entities(), [_card()])
    user = msgs[1]["content"]
    assert q in user
    assert "2018" in user  # năm
    # bare variables: 1 bảng → dùng `df1`, CẤM dfs["..."]
    assert "df1" in user
    assert "CẤM" in user and "dfs" in user  # có quy tắc cấm dfs
    assert "Lãi tiền gửi" in user  # fact hint
    assert "208253201298" in user  # sample row (value tidy)


def test_build_messages_user_multi_table_uses_table_ref() -> None:
    q = "So sánh LNST 2023 của HPG và VCB?"
    card2 = _card()
    card2b = {
        "table_ref": "VCB_financial_statements_2023_consolidated|table_11",
        "report_id": "VCB_financial_statements_2023_consolidated",
        "position": 11,
        "statement": "income",
        "unit": "VND",
        "unit_factor": 1.0,
        "columns": ["chi_tieu", "Mãsố", "ky", "value"],
        "fact_hints": [],
        "sample_rows": "",
    }
    msgs = build_messages(q, _entities(), [card2, card2b])
    user = msgs[1]["content"]
    # N bảng → liệt kê bare variables df1, df2 + CẤM dfs["<table_ref>"]
    assert "df1" in user and "df2" in user
    assert "table_50" in user
    assert "table_11" in user
    assert "CẤM" in user and "dfs" in user


def test_build_messages_empty_cards() -> None:
    msgs = build_messages("xyz?", _entities(), [])
    assert "không có bảng" in msgs[1]["content"]