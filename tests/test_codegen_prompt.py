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
        "var": var,
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
    assert "pd.read_csv" in sys_text and "không" in sys_text.lower()
    assert "result" in sys_text
    assert "```python" in sys_text  # few-shot
    assert 'dfs["df1"]' in sys_text or "dfs[\"df1\"]" in sys_text  # hợp đồng grader


def test_build_messages_user_has_question_and_schema() -> None:
    q = "Lãi tiền gửi năm 2018 của VJC là bao nhiêu triệu đồng?"
    msgs = build_messages(q, _entities(), [_card()])
    user = msgs[1]["content"]
    assert q in user
    assert "2018" in user  # năm
    assert "df1" in user  # biến bảng
    assert "Lãi tiền gửi" in user  # fact hint
    assert "208253201298" in user  # sample row (value tidy)


def test_build_messages_empty_cards() -> None:
    msgs = build_messages("xyz?", _entities(), [])
    assert "không có bảng" in msgs[1]["content"]